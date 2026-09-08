from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
from loguru import logger
from sqlalchemy import text
from tqdm import tqdm

from DataCollection.CalendarManager import TradingCalendarAnalyzer


TABLE = "stock_daily_kline"
OVERLAP_DAYS = 15
BATCH_SIZE = 300          # 每批次处理 300 只
BATCH_INTERVAL = 10       # 批次间休息 10 秒
STAGGER_DELAY = 15        # 两管道错峰 15 秒
FETCH_WORKERS = 2         # 每管道并发取数线程数
_STOCK_FETCH_LOCK = threading.Semaphore(4)  # 两管道共 4 worker，各需 1 permit；原 Semaphore(2) 导致 B 管道饿死


class IncrementalSyncEngine:
    def __init__(
        self,
        db_engine: Any,
        default_start: str | None = None,
        cache_dir: str | None = None,
        enable_research_report_filter: bool = False,
        research_report_min_count: int = 1,
    ) -> None:
        self._engine = db_engine
        self._default_start = self.align_to_trading_day(default_start) if default_start else None
        self._enable_research_report_filter = enable_research_report_filter
        self._research_report_min_count = research_report_min_count
        self._cache_dir = cache_dir
        if not self._cache_dir:
            try:
                from UtilsManager.ConfigParser import Config
                self._cache_dir = os.path.join(Config().CACHE_DIRECTORY, "kline_batches")
            except Exception:
                self._cache_dir = os.path.join(
                    os.environ.get("TEMP", "/tmp"), "opencode", "kline_batches"
                )
        os.makedirs(self._cache_dir, exist_ok=True)
        try:
            _cal = TradingCalendarAnalyzer()
            # 北京时间：本机挂钟 → 本机实际 UTC 偏移 → UTC → UTC+8。
            # 本机时区非北京时间时，直接用本地时间会偏移数小时（如 UTC-7 机器）。
            _bj_now = datetime.now().astimezone().astimezone(timezone(timedelta(hours=8)))
            self._trade_date = self._expected_kline_date(_cal, _bj_now) or date.today()
        except Exception:
            # 兜底仍严格使用北京时间,避免非北京时区机器基准日错一天
            self._trade_date = datetime.now().astimezone().astimezone(timezone(timedelta(hours=8))).date()
        self._trade_date_str = self._trade_date.isoformat().replace("-", "")
        # 全局共享 Session，提升连接池匹配两管道各 2 worker（共 4 路并发）
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._cleanup_old_cache()

    @staticmethod
    def _expected_kline_date(cal: Any, now_bj: datetime) -> date | None:
        """K 线新鲜度基准日（回测/复盘共表去重下载的核心）。

        now_bj 须为北京时间（aware/naive 均可，仅用 date/hour/minute）。
        兜底逻辑：若日历在收盘前仍返回"今天"（旧版行为），回退到上一交易日；
        当前 TradingCalendarAnalyzer 已按北京时间 15:30 收盘阈值修正，该回退
        通常不再触发，仅作防御。
        """
        try:
            last_str = cal.get_last_trading_day()
            last = datetime.strptime(last_str, "%Y-%m-%d").date()
            if (
                last == now_bj.date()
                and (now_bj.hour < 15 or (now_bj.hour == 15 and now_bj.minute < 30))
            ):
                dates = sorted(cal.get_official_trading_dates())
                try:
                    idx = dates.index(last_str)
                except ValueError:
                    idx = -1
                if idx > 0:
                    last = datetime.strptime(dates[idx - 1], "%Y-%m-%d").date()
            return last
        except Exception:
            return None

    # ── public API ──────────────────────────────────────────────

    def _cleanup_old_cache(self) -> None:
        """清理超过 7 天的缓存文件,以及脏 close_normal_*.csv(日期不匹配当前交易日)."""
        try:
            now = datetime.now()
            today_tag = f"close_normal_{self._trade_date_str}.csv"
            for fname in os.listdir(self._cache_dir):
                # 清理脏 close_normal 缓存(日期与当前交易日不一致)
                if fname.startswith("close_normal_") and fname != today_tag:
                    os.remove(os.path.join(self._cache_dir, fname))
                    continue
                # 清理超过 7 天的旧缓存
                fpath = os.path.join(self._cache_dir, fname)
                if os.path.isfile(fpath):
                    age = now - datetime.fromtimestamp(os.path.getmtime(fpath))
                    if age.days > 7:
                        os.remove(fpath)
        except Exception as e:
            logger.warning(f"缓存清理失败: {e}")

    def sync_all(self, symbols_prefixed: list[str], force_start_iso: str | None = None) -> int:
        """同步股票日线数据。

        force_start_iso: 传入时忽略 stale 检查，强制从该日期起全量回填
        （用于指标预热历史补全；写入幂等，可安全覆盖已存在区间）。
        """
        try:
            remaining = self._get_stale_symbols(symbols_prefixed) if force_start_iso is None else list(symbols_prefixed)

            cached = self._load_failed_set()
            if cached:
                cached = self._drop_dead_symbols(cached)
                old_len = len(remaining)
                remaining = sorted(set(remaining) | cached)
                added = len(remaining) - old_len
                if added:
                    logger.info(f"加载 {len(cached)} 只缓存失败股票,待同步 {len(remaining)} 只(新增 {added} 只)")

            if not remaining:
                logger.info("所有股票已有最新交易日数据,无需同步")
                return 0

            start_iso = force_start_iso or self._calc_start_iso(remaining)
            # P2 审计修复：end_iso 不允许超过当前日期，防止请求未来数据导致腾讯 API SSL 失败
            _today = date.today().isoformat()
            end_iso = min(self._trade_date.isoformat(), _today)
            logger.info(f"同步 {len(remaining)} 只, {start_iso} ~ {end_iso}" + ("（强制回填）" if force_start_iso else ""))

            mid = len(remaining) // 2
            half_a = remaining[:mid]
            half_b = remaining[mid:]
            logger.info(f"双管道: 前半段 {len(half_a)} 只 | 后半段 {len(half_b)} 只")

            import threading
            results: dict = {"a": [0, []], "b": [0, []]}

            def run(label: str, half: list[str]):
                ins, fails = self._run_pipeline(
                    half, start_iso, end_iso, label=label, force=force_start_iso is not None
                )
                results[label.lower()] = [ins, fails]

            ta = threading.Thread(target=run, args=("A", half_a), daemon=True)
            tb = threading.Thread(target=run, args=("B", half_b), daemon=True)

            ta.start()
            logger.info(f"管道 A 启动,{STAGGER_DELAY}s 后启动管道 B")
            time.sleep(STAGGER_DELAY)
            tb.start()

            ta.join()
            tb.join()

            all_failures = results["a"][1] + results["b"][1]
            total = results["a"][0] + results["b"][0]
            self._save_failed_set(set(all_failures))
            logger.info(f"同步完成,总写入 {total} 行(失败 {len(all_failures)} 只)")
            return total
        except BaseException as e:
            import traceback
            logger.error(f"sync_all 发生致命错误: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            raise

    def _calc_start_iso(self, symbols: list[str]) -> str:
        min_latest = self._get_min_latest_date(symbols)
        if min_latest is None:
            return (
                datetime.strptime(self._default_start, "%Y%m%d").strftime("%Y-%m-%d")
                if self._default_start else "2000-01-01"
            )
        return (min_latest - timedelta(days=OVERLAP_DAYS + 1)).isoformat()

    # ── Dual Pipeline ───────────────────────────────────────────

    def _run_pipeline(self, symbols: list[str], start_iso: str, end_iso: str, label: str = "", force: bool = False) -> tuple[int, list[str]]:
        start = start_iso.replace("-", "")
        end = end_iso.replace("-", "")
        total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE
        inserted = 0
        all_failures: list[str] = []

        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            batch_no = i // BATCH_SIZE + 1
            desc = f"  P{label} batch {batch_no}/{total_batches}"
            logger.info(f"管道{label} {desc}: {len(batch)} 只")
            count, failures = self._process_batch(batch, start, end, desc=desc, force=force)
            all_failures.extend(failures)
            inserted += count
            if i + BATCH_SIZE < len(symbols):
                time.sleep(BATCH_INTERVAL)

        logger.info(f"管道{label} 完成,写入 {inserted} 行(失败 {len(all_failures)} 只)")
        return inserted, all_failures

    def _batch_get_overlap_history(self, symbols: list[str], start: str) -> dict[str, dict[str, tuple[float | None, float | None]]]:
        """批量获取每只股票重叠区 [start, ∞) 的 DB 历史,一次查询。

        返回 {symbol: {trade_date_iso: (close_normal, adj_factor)}}，
        供增量判定对同一天做新旧对比（旧值=DB 已存值，新值=本次拉取值）。
        """
        if not symbols:
            return {}
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT symbol, trade_date::date AS d, close_normal, adj_factor
                    FROM {TABLE}
                    WHERE symbol = ANY(:symbols) AND trade_date::date >= :start
                """),
                {"symbols": symbols, "start": datetime.strptime(start, "%Y%m%d").date()},
            ).fetchall()
        out: dict[str, dict[str, tuple[float | None, float | None]]] = {}
        for sym, d, cn, adj in rows:
            out.setdefault(sym, {})[d.isoformat()] = (
                float(cn) if cn is not None else None,
                float(adj) if adj is not None else None,
            )
        return out

    def _batch_get_last_date(self, symbols: list[str]) -> dict[str, date | None]:
        """批量获取每股 DB 内最新交易日(全区间),一次查询,用于停牌/缺口预分类."""
        if not symbols:
            return {}
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT symbol, MAX(trade_date::date) AS latest
                    FROM {TABLE}
                    WHERE symbol = ANY(:symbols)
                    GROUP BY symbol
                """),
                {"symbols": symbols},
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    @staticmethod
    def _safe_num(v: Any, default: float | None = None) -> float | None:
        """尽可能将腾讯返回字段转为有限浮点数;格式异常返回 default."""
        try:
            f = float(v)
            return f if math.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    TX_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"

    def _fetch_one_stock(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        """获取一只股票不复权 + 后复权数据,返回合并 DataFrame."""
        # Windows SSL 并发必崩 0xC0000005，全局锁确保同一时刻只发一个请求
        with _STOCK_FETCH_LOCK:
            return self._do_fetch_one_stock(symbol, start, end)

    def _do_fetch_one_stock(self, symbol: str, start: str, end: str) -> pd.DataFrame | None:
        start_year = int(start[:4])
        end_year = int(end[:4])
        # P2 审计修复：跳过未来年份，腾讯 API 对不存在数据的未来年份返回 SSL 错误
        _current_year = date.today().year
        if start_year > _current_year:
            logger.warning(f"腾讯API {symbol} 同步起始年份 {start_year} 超过当前年份 {_current_year}，跳过")
            return None
        end_year = min(end_year, _current_year)

        # SSL 降级标志：本股票首次遇到 SSL 失败后，后续请求降级 verify=False
        _ssl_degraded = [False]

        def _tx_raw(adjust: str) -> list[list]:
            rows: list[list] = []
            for year in range(start_year, end_year + 1):
                # 再次确认：不请求未来年份
                if year > _current_year:
                    continue
                key = "hfqday" if adjust == "hfq" else "day"
                param = f"{symbol},day,{year}-01-01,{year+1}-12-31,640,{adjust}"
                var = f"kline_day{adjust}{year}"
                # P2 审计修复：指数退避重试 — 所有异常统一退避，重试次数 5，per-symbol 速率限制
                for attempt in range(5):
                    try:
                        # P2 SSL 容错降级：首次失败后使用 verify=False 重试
                        _verify = True if not _ssl_degraded[0] else False
                        r = self._session.get(
                            self.TX_URL,
                            params={"_var": var, "param": param, "r": "0.8205"},
                            timeout=15,
                            verify=_verify,
                        )
                        if r.status_code == 429:
                            # 速率限制：指数退避 + 抖动
                            _backoff = min(2 ** attempt + random.uniform(0, 1), 30)
                            logger.warning(
                                f"腾讯API 429: {symbol} {adjust} {year}，等待 {_backoff:.1f}s 后重试"
                            )
                            time.sleep(_backoff)
                            continue
                        r.raise_for_status()
                        if "={" not in r.text:
                            # 结构异常（无 JSON 载荷）：视为该年无数据，不消耗重试
                            time.sleep(0.5)
                            break
                        data = json.loads(r.text[r.text.find("={") + 1:])
                        # P2 审计修复：退市股/无数据年份，腾讯返回 data=[] 等非标准结构，
                        # 按 data["data"][symbol] 取数会抛 TypeError 并烧掉 5 次指数退避重试；
                        # 结构异常按"该年无数据"优雅跳过（继续下一年的正常年份数据）。
                        node = data.get("data") if isinstance(data, dict) else None
                        if not isinstance(node, dict):
                            time.sleep(0.5)
                            break
                        sub = node.get(symbol)
                        if not isinstance(sub, dict):
                            # 该股该年无 K 线（如退市早于该年），跳过本年份
                            time.sleep(0.5)
                            break
                        rows.extend(sub.get(key, []))
                        # P2 速率限制：每只股票请求间隔 ≥ 0.5s
                        time.sleep(0.5)
                        break
                    except Exception as e:
                        # P2 SSL 容错：如果是 SSL 错误且未降级，标记降级后重试
                        if "SSL" in type(e).__name__ and not _ssl_degraded[0]:
                            _ssl_degraded[0] = True
                            logger.warning(
                                f"腾讯API {symbol} SSL 验证失败（自签名证书），降级为 verify=False 重试"
                            )
                            continue  # 立即重试，不消耗 attempt 计时
                        if attempt < 4:
                            # P2 指数退避：base_delay=2s, max=30s, 加 0~1s 抖动防雷群效应
                            _backoff = min(2 ** attempt + random.uniform(0, 1), 30)
                            logger.warning(
                                f"腾讯API {symbol} {adjust} {year} 异常: {type(e).__name__}，"
                                f"等待 {_backoff:.1f}s 后第 {attempt+2} 次重试"
                            )
                            time.sleep(_backoff)
                        else:
                            logger.error(
                                f"腾讯API {symbol} {adjust} 第{year}年 5次重试均失败: {type(e).__name__}: {e}"
                            )
                            return None
            return rows

        raw_rows = _tx_raw("")
        hfq_rows = _tx_raw("hfq")
        if raw_rows and hfq_rows:
            return self._build_qq_df(symbol, start, end, raw_rows, hfq_rows)

        # 腾讯对部分股票(次新股/新上市)不提供后复权数据,响应中只有 day 无 hfqday。
        # 优先用 asharehub 复权因子重建后复权价(raw × factor)；
        # 因子不可用或全为 1.0(从未除权,后复权=不复权)时降级为不复权写入。
        if raw_rows:
            factor_map = self._fetch_ah_factor_map(symbol, start, end)
            if factor_map:
                non_one = [d for d, f in factor_map.items() if abs(f - 1.0) > 1e-9]
                if non_one:
                    logger.warning(
                        f"腾讯API {symbol} 无后复权数据,使用 asharehub 复权因子重建({len(non_one)} 个除权因子)"
                    )
                    return self._build_qq_df_from_factor(symbol, start, end, raw_rows, factor_map)
            logger.warning(
                f"腾讯API {symbol} 无后复权数据,降级为不复权写入(adj_factor=1.0,asharehub 因子全为1或不可用)"
            )
            return self._build_qq_df(symbol, start, end, raw_rows, raw_rows)

        # 腾讯API 失败时直接返回 None，由批处理器标记为失败股票，下次重试
        logger.warning(f"腾讯API {symbol} 返回空,标记为失败，将在下次同步时重试")
        return None

    def _fetch_ah_factor_map(self, symbol: str, start: str, end: str) -> dict[str, float] | None:
        """从 asharehub 拉取复权因子 {YYYYMMDD: factor}（累计因子,后复权价=未复权价×因子）。

        P2（审计）：trade_date 键统一归一化为 YYYYMMDD——API 可能返回
        YYYY-MM-DD 字符串 / date / datetime（含 pandas Timestamp）对象；旧实现
        直接 str() 作 key，回退路径（_build_qq_df_from_factor 用 d_compact=YYYYMMDD
        查表）全部 miss → 每股被静默跳过（仅"跳过 N 天"告警）。格式探测：
        无法解析为 YYYYMMDD 的行告警并跳过。
        """
        try:
            from UtilsManager.ConfigParser import Config
            from UtilsManager.AShareHubClient import make_asharehub_client
            cfg = Config()
            key = getattr(cfg, "ASHAREHUB_API_KEY", "") or ""
            if not key:
                return None
            code = symbol[2:]
            market = symbol[:2]
            suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(market, "SH")
            client = make_asharehub_client(api_key=key)
            df = client.adj_factor(
                symbol=f"{code}.{suffix}",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
            )
            if df is None or df.empty:
                return None
            out: dict[str, float] = {}
            bad = 0
            for _, r in df.iterrows():
                td = r["trade_date"]
                if isinstance(td, (datetime, date)):
                    key_n = td.strftime("%Y%m%d")
                else:
                    s = str(td).strip()
                    key_n = s[:10].replace("-", "") if len(s) >= 10 and s[4] == "-" else s.replace("-", "").replace("/", "")
                if len(key_n) != 8 or not key_n.isdigit():
                    bad += 1
                    continue
                out[key_n] = float(r["adj_factor"])
            if bad:
                logger.warning(
                    f"asharehub 复权因子 {symbol} 跳过 {bad}/{len(df)} 行："
                    f"trade_date 格式异常（期望 YYYYMMDD/YYYY-MM-DD/date 对象）"
                )
            return out or None
        except Exception as e:
            logger.warning(f"asharehub 复权因子获取失败 {symbol}: {type(e).__name__}: {e}")
            return None

    def _build_qq_df_from_factor(self, symbol: str, start: str, end: str,
                                 raw_rows: list[list], factor_map: dict[str, float]) -> pd.DataFrame | None:
        """用不复权行情 × asharehub 复权因子重建 DataFrame（含除权日连续价格）。

        P0-12 审计修复（复权语义统一）：
        open/close/high/low = 不复权原始价（交易所真实成交价，涨跌停/成交/估值用）；
        open_normal/close_normal/high_normal/low_normal = 后复权价（原始价 × 累计因子，
        信号/止损/指标用）；adj_factor = 累计复权因子（后复权价 ÷ 原始价）。
        """
        raw_map = {str(r[0]): r for r in raw_rows}
        filtered = [d for d in sorted(raw_map) if start <= d.replace("-", "") <= end]
        if not filtered:
            return None
        out: dict[str, list] = {
            "symbol": [], "trade_date": [], "open": [], "close": [],
            "high": [], "low": [], "open_normal": [], "close_normal": [],
            "high_normal": [], "low_normal": [],
            "volume": [], "amount": [], "adj_factor": [],
        }
        skipped = 0
        for d in filtered:
            raw = raw_map[d]
            d_compact = d.replace("-", "")
            factor = factor_map.get(d_compact)
            if factor is None or not (math.isfinite(factor) and factor > 0):
                skipped += 1
                continue
            close_raw = float(raw[2])
            if not (math.isfinite(close_raw) and close_raw > 0):
                skipped += 1
                continue
            out["symbol"].append(symbol)
            out["trade_date"].append(d)
            out["open"].append(float(raw[1]))
            out["close"].append(close_raw)
            out["high"].append(float(raw[3]))
            out["low"].append(float(raw[4]))
            out["open_normal"].append(float(raw[1]) * factor)
            out["close_normal"].append(close_raw * factor)
            out["high_normal"].append(float(raw[3]) * factor)
            out["low_normal"].append(float(raw[4]) * factor)
            vol = self._safe_num(raw[5] if len(raw) > 5 else None)
            if vol is None or vol < 0:
                skipped += 1
                logger.warning(f"asharehub 因子重建 {symbol} 丢弃交易日 {d}: 成交量异常(vol={raw[5] if len(raw) > 5 else None})")
                continue
            out["volume"].append(int(vol * 100))
            amt = self._safe_num(raw[8] if len(raw) > 8 else None)
            out["amount"].append((amt * 10000) if amt is not None else 0.0)
            out["adj_factor"].append(factor)
        if skipped:
            logger.warning(f"asharehub 因子重建 {symbol} 跳过 {skipped}/{len(filtered)} 天(因子缺失)")
        return pd.DataFrame(out)

    def _build_qq_df(self, symbol: str, start: str, end: str,
                     raw_rows: list[list], hfq_rows: list[list]) -> pd.DataFrame | None:
        """原始价 + 后复权价合并重建 DataFrame。

        P0-12 审计修复（复权语义统一）：open/close/high/low = 不复权原始价；
        open_normal/close_normal/high_normal/low_normal = 后复权价；
        adj_factor = 后复权收盘 ÷ 原始收盘（累计因子）。
        """
        raw_map = {str(r[0]): r for r in raw_rows}
        hfq_map = {str(h[0]): h for h in hfq_rows}
        common = sorted(set(raw_map) & set(hfq_map))
        filtered = [d for d in common if start <= d.replace("-", "") <= end]
        if not filtered:
            return None
        out: dict[str, list] = {
            "symbol": [], "trade_date": [], "open": [], "close": [],
            "high": [], "low": [], "open_normal": [], "close_normal": [],
            "high_normal": [], "low_normal": [],
            "volume": [], "amount": [], "adj_factor": [],
        }
        skipped = 0
        for d in filtered:
            raw = raw_map[d]
            hfq = hfq_map[d]
            close_raw = float(raw[2])
            close_hfq = float(hfq[2])
            if not (math.isfinite(close_raw) and close_raw > 0):
                # 原始价异常：无法重建，丢弃
                skipped += 1
                logger.warning(f"腾讯API {symbol} 丢弃交易日 {d}：原始收盘价异常(close_raw={close_raw})")
                continue
            # P1-8/P1-9 修复：腾讯hfq短暂异常（如负复权价）不丢弃整行
            # 使用上一个有效交易日的 adj_factor 进行降级重建
            adj_factor = 1.0
            use_hfq = math.isfinite(close_hfq) and close_hfq > 0
            if use_hfq:
                adj_factor = close_hfq / close_raw
            else:
                # hfq 异常：向前查找上一个有效的 adj_factor
                use_hfq = False
                # 从已构建的行中取最后一个有效 adj_factor
                _prev_factor = out["adj_factor"][-1] if out["adj_factor"] else 1.0
                if math.isfinite(_prev_factor) and _prev_factor > 0:
                    adj_factor = _prev_factor
                else:
                    # 极端情况：第一条记录就异常，使用 1.0
                    adj_factor = 1.0
                logger.warning(
                    f"腾讯API {symbol} {d} hfq收盘价异常(close_hfq={close_hfq})，"
                    f"使用上一交易日adj_factor={adj_factor:.6f}降级重建"
                )

            out["symbol"].append(symbol)
            out["trade_date"].append(d)
            out["open"].append(float(raw[1]))
            out["close"].append(close_raw)
            out["high"].append(float(raw[3]))
            out["low"].append(float(raw[4]))
            if use_hfq:
                out["open_normal"].append(float(hfq[1]))
                out["close_normal"].append(close_hfq)
                out["high_normal"].append(float(hfq[3]))
                out["low_normal"].append(float(hfq[4]))
            else:
                # P1-8 降级：hfq不可用，用 raw × adj_factor 重建
                out["open_normal"].append(float(raw[1]) * adj_factor)
                out["close_normal"].append(close_raw * adj_factor)
                out["high_normal"].append(float(raw[3]) * adj_factor)
                out["low_normal"].append(float(raw[4]) * adj_factor)
            vol = self._safe_num(raw[5] if len(raw) > 5 else None)
            if vol is None or vol < 0:
                # 成交量缺失/异常:无法重建,丢弃该日
                skipped += 1
                logger.warning(f"腾讯API {symbol} 丢弃交易日 {d}: 成交量异常(vol={raw[5] if len(raw) > 5 else None})")
                continue
            out["volume"].append(int(vol * 100))
            amt = self._safe_num(raw[8] if len(raw) > 8 else None)
            out["amount"].append((amt * 10000) if amt is not None else 0.0)
            out["adj_factor"].append(adj_factor)
        if skipped:
            logger.warning(f"腾讯API {symbol} 丢弃 {skipped}/{len(filtered)} 个非法价格交易日(close<=0 或 NaN)")
        return pd.DataFrame(out)

    def _detect_overlap_correction(self, db_rows: dict[str, tuple[float | None, float | None]],
                                   grp: pd.DataFrame) -> tuple[bool, str, str]:
        """重叠区同日期新旧对比，判断是否触发全量重拉（P0-13）。

        逐日对齐 DB 与本次拉取的同一天值，统一 0.1% 容差判定：
          - adj_factor 同日期 ratio ∉ [0.999, 1.001] → 除权/因子修正
          - close_normal 同日期相对偏差 > 0.001      → 数据源修正历史
        加滞回：需 ≥2 个不同交易日均不一致才判定重拉，过滤数据源瞬时抖动/单日修正。
        返回 (是否重拉, 原因, DB 最新交易日 ISO)。
        """
        latest_iso = max(db_rows)
        grp_by_date = grp.set_index("trade_date")
        _TOL = 0.001  # 0.1% 统一容差
        adj_bad: set[str] = set()
        cn_bad: set[str] = set()
        for d, (db_cn, db_adj) in db_rows.items():
            if d not in grp_by_date.index:
                continue
            row = grp_by_date.loc[d]
            new_adj = row.get("adj_factor")
            if new_adj is not None and db_adj is not None and db_adj != 0:
                ratio = new_adj / db_adj
                if ratio > 1.0 + _TOL or ratio < 1.0 - _TOL:
                    adj_bad.add(d)
            new_cn = row.get("close_normal")
            if new_cn is not None and db_cn is not None and db_cn != 0:
                dev = abs(new_cn / db_cn - 1.0)
                if dev > _TOL:
                    cn_bad.add(d)
        bad_dates = adj_bad | cn_bad
        if len(bad_dates) >= 2:
            return True, (f"重叠区 {len(bad_dates)} 天不一致"
                          f"(adj_factor {len(adj_bad)} 天, close_normal {len(cn_bad)} 天)"), latest_iso
        return False, "overlap stable", latest_iso

    def _process_batch(self, symbols: list[str], start: str, end: str, desc: str = "", force: bool = False) -> tuple[int, list[str]]:
        """并发取窗口数据 → 批量 DB 查询 → 内存判断 → 一次写入.

        force=True 时忽略 DB 已有数据/除权检测,将拉取区间全量覆盖写入
        （幂等 upsert,用于指标预热历史回填;否则已有数据只写增量,
        早期历史会被丢弃导致回填永不生效）。
        """
        # Step 1: 预分类 + 并发取窗口数据
        # 1a. 预查每股 DB 最新交易日,识别窗口内无成交的疑似停牌/长空窗股：
        #     统一窗口对它们没有可拉数据,不该进并发拉取池(浪费请求/污染失败缓存);
        #     改为按缺口区间 [last+1, end] 单独补拉,复牌或漏采恢复时自动补全。
        if force:
            fetch_syms: list[str] = list(symbols)
            fill_candidates: dict[str, tuple[date, date]] = {}
        else:
            start_date = datetime.strptime(start, "%Y%m%d").date()
            end_date = datetime.strptime(end, "%Y%m%d").date()
            last_map = self._batch_get_last_date(symbols)
            fill_candidates = {}
            fetch_syms = []
            for sym in symbols:
                ld = last_map.get(sym)
                if ld is not None and ld < start_date and (end_date - ld).days > OVERLAP_DAYS:
                    fill_candidates[sym] = (ld + timedelta(days=1), end_date)
                else:
                    fetch_syms.append(sym)

        all_data: dict[str, pd.DataFrame] = {}
        failed: list[str] = []
        # position 区分管道: A=0, B=1, 防止 tqdm 互相覆盖
        _pos = 0 if 'A' in desc else (1 if 'B' in desc else None)
        with tqdm(total=len(fetch_syms), desc=desc, unit="stk", leave=False, position=_pos) as pbar:
            with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
                futures = {ex.submit(self._fetch_one_stock, sym, start, end): sym for sym in fetch_syms}
                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        df = future.result()
                        if df is not None:
                            all_data[sym] = df
                        else:
                            failed.append(sym)
                    except Exception as e:
                        logger.error(f"  [{sym}] fetch 异常: {type(e).__name__}: {e}")
                        failed.append(sym)
                    pbar.update(1)

        # 1b. 疑似停牌/长空窗股缺口补拉 [last+1, end];成功即与 DB 无重叠,可直接写
        if fill_candidates:
            with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
                futures = {
                    ex.submit(self._fetch_one_stock, sym, s.strftime("%Y%m%d"), e.strftime("%Y%m%d")): sym
                    for sym, (s, e) in fill_candidates.items()
                }
                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        df = future.result()
                        if df is not None and not df.empty:
                            all_data[sym] = df
                            logger.info(f"  [{sym}] 缺口补拉 {fill_candidates[sym][0]} ~ {end} 成功,恢复 {len(df)} 行")
                        else:
                            logger.info(f"  [{sym}] 窗口内无成交(最近交易 {last_map.get(sym)}),判定停牌跳过")
                    except Exception as e:
                        logger.error(f"  [{sym}] 缺口补拉异常: {type(e).__name__}: {e}")

        if not force:
            # 1c. 区分真失败与"窗口内无成交"(停牌)：
            #     窗口内应有数据却拉取失败,或 DB 无历史且拉不到 → 真失败,留待下次重试;
            #     最近交易日早在窗口 start 之前 → 停牌,不标记失败,避免失败缓存被停牌股污染。
            final_failed: list[str] = []
            for sym in failed:
                ld = last_map.get(sym)
                if ld is None or ld >= start_date:
                    final_failed.append(sym)
                else:
                    logger.info(f"  [{sym}] 窗口内无成交(最近交易 {ld}),跳过(不标记失败)")
            failed = final_failed

        if not all_data:
            logger.info(f"  {desc} 完成: 成功 0 只, 失败 {len(failed)} 只")
            return 0, failed

        # Step 2: bulk DB reads (force 模式全量覆盖,无需读取已有状态)
        syms = list(all_data.keys())
        db_map = self._batch_get_overlap_history(syms, start) if not force else {}

        # Step 3: identify split stocks（P0-13 同日期新旧对比判定）
        to_write: list[pd.DataFrame] = []
        written_symbols: set[str] = set()
        split_syms: list[str] = []

        for sym in syms:
            grp = all_data[sym].sort_values("trade_date")
            if force:
                # 强制回填:全量覆盖拉取区间,跳过增量/除权检测
                to_write.append(grp)
                written_symbols.add(sym)
                continue
            db_rows = db_map.get(sym)
            if not db_rows or not any(cn is not None for cn, _ in db_rows.values()):
                # 首次拉取或 DB 缺少 close_normal:全量写入
                to_write.append(grp)
                written_symbols.add(sym)
                continue

            should_repull, reason, latest_iso = self._detect_overlap_correction(db_rows, grp)
            if should_repull:
                logger.warning(f"  [{sym}] {reason}, 触发全量重拉")
                split_syms.append(sym)
                continue

            new = grp[grp["trade_date"] > latest_iso]
            if not new.empty:
                to_write.append(new)
                written_symbols.add(sym)

        # Step 4: concurrent full-history fetch for split stocks
        if split_syms:
            full_start = self._default_start.replace("-", "") if self._default_start else "20190101"
            full_end = self._trade_date.strftime("%Y%m%d")
            logger.info(f"  {len(split_syms)} 只触发重拉(除权/数据源修正),并发拉取全量历史")
            with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as ex:
                futures = {ex.submit(self._fetch_one_stock, sym, full_start, full_end): sym for sym in split_syms}
                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        full = future.result()
                        if full is not None and not full.empty:
                            to_write.append(full)
                            written_symbols.add(sym)
                        else:
                            to_write.append(all_data[sym])
                            written_symbols.add(sym)
                    except Exception as e:
                        logger.error(f"  [split-refetch {sym}] 异常: {type(e).__name__}: {e}")
                        to_write.append(all_data[sym])
                        written_symbols.add(sym)

        # Step 5: write all at once
        if to_write:
            final = pd.concat(to_write, ignore_index=True)
            self._write_batch(final)

        written_cnt = len(written_symbols)
        logger.info(f"  {desc} 完成: 成功 {written_cnt} 只, 失败 {len(failed)} 只")
        return written_cnt, failed

    # ── stock pool (merged from StockSyncEngine) ──────────────

    def get_stock_pool_from_db(self) -> pd.DataFrame:
        query = """
            SELECT stock_code AS ts_code, stock_code AS 股票代码,
                   stock_name AS name, industry_name AS industry
            FROM stock_basic_info_sw ORDER BY stock_code
        """
        with self._engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        def _strip(s: str) -> str:
            s = str(s)
            for pfx in ("sh", "sz", "bj"):
                if s.startswith(pfx):
                    s = s[len(pfx):]
                    break
            return s.zfill(6)
        if "股票代码" in df.columns:
            df["股票代码"] = df["股票代码"].apply(_strip)
        df["ts_code"] = df["ts_code"].apply(_strip)
        for col in ("ts_code", "name", "industry", "股票代码"):
            if col not in df.columns:
                df[col] = "N/A"
        return df[["ts_code", "name", "industry", "股票代码"]]

    @staticmethod
    def filter_st_stocks(df: pd.DataFrame) -> pd.DataFrame:
        if "name" not in df.columns:
            return df
        pattern = r"(?:\s*(?:\*|★|※|•|·))?(?:[Ss][Tt])|退市|IPO终止"
        return df[~df["name"].astype(str).str.contains(pattern, na=False)].copy()

    def filter_main_board(self, df: pd.DataFrame) -> pd.DataFrame:
        """仅保留沪深主板股票（60x/00x 开头）。

        系统仅覆盖沪深主板，创业板/科创板/北交所已从业务中剔除。
        """
        codes = df["股票代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True)
        return df[codes.str.match(r"^(60|00)", na=False)].copy()

    def _filter_by_research_report(self, pure_codes: set[str]) -> set[str]:
        if not self._enable_research_report_filter:
            return pure_codes
        try:
            import akshare as ak
            for attempt in range(3):
                try:
                    raw = ak.stock_profit_forecast_em()
                    break
                except Exception:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        return pure_codes
            if raw is None or raw.empty:
                return pure_codes
            df = raw.copy()
            if "代码" in df.columns and "股票代码" not in df.columns:
                df.rename(columns={"代码": "股票代码"}, inplace=True)
            df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
            for col in df.columns:
                if "买入" in col:
                    rating_col = col
                    break
            else:
                return pure_codes
            df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce").fillna(0)
            qualified = set(df.loc[df[rating_col] > self._research_report_min_count, "股票代码"].unique())
            before = len(pure_codes)
            pure_codes &= qualified
            logger.info(f"研报过滤: {before} → {len(pure_codes)}(买入>{self._research_report_min_count}次)")
        except Exception as e:
            logger.warning(f"研报过滤异常: {e},跳过研报过滤")
        return pure_codes

    def sync_stock_pool_and_kline(self, target_date: str | None = None) -> set[str]:
        from UtilsManager.CodeNormalizer import CodeNormalizer
        from UtilsManager.IDataProvider import backtest_lock_held

        # ── 回测数据隔离（P3.3 制度化）：回测进程持有会话级 advisory lock 时禁止写入 ──
        # 运行中改写 stock_daily_kline 会导致回测窗口数据漂移、信号缓存静默失效，
        # 且内容指纹可能在行数/日期不变时命中旧缓存。检测到锁即整体跳过本次同步。
        try:
            if backtest_lock_held(self._engine):
                logger.warning(
                    f"回测运行中（advisory lock {BACKTEST_ADVISORY_LOCK_KEY} 被占用），"
                    "跳过本次 K 线同步以避免污染回测数据"
                )
                return set()
        except Exception as e:
            logger.warning(f"回测隔离锁探测失败，视为空闲继续同步: {e}")

        if target_date is None:
            target_date = TradingCalendarAnalyzer().get_last_trading_day()
        today_tag = target_date.replace("-", "")

        pool = self.get_stock_pool_from_db()
        before = len(pool)
        pool = self.filter_st_stocks(pool)
        pool = self.filter_main_board(pool)
        logger.info(f"股票池: {before} → {len(pool)}(过滤ST及非主板)")

        pure_codes = set(pool["股票代码"].unique())
        # K 线同步覆盖全 A 股(已过滤 ST/板块),保持数据完整供回测使用
        symbols = [CodeNormalizer.add_market_prefix(c) for c in sorted(pure_codes)]
        inserted = self.sync_all(symbols)
        logger.info(f"K线同步完成,新增 {inserted} 行")

        # 研报过滤仅影响分析池,不影响 K 线数据完整性
        analysis_pool = self._filter_by_research_report(pure_codes)

        save_dir = os.path.dirname(self._cache_dir) if os.path.isdir(self._cache_dir) else os.getcwd()
        out = os.path.join(save_dir, f"final_filtered_stocks_{today_tag}.txt")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for c in sorted(analysis_pool):
                f.write(f"{c}\n")
        logger.info(f"最终股票列表已保存: {len(analysis_pool)} 只 → {out}")
        return analysis_pool

    def _get_min_latest_date(self, symbols: list[str]) -> date | None:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT MIN(latest) FROM (
                        SELECT MAX(trade_date::date) AS latest FROM {TABLE}
                        WHERE symbol = ANY(:symbols) GROUP BY symbol
                    ) sub
                """),
                {"symbols": symbols},
            ).scalar()
        return rows

    # ── stale filter (P0-1) ─────────────────────────────────────

    def _get_stale_symbols(self, symbols: list[str]) -> list[str]:
        if not symbols:
            return []
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT symbol FROM {TABLE}
                    WHERE symbol = ANY(:symbols)
                      AND trade_date::date = :trade_date
                """),
                {"symbols": symbols, "trade_date": self._trade_date.isoformat()},
            ).fetchall()
        up_to_date = {row[0] for row in rows}
        stale = [s for s in symbols if s not in up_to_date]
        skipped = len(symbols) - len(stale)
        if skipped:
            logger.info(f"跳过 {skipped} 只(已有 {self._trade_date_str} 数据),需处理 {len(stale)} 只")
        return stale


    # ── trading day alignment ────────────────────────────────────

    @staticmethod
    def align_to_trading_day(date_str: str) -> str:
        """将 YYYYMMDD 对齐到当天或之后的首个交易日,返回 YYYYMMDD."""
        try:
            from DataCollection.CalendarManager import TradingCalendarAnalyzer
            cal = TradingCalendarAnalyzer()
            dates = sorted(cal.get_official_trading_dates())
            dt = datetime.strptime(date_str, "%Y%m%d")
            formatted = dt.strftime("%Y-%m-%d")
            for d in dates:
                if d >= formatted:
                    return d.replace("-", "")
        except Exception:
            pass
        return date_str

    _align_to_trading_day = align_to_trading_day

    # ── database ────────────────────────────────────────────────


    def _write_batch(self, df: pd.DataFrame) -> None:
        """幂等写入：(symbol, trade_date) 唯一约束 + ON CONFLICT DO UPDATE."""
        if df.empty:
            return
        records = df.rename(columns={}).to_dict("records")
        columns = ["symbol", "trade_date", "open", "close", "high", "low",
                   "open_normal", "close_normal", "high_normal", "low_normal",
                   "volume", "amount", "adj_factor"]
        placeholders = ", ".join(f":{c}" for c in columns)
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c not in ("symbol", "trade_date"))
        with self._engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {TABLE} ({', '.join(columns)})
                    VALUES ({placeholders})
                    ON CONFLICT (symbol, trade_date) DO UPDATE SET {updates}
                """),
                records,
            )

    # ── failed-symbols cache ─────────────────────────────────────

    def _failed_cache_path(self) -> str:
        return os.path.join(self._cache_dir, f"failed_symbols_{self._trade_date_str}.txt")

    def _drop_dead_symbols(self, symbols: set[str]) -> set[str]:
        """剔除永不上市的股票(名称含 IPO终止),避免每次同步都无效重试。
        
        P3-5 审计修复(P0): 不再拦截"退市"类股票——已退市标的的历史K线对回测至关重要，
        需要在退市前区间内正常拉取数据。仅保留 IPO终止的过滤(这些标的从未产生过交易数据)。
        """
        if not symbols:
            return symbols
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text("SELECT stock_code FROM stock_basic_info_sw WHERE stock_code = ANY(:codes)"),
                    {"codes": sorted(symbols)},
                ).fetchall()
                known = {str(r[0]) for r in rows}
                # 仅剔除 IPO终止(从未上市); 退市股票允许同步(需保留其退市前K线)
                if known:
                    name_rows = conn.execute(
                        text("SELECT stock_code FROM stock_basic_info_sw "
                             "WHERE stock_code = ANY(:codes) AND stock_name LIKE '%IPO终止%'"),
                        {"codes": sorted(known)},
                    ).fetchall()
                    dead = {str(r[0]) for r in name_rows}
                else:
                    dead = set()
            kept = set(symbols) - dead
            if dead:
                logger.info(f"跳过 {len(dead)} 只 IPO终止 股票,不再重试")
            return kept
        except Exception as e:
            logger.warning(f"剔除失效股票异常: {e},按原名单重试")
            return symbols

    def _load_failed_set(self) -> set[str]:
        path = self._failed_cache_path()
        if not os.path.exists(path):
            return set()
        try:
            with open(path, encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except Exception as e:
            logger.warning(f"读取失败股票缓存异常: {e}")
            return set()

    def _save_failed_set(self, symbols: set[str]) -> None:
        path = self._failed_cache_path()
        if not symbols:
            if os.path.exists(path):
                os.remove(path)
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sym in sorted(symbols):
                f.write(f"{sym}\n")
        logger.warning(f"缓存 {len(symbols)} 只失败股票 → {os.path.basename(path)}")
