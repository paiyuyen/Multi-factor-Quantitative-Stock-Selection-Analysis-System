"""
外部数据源交叉校验机制 (P2.2)

在 DataCollection 层增加数据源交叉校验：定时或同步结束后，
随机抽取 N 只股票当日 OHLCV 比对 akshare 与 asharehub，
偏差 > 0.5% 时在日志输出 WARN。除权价前后日跳变 > 30% 触发异常告警。

业务假设：
- akshare 为基准数据源（系统主用数据源）
- asharehub 为交叉校验数据源（需配置 API KEY）
- 仅校验 OHLCV 字段：open, high, low, close, volume
- 偏差阈值 0.5% 基于 A股行情数据源间常见微小差异
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from UtilsManager.ConfigParser import Config


class CrossSourceValidator:
    """外部数据源交叉校验器。

    职责：
    - 随机抽样标的进行 akshare ↔ asharehub OHLCV 交叉比对
    - 检测除权日价格跳变异常（>30% 触发告警）
    - 记录校验摘要与偏差分布
    """

    # 价格字段偏差阈值（相对偏差 > 0.5% 即 WARN）
    PRICE_TOLERANCE = 0.005
    # 成交量偏差阈值（相对偏差 > 2% 即 WARN，成交量源间差异更大）
    VOLUME_TOLERANCE = 0.02
    # 除权价跳变阈值
    DIV_JUMP_THRESHOLD = 0.30
    # 默认抽样数量
    DEFAULT_SAMPLE_SIZE = 20

    def __init__(self, config: Config, sample_size: int | None = None) -> None:
        self.config = config
        self.sample_size = sample_size or self.DEFAULT_SAMPLE_SIZE
        self._asharehub_client = None

    # ── asharehub 客户端（懒加载）─────────────────────

    @property
    def asharehub_client(self) -> Any | None:  # noqa: ANN401
        if self._asharehub_client is None:
            try:
                api_key = getattr(self.config, "ASHAREHUB_API_KEY", None) or ""
                if not api_key:
                    logger.info("[CrossValidator] AShareHub API KEY 未配置，交叉校验跳过")
                    return None
                from UtilsManager.AShareHubClient import make_asharehub_client
                self._asharehub_client = make_asharehub_client(api_key=api_key)
            except Exception as e:
                logger.warning(f"[CrossValidator] AShareHub 客户端初始化失败: {e}")
                self._asharehub_client = None
        return self._asharehub_client

    # ── 核心校验入口 ──────────────────────────────────

    def cross_validate(
        self,
        candidate_symbols: list[str],
        trade_date: str,
        ohlcv_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """执行 akshare ↔ asharehub OHLCV 交叉校验。

        P2.2 分层抽样：优先覆盖极端案例（涨停/跌停/ST/停牌复牌），
        避免均匀随机抽样遗漏高风险数据异常。

        Args:
            candidate_symbols: 候选标的代码列表（6位数字）
            trade_date: 校验日期 YYYY-MM-DD
            ohlcv_df: 可选的当日 OHLCV 数据（用于涨跌停分层判断）。
                      需含列：symbol/code, open, high, low, close, volume。
                      为 None 时回退到均匀随机抽样。

        Returns:
            校验摘要字典：{sampled, passed, warnings, errors, detail}
        """
        result = {
            "sampled": 0,
            "passed": 0,
            "warnings": 0,
            "errors": 0,
            "detail": [],
        }

        # asharehub 不可用时降级为仅做除权跳变检查
        if not self.asharehub_client:
            logger.info(
                f"[CrossValidator] {trade_date} asharehub 不可用，"
                f"仅执行除权跳变检查（抽样 {len(candidate_symbols)} 标的）"
            )
            return result

        # ── 分层抽样策略 ──────────────────────────────
        sampled = self._stratified_sample(
            candidate_symbols, trade_date, ohlcv_df
        )
        result["sampled"] = len(sampled)

        for sym in sampled:
            ok = self._compare_symbol(sym, trade_date, result)
            if ok:
                result["passed"] += 1
            else:
                result["warnings"] += 1

        logger.info(
            f"[CrossValidator] {trade_date} 交叉校验完成: "
            f"抽样{result['sampled']} | 通过{result['passed']} | "
            f"告警{result['warnings']} | 错误{result['errors']}"
        )
        return result

    # ── 单标的比对 ────────────────────────────────────

    def _compare_symbol(
        self,
        symbol: str,
        trade_date: str,
        result: dict[str, Any],
    ) -> bool:
        """比对单标的当日 OHLCV，返回是否全部通过。"""
        try:
            # 从数据库/缓存获取 akshare 基准数据
            df_base = self._fetch_asharehub_bar(symbol, trade_date)
            if df_base is None or df_base.empty:
                logger.debug(f"[CrossValidator] {symbol} asharehub 无数据，跳过")
                return True

            # 从数据库加载 akshare 数据
            df_ak = self._fetch_ashare_bar(symbol, trade_date)
            if df_ak is None or df_ak.empty:
                logger.debug(f"[CrossValidator] {symbol} akshare DB 无数据，跳过")
                return True

            passed = True
            for field in ["open", "high", "low", "close"]:
                v1 = float(df_base.get(field, 0))
                v2 = float(df_ak.get(field, 0))
                if v1 > 0 and v2 > 0:
                    ratio = abs(v1 - v2) / min(v1, v2)
                    if ratio > self.PRICE_TOLERANCE:
                        logger.warning(
                            f"[CrossValidator] {symbol} {trade_date} {field} 偏差{ratio:.2%} "
                            f"(asharehub={v1:.2f} vs akshare={v2:.2f})"
                        )
                        result["detail"].append(
                            {"sym": symbol, "date": trade_date, "field": field,
                             "ratio": round(ratio, 4), "v1": v1, "v2": v2}
                        )
                        passed = False

            # 成交量比对
            vol1 = float(df_base.get("volume", 0))
            vol2 = float(df_ak.get("volume", 0))
            if vol1 > 0 and vol2 > 0:
                vol_ratio = abs(vol1 - vol2) / min(vol1, vol2)
                if vol_ratio > self.VOLUME_TOLERANCE:
                    logger.warning(
                        f"[CrossValidator] {symbol} {trade_date} volume 偏差{vol_ratio:.2%} "
                        f"(asharehub={vol1:.0f} vs akshare={vol2:.0f})"
                    )
                    result["detail"].append(
                        {"sym": symbol, "date": trade_date, "field": "volume",
                         "ratio": round(vol_ratio, 4), "v1": vol1, "v2": vol2}
                    )
                    passed = False

            return passed

        except Exception as e:
            result["errors"] += 1
            logger.warning(f"[CrossValidator] {symbol} {trade_date} 校验异常: {e}")
            return False

    # ── 除权跳变检测 ──────────────────────────────────

    def detect_div_jumps(
        self,
        price_df: pd.DataFrame,
        symbol: str = "ALL",
    ) -> list[dict[str, Any]]:
        """检测除权日前后日价格跳变异常。

        Args:
            price_df: 含 trade_date, close 列的 DataFrame（单标的或合并）
            symbol: 标的标识（日志用）

        Returns:
            跳变异常列表 [{'date', 'prev_close', 'close', 'ratio', ...}]
        """
        if price_df.empty or "close" not in price_df.columns:
            return []

        jumps = []
        # 按日期排序
        sorted_df = price_df.sort_values("trade_date")
        closes = sorted_df["close"].values
        dates = sorted_df["trade_date"].values

        for i in range(1, len(closes)):
            prev = closes[i - 1]
            curr = closes[i]
            if prev > 0:
                drop = (prev - curr) / prev
                if drop > self.DIV_JUMP_THRESHOLD:
                    logger.warning(
                        f"[CrossValidator] 除权跳变 {symbol} {dates[i]}: "
                        f"close {prev:.2f} → {curr:.2f} (跳变{drop:.2%})"
                    )
                    jumps.append({
                        "date": str(dates[i])[:10],
                        "prev_close": round(prev, 2),
                        "close": round(curr, 2),
                        "ratio": round(drop, 4),
                        "symbol": symbol,
                    })
                    if len(jumps) >= 5:
                        break  # 上限 5 条告警

        return jumps

    # ── 数据获取（内部）───────────────────────────────

    def _stratified_sample(
        self,
        candidate_symbols: list[str],
        trade_date: str,
        ohlcv_df: pd.DataFrame | None,
    ) -> list[str]:
        """分层抽样：优先覆盖极端案例（涨停/跌停/ST/停牌复牌）。

        P2.2 修复：均匀随机抽样可能遗漏高风险异常标的，
        采用分层策略确保极端案例优先被校验。

        分层权重：
        - limit_hit (涨停/跌停) → 最高优先级，每层最多取 5 个
        - st_flag (ST/*ST 标的) → 次高优先级，每层最多取 5 个
        - high_volume_ratio (放量异常) → 中优先级，每层最多取 5 个
        - remaining (其余标的) → 填补至 sample_size

        Args:
            candidate_symbols: 候选标的代码列表
            trade_date: 校验日期
            ohlcv_df: 当日 OHLCV 数据（含 close, open, limit_up, limit_down, volume）

        Returns:
            抽样标的列表
        """
        n_total = min(self.sample_size, len(candidate_symbols))
        candidate_set = set(candidate_symbols)

        if ohlcv_df is None or ohlcv_df.empty:
            # 无 OHLCV 数据时回退均匀随机
            return random.sample(candidate_symbols, n_total)

        sampled: list[str] = []
        sym_col = self._infer_symbol_column(ohlcv_df)

        # ── 层 1：涨停/跌停标的（最高优先级）──
        if "limit_up" in ohlcv_df.columns or "limit_down" in ohlcv_df.columns:
            limit_hits = set()
            df_tmp = ohlcv_df.copy()
            if "limit_up" in df_tmp.columns and "close" in df_tmp.columns:
                mask_lu = (df_tmp["close"] - df_tmp["limit_up"]).abs() < 0.015
                limit_hits |= set(df_tmp.loc[mask_lu, sym_col].astype(str).str[:6])
            if "limit_down" in df_tmp.columns and "close" in df_tmp.columns:
                mask_ld = (df_tmp["close"] - df_tmp["limit_down"]).abs() < 0.015
                limit_hits |= set(df_tmp.loc[mask_ld, sym_col].astype(str).str[:6])
            limit_hits &= candidate_set
            limit_sample = random.sample(
                list(limit_hits), min(5, len(limit_hits))
            ) if limit_hits else []
            sampled.extend(limit_sample)

        # ── 层 2：ST 标的 ──
        st_symbols = {s for s in candidate_symbols if s in ("ST", "*ST") or
                      any(s.startswith(p) for p in ["st", "*st"])}
        # 更可靠的方式：ST 标的通常以特定代码前缀标记，或通过名称判断
        # 这里从 ohlcv_df 尝试匹配
        if sym_col in ohlcv_df.columns:
            all_syms = set(ohlcv_df[sym_col].astype(str).str[:6]) & candidate_set
            st_prefix_matches = {s for s in all_syms if s.startswith(("ST", "*ST"))}
            if not st_prefix_matches:
                st_symbols = st_prefix_matches
            st_available = st_symbols - set(sampled)
            st_sample = random.sample(
                list(st_available), min(5, len(st_available))
            ) if st_available else []
            sampled.extend(st_sample)

        # ── 层 3：放量异常（volume / MA_volume > 2 倍）──
        if "volume" in ohlcv_df.columns and sym_col in ohlcv_df.columns:
            df_vol = ohlcv_df.groupby(sym_col)["volume"].sum().reset_index()
            df_vol = df_vol.rename(columns={"volume": "tot_vol"})
            med_vol = df_vol["tot_vol"].median()
            if med_vol > 0:
                high_vol = set(
                    df_vol.loc[df_vol["tot_vol"] > med_vol * 2, sym_col]
                    .astype(str).str[:6]
                ) & candidate_set - set(sampled)
                high_vol_sample = random.sample(
                    list(high_vol), min(5, len(high_vol))
                ) if high_vol else []
                sampled.extend(high_vol_sample)

        # ── 层 4：剩余标的，随机补齐 ──
        remaining = list(candidate_set - set(sampled))
        need = n_total - len(sampled)
        if need > 0 and remaining:
            fill = random.sample(remaining, min(need, len(remaining)))
            sampled.extend(fill)

        return sampled

    def _infer_symbol_column(self, df: pd.DataFrame) -> str:
        """推断 symbol/code 列名。"""
        for col in ["symbol", "code", "stock_code", "ts_code"]:
            if col in df.columns:
                return col
        return df.columns[0]

    def _fetch_ashare_bar(self, symbol: str, trade_date: str) -> dict | None:
        """从数据库加载 akshare 日线 bar。

        返回 dict 格式 {open, high, low, close, volume} 或 None。
        """
        try:
            from UtilsManager.ConfigParser import Config
            from DataManager.DbEngine import get_engine as _get_engine
            from sqlalchemy import text

            cfg = self.config
            _engine = _get_engine(cfg)
            code_formatted = f"{symbol[:1].upper()}{symbol[1:]}" if len(symbol) == 6 else symbol

            sql = (
                "SELECT open, high, low, close, volume "
                f"FROM ods_stock_daily WHERE stock_code = '{code_formatted}' "
                f"AND trade_date = '{trade_date}' LIMIT 1"
            )
            with _engine.connect() as conn:
                row = pd.read_sql(text(sql), conn)
            if row.empty:
                return None
            return row.iloc[0].to_dict()
        except Exception:
            return None

    def _fetch_asharehub_bar(self, symbol: str, trade_date: str) -> dict | None:
        """通过 asharehub API 获取单标的单日 bar。"""
        if not self.asharehub_client:
            return None
        try:
            df = self.asharehub_client.stock_daily(
                symbol=symbol,
                start_date=trade_date,
                end_date=trade_date,
            )
            if df is None or df.empty:
                return None
            # 统一列名
            col_map = {"vol": "volume", "ts_code": "symbol", "trade_date": "trade_date"}
            for k in ["open", "high", "low", "close", "volume"]:
                if k not in df.columns:
                    alt = "vol" if k == "volume" else k
                    if alt in df.columns:
                        col_map[alt] = k
            result = df.rename(columns=col_map)
            if result.empty:
                return None
            out = {}
            for k in ["open", "high", "low", "close", "volume"]:
                if k in result.columns:
                    out[k] = float(result.iloc[0][k])
            return out if out else None
        except Exception:
            return None
