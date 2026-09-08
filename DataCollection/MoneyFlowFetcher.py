from __future__ import annotations

import os
import sys
import time
from typing import Any

import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UtilsManager.ConfigParser import Config


class MoneyFlowFetcher:
    """全市场资金流向获取器（按订单大小分类：小/中/大/特大单）。

    通过 AShareHub /v1/flows/moneyflow 获取，不传 ts_code 即全市场。
    缓存策略：当日首次分页拉取 → 按实际交易日归档写 CSV；当日再次运行直接读缓存。
    429 限流时自动重试（指数退避），分页间隔可配置。

    P0-8① PIT 语义：接口无日期参数（始终返回最新数据），因此：
      - 仅允许在"今天"拉取并按数据实际交易日归档（交易日归档）；
      - 历史日期请求一律只读归档文件，无归档则拒绝拉取（防止把"当日最新"写入
        历史日期缓存，造成回看/重放场景前视）。
    """

    API_PAGE_SIZE = 2000

    def __init__(self, config: Config) -> None:
        self.config = config
        if hasattr(config, 'ASHAREHUB_API_KEY'):
            self.api_key = config.ASHAREHUB_API_KEY
        else:
            self.api_key = None
        if hasattr(config, 'TEMP_DATA_DIRECTORY'):
            self._cache_dir = config.TEMP_DATA_DIRECTORY
        else:
            self._cache_dir = os.path.expanduser("~/Downloads/CoreNews_Reports/cache")
        self._client = None
        self._retry = getattr(config, 'MONEYFLOW_RETRY', 3)
        self._page_delay = getattr(config, 'MONEYFLOW_PAGE_DELAY', 1.0)

    @property
    def _today(self) -> str:
        try:
            from DataCollection.CalendarManager import TradingCalendarAnalyzer
            return TradingCalendarAnalyzer().get_last_trading_day().replace("-", "")
        except Exception:
            from datetime import datetime
            return datetime.now().strftime("%Y%m%d")

    @property
    def _cache_path(self) -> str:
        return self._cache_path_for(self._today)

    def _cache_path_for(self, date_str: str) -> str:
        return os.path.join(self._cache_dir, f"moneyflow_{date_str}.csv")

    @property
    def client(self) -> Any:  # noqa: ANN401
        if self._client is None and self.api_key:
            from UtilsManager.AShareHubClient import make_asharehub_client
            self._client = make_asharehub_client(api_key=self.api_key)
        return self._client

    def fetch_all(self, date: str | None = None) -> pd.DataFrame:
        """获取指定日期全市场资金流向数据，带日级缓存。

        Args:
            date: 日期字符串 YYYYMMDD 或 YYYY-MM-DD，默认当天。
                历史日期（非当日）为只读归档查询：接口无日期参数，无归档则返回空，
                不会拉取"当日最新"冒充历史数据（P0-8 防前视）。

        Returns:
            DataFrame，列与 moneyflow API 一致：
            ts_code, trade_date, buy_sm_vol/amount, sell_sm_vol/amount,
            buy_md_vol/amount, sell_md_vol/amount,
            buy_lg_vol/amount, sell_lg_vol/amount,
            buy_elg_vol/amount, sell_elg_vol/amount,
            net_mf_vol, net_mf_amount
        """
        if not self.api_key:
            logger.info("[MoneyFlow] API 密钥未配置，跳过。")
            return pd.DataFrame()

        target_date = str(date).replace("-", "") if date is not None else self._today

        # ── P0-8① PIT：历史日期只读归档，绝不允许拉取当日最新冒充历史 ──
        if target_date != self._today:
            archive_path = self._cache_path_for(target_date)
            if os.path.exists(archive_path):
                try:
                    archived = pd.read_csv(archive_path)
                    logger.info(
                        f"[MoneyFlow] 历史日期 {target_date} 命中归档: "
                        f"{os.path.basename(archive_path)} ({len(archived)} 条)"
                    )
                    return archived
                except Exception as e:
                    logger.error(f"[MoneyFlow] 历史归档读取失败: {e}")
                    return pd.DataFrame()
            logger.error(
                f"[MoneyFlow] 历史日期 {target_date} 无归档：接口无日期参数无法回补历史，"
                "拒绝拉取当日最新数据写入历史缓存（防前视 P0-8）。"
            )
            return pd.DataFrame()

        # 当日仅读当日缓存
        cache_path = self._cache_path_for(target_date)

        if os.path.exists(cache_path):
            try:
                cached = pd.read_csv(cache_path)
                logger.info(f"[MoneyFlow] 读取当日缓存: {os.path.basename(cache_path)} ({len(cached)} 条)")
                return cached
            except Exception as e:
                logger.info(f"[MoneyFlow] 缓存读取失败，将重新拉取: {e}")

        if not self.client:
            logger.info("[MoneyFlow] 客户端初始化失败，跳过。")
            return pd.DataFrame()

        all_dfs = []
        logger.info(f"[MoneyFlow] 正在从 AShareHub 获取全市场资金流向...")

        for attempt in range(1, self._retry + 2):
            try:
                if attempt > 1:
                    wait = min(2 ** attempt, 30)
                    logger.info(f"  [资金流 重试 {attempt-1}/{self._retry}] 等待 {wait}s...")
                    time.sleep(wait)
                # moneyflow 不带参数返回最新数据（仅当日可拉取）
                df = self.client.moneyflow()
                if df is not None and not df.empty:
                    all_dfs.append(df)
                    logger.info(f"  [资金流] 返回 {len(df)} 行")
                    break
            except Exception as e:
                last_err = e
                if attempt <= self._retry:
                    time.sleep(2 ** attempt)
                    continue
                logger.info(f"[MoneyFlow] 获取失败: {e}")
                break
        if not all_dfs:
            logger.info("[MoneyFlow] 未获取到任何资金流向数据。")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"[MoneyFlow] 获取完成，共 {len(combined)} 条记录")

        # ── P0-8① 按实际交易日归档（接口返回数据所属交易日），不按请求日期命名 ──
        actual_date = target_date
        if "trade_date" in combined.columns:
            td_series = combined["trade_date"].astype(str).str.replace("-", "")
            td_series = td_series[td_series.str.len() == 8]
            if not td_series.empty:
                actual_date = td_series.max()
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            archive_path = self._cache_path_for(actual_date)
            combined.to_csv(archive_path, index=False, encoding="utf-8-sig")
            logger.info(f"[MoneyFlow] 已按交易日 {actual_date} 归档: {os.path.basename(archive_path)}")
        except Exception as e:
            logger.info(f"[MoneyFlow] 归档写入失败: {e}")

        return combined
