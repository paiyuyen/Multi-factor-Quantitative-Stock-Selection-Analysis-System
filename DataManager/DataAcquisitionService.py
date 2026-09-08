"""
数据获取服务类

负责从akshare等外部数据源获取原始数据，并管理数据缓存。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from loguru import logger

from UtilsManager.ConfigParser import Config
from DataCollection.CalendarManager import TradingCalendarAnalyzer
from DataManager.ColumnNames import ColumnNames
from DataManager.DataFetcher import DataFetcher
from DataManager.DataSchemas import SchemaValidator
from DataManager.DataValidator import DataValidator
from UtilsManager.Exceptions import DataFetchError, handle_exception_with_recovery
from UtilsManager.UnifiedCacheManager import UnifiedCacheManager


class DataAcquisitionService:
    """
    数据获取服务

    职责：
    - 从akshare接口获取原始数据
    - 管理数据缓存
    - 数据源验证
    - 并行获取优化

    Attributes:
        config: 配置管理器实例
        calendar_mgr: 交易日历管理器
        logger: 日志管理器
        cache_manager: 统一缓存管理器
        data_fetcher: 数据获取器（带缓存）
        data_validator: 数据验证器
    """

    def __init__(self, config: Config, calendar_mgr: TradingCalendarAnalyzer, logger: Any, cache_manager: UnifiedCacheManager, executor: ThreadPoolExecutor | None = None) -> None:  # noqa: ANN401
        self.config = config
        self.calendar_mgr = calendar_mgr
        self.logger = logger
        self.cache_manager = cache_manager
        self.data_fetcher = DataFetcher(config, calendar_mgr)
        self.data_validator = DataValidator()
        self.executor = executor

    def get_all_raw_data(self, today_str: str) -> dict[str, pd.DataFrame]:
        """
        获取所有原始数据


        该方法负责从多个 akshare 接口获取原始数据，包括：
        - 资金流数据（根据配置的周期动态获取）
        - 强势股池数据
        - 连续上涨、量价齐升、持续放量等技术指标
        - 均线突破数据（10日、30日、60日）
        - 行业板块信息
        - 主力研报盈利预测

        所有数据都通过 DataFetcher 获取，支持自动缓存和重试机制。

        Args:
            today_str: 当前交易日字符串 (YYYY-MM-DD格式，日历管理器返回格式)

        Returns:
            Dict[str, pd.DataFrame]: 键值对字典，key为数据名称，value为对应的DataFrame
            包含的键包括：
            - market_fund_flow_raw[_3/_10/_20]: 市场资金流向数据
            - strong_stocks_raw: 强势股池数据
            - consecutive_rise_raw: 连续上涨数据
            - ljqs_raw: 量价齐升数据
            - cxfl_raw: 持续放量数据
            - xstp_10_raw/xstp_30_raw/xstp_60_raw: 均线突破数据
            - industry_board_df: 行业板块成分股映射
            - main_cost_data: 主力成本数据

        Raises:
            Exception: 当数据获取失败时抛出异常
        """
        logger.info("\n>>> 正在初始化数据获取和缓存检查...")

        # 根据配置动态获取资金流数据（并行优化版）
        data = {}

        # akshare接口参数映射（严格对应 stock_fund_flow_individual 的 symbol 参数）
        period_to_akshare = {
            3: ("3日市场资金流向", "3日排行", "market_fund_flow_raw_3"),
            5: ("5日市场资金流向", "5日排行", "market_fund_flow_raw"),
            10: ("10日市场资金流向", "10日排行", "market_fund_flow_raw_10"),
            20: ("20日市场资金流向", "20日排行", "market_fund_flow_raw_20"),
        }

        # 构建待获取的任务列表
        fund_flow_tasks = []
        for period in self.config.FUND_FLOW_PERIODS:
            if period in period_to_akshare:
                desc, symbol, key = period_to_akshare[period]
                fund_flow_tasks.append((period, desc, symbol, key))
            else:
                logger.warning(f"不支持的资金流周期: {period}日（仅支持3,5,10,20）")

        # 串行获取所有资金流数据（避免 py_mini_racer 多线程内存冲突）
        logger.info("\n>>> 正在获取资金流数据...")

        for task in fund_flow_tasks:
            period, desc, symbol, key = task
            try:
                logger.info(f"  - 正在获取: {desc}...")
                fund_flow_df = self.data_fetcher.fetch('stock_fund_flow_individual', desc, symbol=symbol)

                # 验证资金流数据
                if not fund_flow_df.empty:
                    required_cols = [ColumnNames.STOCK_CODE, ColumnNames.LATEST_PRICE]
                    is_valid, missing = self.data_validator.validate_required_columns(
                        fund_flow_df, required_cols, f"{period}日资金流"
                    )

                    if is_valid:
                        # Pandera 数据契约校验
                        is_pandera_valid, pandera_errors = SchemaValidator.validate_fund_flow(fund_flow_df)
                        if is_pandera_valid:
                            logger.info(f"  - [OK] 已获取 {period}日资金流数据 ({len(fund_flow_df)} 条记录)")
                            data[key] = fund_flow_df
                        else:
                            logger.warning(f"  - [WARN] {period}日资金流数据契约校验失败: {pandera_errors}")
                            logger.warning("  - [WARN] 继续使用该数据，但请注意可能存在数据质量问题")
                    else:
                        logger.warning(f"  - [WARN] {period}日资金流数据缺少列: {missing}，跳过")
                else:
                    logger.warning(f"  - [WARN] {period}日资金流数据为空")

            except (ValueError, KeyError, TypeError, ConnectionError, RuntimeError) as e:
                handle_exception_with_recovery(
                    DataFetchError(f"{period}日资金流", str(e)),
                    self.logger,
                    f"获取{period}日资金流数据",
                    default_value=None,
                    raise_on_critical=False,
                )

        # 获取其他数据源（带验证，并行优化版）
        logger.info("\n>>> 正在并行获取其他技术指标数据...")

        strong_date = today_str.replace("-", "")
        data_sources = {
            "strong_stocks_raw": ("stock_zt_pool_strong_em", "强势股池", {"date": strong_date}),
            "consecutive_rise_raw": ("stock_rank_lxsz_ths", "连续上涨", {}),
            "ljqs_raw": ("stock_rank_ljqs_ths", "量价齐升", {}),
            "cxfl_raw": ("stock_rank_cxfl_ths", "持续放量", {}),
        }

        # 定义单个数据源获取的worker函数
        def fetch_data_source_worker(item: Any) -> tuple[str, pd.DataFrame]:  # noqa: ANN401
            key, (api_func, desc, params) = item
            try:
                df = self.data_fetcher.fetch(api_func, desc, **params)

                if not df.empty:
                    required_cols = [ColumnNames.STOCK_CODE]
                    is_valid, missing = self.data_validator.validate_required_columns(df, required_cols, desc)

                    if is_valid:
                        # P2.11: Pandera 数据契约校验（非阻塞，仅WARN）
                        is_schema_valid, schema_errors = SchemaValidator.validate_strategic_ranking(df)
                        if is_schema_valid:
                            logger.info(f"  - [OK] {desc}: {len(df)} 条记录")
                            return key, df
                        else:
                            logger.warning(f"  - [WARN] {desc} Pandera校验失败: {schema_errors}")
                            logger.warning(f"  - [WARN] 继续使用{desc}数据，但请注意可能存在数据质量问题")
                            return key, df
                    else:
                        logger.warning(f"  - [WARN] {desc} 缺少列: {missing}")
                        return key, pd.DataFrame()
                else:
                    logger.warning(f"  - [WARN] {desc} 数据为空")
                    return key, pd.DataFrame()

            except (ValueError, KeyError, TypeError, ConnectionError, RuntimeError) as e:
                logger.error(f"  - [FAIL] 获取{desc}失败: {e}")
                return key, pd.DataFrame()

        # 并行获取所有数据源
        exec_fund = self.executor or ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        try:
            futures = {exec_fund.submit(fetch_data_source_worker, item): item for item in data_sources.items()}

            for future in as_completed(futures):
                try:
                    key, df = future.result()
                    if not df.empty:
                        data[key] = df
                except (TimeoutError, ValueError, TypeError, KeyError) as e:
                    logger.error(f"获取数据源时发生异常: {e}")
        finally:
            if self.executor is None:
                exec_fund.shutdown(wait=True)

        # 均线突破数据 (Akshare接口参数不同，需分开获取，并行优化版)
        logger.info("\n>>> 正在并行获取均线突破数据...")

        xstp_configs = [
            ("xstp_10_raw", "向上突破10日均线", "10日均线"),
            ("xstp_30_raw", "向上突破30日均线", "30日均线"),
            ("xstp_60_raw", "向上突破60日均线", "60日均线"),
        ]

        # 定义单个均线突破数据获取的worker函数
        def fetch_xstp_worker(config: tuple[str, str, str]) -> tuple[str, pd.DataFrame]:
            key, desc, symbol = config
            try:
                df = self.data_fetcher.fetch('stock_rank_xstp_ths', desc, symbol=symbol)

                if not df.empty:
                    required_cols = [ColumnNames.STOCK_CODE]
                    is_valid, missing = self.data_validator.validate_required_columns(df, required_cols, desc)

                    if is_valid:
                        # P2.11: Pandera 数据契约校验（非阻塞，仅WARN）
                        is_schema_valid, schema_errors = SchemaValidator.validate_strategic_ranking(df)
                        if is_schema_valid:
                            logger.info(f"  - [OK] {desc}: {len(df)} 条记录")
                            return key, df
                        else:
                            logger.warning(f"  - [WARN] {desc} Pandera校验失败: {schema_errors}")
                            logger.warning(f"  - [WARN] 继续使用{desc}数据，但请注意可能存在数据质量问题")
                            return key, df
                    else:
                        logger.warning(f"  - [WARN] {desc} 缺少列: {missing}")
                        return key, pd.DataFrame()
                else:
                    logger.warning(f"  - [WARN] {desc} 数据为空")
                    return key, pd.DataFrame()

            except (ValueError, KeyError, TypeError, ConnectionError, RuntimeError) as e:
                logger.error(f"  - [FAIL] 获取{desc}失败: {e}")
                return key, pd.DataFrame()

        # 并行获取所有均线突破数据
        exec_xstp = self.executor or ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        try:
            futures = {exec_xstp.submit(fetch_xstp_worker, config): config for config in xstp_configs}

            for future in as_completed(futures):
                try:
                    key, df = future.result()
                    if not df.empty:
                        data[key] = df
                except (TimeoutError, ValueError, TypeError, KeyError) as e:
                    logger.error(f"获取均线突破数据时发生异常: {e}")
        finally:
            if self.executor is None:
                exec_xstp.shutdown(wait=True)

        # 获取主力成本数据
        logger.info("\n>>> 正在获取主力成本数据...")
        try:
            from LogicAnalyzer.Distribution import MainCostDataManager

            cost_manager = MainCostDataManager(
                cache_enabled=True,
                cache_dir=os.path.join(self.config.CACHE_DIRECTORY, "cost_data_cache"),
            )
            main_cost_df = cost_manager.get_main_cost_data()

            if not main_cost_df.empty:
                # 标准化列名：将 "代码" 重命名为 "股票代码"
                if (
                    ColumnNames.AKSHARE_CODE_RAW in main_cost_df.columns
                    and ColumnNames.STOCK_CODE not in main_cost_df.columns
                ):
                    main_cost_df.rename(columns={ColumnNames.AKSHARE_CODE_RAW: ColumnNames.STOCK_CODE}, inplace=True)

                # 验证必需列
                required_cols = [ColumnNames.STOCK_CODE, ColumnNames.MAIN_COST]
                is_valid, missing = self.data_validator.validate_required_columns(
                    main_cost_df, required_cols, "主力成本数据"
                )

                if is_valid:
                    # 先分析数据，生成所需的计算列
                    main_cost_df = cost_manager.analyze_cost_data(main_cost_df)
                    
                    # 再进行 Pandera 数据契约校验
                    is_pandera_valid, pandera_errors = SchemaValidator.validate_main_cost(main_cost_df)
                    if is_pandera_valid:
                        data["main_cost_data"] = main_cost_df
                        logger.info(f"  - [OK] 主力成本数据: {len(main_cost_df)} 条记录")
                    else:
                        logger.warning(f"  - [WARN] 主力成本数据契约校验失败: {pandera_errors}")
                        logger.warning("  - [WARN] 继续使用该数据，但请注意可能存在数据质量问题")
                        data["main_cost_data"] = main_cost_df

                    # 打印主力成本数据摘要
                    cost_manager.print_cost_summary(main_cost_df)
                else:
                    logger.warning(f"  - [WARN] 主力成本数据缺少列: {missing}")
                    data["main_cost_data"] = pd.DataFrame()
            else:
                logger.warning("  - [WARN] 主力成本数据为空")
                data["main_cost_data"] = pd.DataFrame()

        except (ValueError, TypeError, OSError, ImportError) as e:
            logger.error(f"  - [FAIL] 获取主力成本数据失败: {e}")
            data["main_cost_data"] = pd.DataFrame()

        try:
            # 从已获取数据中提取候选股票代码清单
            candidate_syms = set()
            for key in ("market_fund_flow_raw", "market_fund_flow_raw_3", "strong_stocks_raw"):
                df_src = data.get(key)
                if df_src is not None and not df_src.empty and ColumnNames.STOCK_CODE in df_src.columns:
                    candidate_syms.update(df_src[ColumnNames.STOCK_CODE].dropna().unique())
            if candidate_syms:
                from DataCollection.CrossSourceValidator import CrossSourceValidator
                validator = CrossSourceValidator(self.config)
                validator.cross_validate(list(candidate_syms), today_str)
        except Exception as e:
            logger.warning(f"[P2.2] 交叉校验异常（不影响主流程）: {e}")

        return data


