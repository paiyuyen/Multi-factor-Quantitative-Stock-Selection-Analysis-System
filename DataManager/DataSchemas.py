"""
Pandera 数据合约定义

为核心 DataFrame 定义严格的数据模式，确保数据完整性和一致性。
"""

import pandas as pd
from pandera import Column, DataFrameSchema


def create_stock_basic_schema() -> DataFrameSchema:
    """
    股票基础信息数据 Schema

    包含股票代码、名称、行业等基础信息。
    """
    return DataFrameSchema(
        {
            "股票代码": Column(str, nullable=False, coerce=True),
            "股票简称": Column(str, nullable=False, coerce=True),
            "行业": Column(str, nullable=True, coerce=True),
        }
    )


def create_stock_price_schema() -> DataFrameSchema:
    """
    股票价格数据 Schema
    """
    return DataFrameSchema(
        {
            "股票代码": Column(str, nullable=False, coerce=True),
            "最新价": Column(float, nullable=True, coerce=True),
        }
    )


def create_fund_flow_schema() -> DataFrameSchema:
    """
    资金流数据 Schema
    """
    return DataFrameSchema(
        {
            "股票代码": Column(str, nullable=False, coerce=True),
            "股票简称": Column(str, nullable=False, coerce=True),
            "最新价": Column(float, nullable=True, coerce=True),
        },
        strict=False,
    )


def create_industry_board_schema() -> DataFrameSchema:
    """
    行业板块数据 Schema
    """
    return DataFrameSchema(
        {
            "板块名称": Column(str, nullable=False, coerce=True),
            "板块代码": Column(str, nullable=False, coerce=True),
            "涨跌幅": Column(float, nullable=True, coerce=True),
        },
        strict=False,
    )


def create_main_cost_schema() -> DataFrameSchema:
    """
    主力成本数据 Schema
    """
    return DataFrameSchema(
        {
            "股票代码": Column(str, nullable=False, coerce=True),
            "主力成本": Column(float, nullable=True, coerce=True),
            "主力成本差价": Column(float, nullable=True, coerce=True),
            "成本位置": Column(str, nullable=True, coerce=True),
            "主力控盘强度": Column(str, nullable=True, coerce=True),
        },
        strict=False,
    )


def create_strategic_ranking_schema() -> DataFrameSchema:
    """
    P2.11 排名类数据 Schema（强势股池 / 连续上涨 / 量价齐升 / 持续放量 / 均线突破）

    这些 akshare 接口返回结构相似：至少包含"股票代码"列。
    strict=False 允许额外列（各接口返回字段不完全一致）。
    """
    return DataFrameSchema(
        {
            "股票代码": Column(str, nullable=False, coerce=True),
        },
        strict=False,
    )


def create_final_report_schema() -> DataFrameSchema:
    """
    最终分析报告 Schema
    """
    return DataFrameSchema(
        {
            "股票代码": Column(str, nullable=False, coerce=True),
            "股票简称": Column(str, nullable=False, coerce=True),
            "行业": Column(str, nullable=True, coerce=True),
            "所属行业信号": Column(str, nullable=True, coerce=True),
            "最新价": Column(float, nullable=True, coerce=True),
            "强势股": Column(str, nullable=True, coerce=True),
            "量价齐升": Column(str, nullable=True, coerce=True),
            "连涨天数": Column(int, nullable=True, coerce=True),
            "放量天数": Column(int, nullable=True, coerce=True),

            "多头排列趋势": Column(str, nullable=True, coerce=True),
            "股票链接": Column(str, nullable=True, coerce=True),
        },
        strict=False,
    )


class SchemaValidator:
    """
    数据校验器

    提供便捷的 DataFrame 校验接口。
    """

    @staticmethod
    def validate_stock_basic(df: pd.DataFrame, lazy: bool = True) -> tuple[bool, list[str]]:
        """
        校验股票基础信息数据

        Args:
            df: 待校验的 DataFrame
            lazy: 是否延迟校验（收集所有错误后再抛出）

        Returns:
            (是否通过, 错误列表)
        """
        try:
            schema = create_stock_basic_schema()
            schema.validate(df, lazy=lazy)
            return True, []
        except Exception as e:
            return False, [str(e)]

    @staticmethod
    def validate_stock_price(df: pd.DataFrame, lazy: bool = True) -> tuple[bool, list[str]]:
        """
        校验股票价格数据
        """
        try:
            schema = create_stock_price_schema()
            schema.validate(df, lazy=lazy)
            return True, []
        except Exception as e:
            return False, [str(e)]

    @staticmethod
    def validate_fund_flow(df: pd.DataFrame, lazy: bool = True) -> tuple[bool, list[str]]:
        """
        校验资金流数据
        """
        try:
            schema = create_fund_flow_schema()
            schema.validate(df, lazy=lazy)
            return True, []
        except Exception as e:
            return False, [str(e)]

    @staticmethod
    def validate_industry_board(df: pd.DataFrame, lazy: bool = True) -> tuple[bool, list[str]]:
        """
        校验行业板块数据
        """
        try:
            schema = create_industry_board_schema()
            schema.validate(df, lazy=lazy)
            return True, []
        except Exception as e:
            return False, [str(e)]


    @staticmethod
    def validate_main_cost(df: pd.DataFrame, lazy: bool = True) -> tuple[bool, list[str]]:
        """
        校验主力成本数据
        """
        try:
            schema = create_main_cost_schema()
            schema.validate(df, lazy=lazy)
            return True, []
        except Exception as e:
            return False, [str(e)]


    @staticmethod
    def validate_final_report(df: pd.DataFrame, lazy: bool = True) -> tuple[bool, list[str]]:
        """
        校验最终报告数据
        """
        try:
            schema = create_final_report_schema()
            schema.validate(df, lazy=lazy)
            return True, []
        except Exception as e:
            return False, [str(e)]

    # ── P2.11：排名类数据源（强势股池 / 连续上涨 / 量价齐升 / 持续放量 / 均线突破） ──
    @staticmethod
    def validate_strategic_ranking(df: pd.DataFrame, lazy: bool = True) -> tuple[bool, list[str]]:
        """
        P2.11 校验排名类数据源（强势股池 / 连续上涨 / 量价齐升 / 持续放量 / 均线突破）。

        校验逻辑：确保至少包含"股票代码"列，允许其他接口差异列。
        """
        try:
            schema = create_strategic_ranking_schema()
            schema.validate(df, lazy=lazy)
            return True, []
        except Exception as e:
            return False, [str(e)]
