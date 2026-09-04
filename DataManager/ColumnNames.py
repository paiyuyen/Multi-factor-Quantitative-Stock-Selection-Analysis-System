"""
列名常量定义

统一管理所有DataFrame列名，避免魔法字符串散落在代码中。
"""


class ColumnNames:
    """
    股票分析报告列名常量

    按功能分组：
    - 基础信息列
    - 技术指标列
    - 资金流列
    - 信号列
    - 主力成本列
    - 均线突破列
    """

    # ==================== 基础信息列 ====================
    STOCK_CODE = "股票代码"
    STOCK_NAME = "股票简称"
    INDUSTRY = "行业"
    # P0-7 ①：申万一级行业（独立映射表 stock_basic_info_sw_l1；行业保持二级
    # 语义供行业信号映射，一级列供宏观 tilt 与行业一级中性化使用）
    INDUSTRY_L1 = "行业一级"
    INDUSTRY_SIGNAL = "所属行业信号"
    LATEST_PRICE = "最新价"
    STOCK_LINK = "股票链接"

    # P2.5 修复：K线数据标准列名（Indicators/prepare 共用，消除魔法字符串）
    CLOSE_NORMAL = "close_normal"  # 后复权价
    CLOSE_ADJ = "close"            # 前复权价/原始收盘价
    VOLUME_DATA = "volume"         # 成交量
    MA_VOLUME_5 = "MA_Volume_5"   # 5日成交量均线
    NAME_COL = "name"             # 股票名称

    # akshare原始列名（数据清洗阶段使用）
    AKSHARE_CODE_RAW = "代码"  # 主力成本等接口返回的原始列名
    AKSHARE_INDUSTRY_BOARD_NAME = "板块名称"
    AKSHARE_INDUSTRY_BOARD_CODE = "板块代码"

    # ==================== 技术指标列 - MACD ====================
    MACD_TREND = "MACD趋势"
    MACD_CROSS = "金叉信号"
    MACD_HIST_MOMENTUM = "柱状动能"
    DIF_SLOPE = "DIF斜率"
    DIVERGENCE_SIGNAL = "背离信号"
    VOLUME_PRICE_CONFIRM = "量价配合"
    MACD_TREND_TYPE = "MACD趋势分类"

    # ==================== 技术指标列 - 其他 ====================
    KDJ_SIGNAL = "KDJ_Signal"
    CCI_SIGNAL = "CCI_Signal"
    RSI_SIGNAL = "RSI_Signal"
    BOLL_SIGNAL = "BOLL_Signal"
    KLINE_PATTERN = "K线形态"
    KLINE_PATTERN_SIGNAL = "K线形态信号"

    # ==================== 资金流列 ====================
    # 注意：单位统一为"万元"
    FUND_FLOW_3D = "3日资金流入万元"
    FUND_FLOW_5D = "5日资金流入万元"
    FUND_FLOW_10D = "10日资金流入万元"
    FUND_FLOW_20D = "20日资金流入万元"

    # 资金流动能
    FUND_MOMENTUM = "资金动能"
    FUND_MOMENTUM_SCORE = "资金动能评分"
    FUND_MOMENTUM_STATUS = "资金动能状态"

    # ==================== 信号列 ====================
    STRONG_STOCK = "强势股"
    PRICE_VOLUME_RISE = "量价齐升"
    CONSECUTIVE_RISE_DAYS = "连涨天数"
    VOLUME_INCREASE_DAYS = "放量天数"

    # ==================== 主力成本列 ====================
    CHIP_95_PRICE = "95%筹码价"
    MAIN_COST = "主力成本"
    MAIN_COST_DIFF = "主力成本差价"
    COST_POSITION = "成本位置"
    MAIN_CONTROL_STRENGTH = "主力控盘强度"
    INSTITUTION_PARTICIPATION = "机构参与度"
    INSTITUTION_LEVEL = "机构参与度等级"
    MAIN_COST_DIFF_PERCENT = "主力成本差价百分比"

    # ==================== 均线突破列 ====================
    CURRENT_PRICE = "当前价格"
    MA10_PRICE = "10日均线价"
    MA30_PRICE = "30日均线价"
    MA60_PRICE = "60日均线价"

    # ==================== 趋势评分列 ====================
    BULL_TREND = "多头排列趋势"

    # ==================== akshare原始列名映射 ====================
    # akshare接口返回的原始列名（用于数据提取阶段）
    AKSHARE_NET_FLOW = "净流入"
    AKSHARE_FUND_FLOW_NET = "资金流入净额"
    AKSHARE_MAIN_NET_FLOW = "今日主力净流入-净额"

    # akshare资金流原始列名候选列表
    AKSHARE_FLOW_COLUMNS = [
        AKSHARE_NET_FLOW,
        AKSHARE_FUND_FLOW_NET,
        AKSHARE_MAIN_NET_FLOW,
    ]

    # ==================== 综合分析列 ====================
    COMPREHENSIVE_ANALYSIS = "综合分析结论"
    COMPREHENSIVE_SCORE = "综合分析评分"
    COMPREHENSIVE_LEVEL = "综合级别"
    RISK_LEVEL = "风险等级"

    # ==================== 行业百分位列 ====================
    SCORE_PCT_INDUSTRY = "行业内评分百分位"
    MOMENTUM_PCT_INDUSTRY = "行业内动量百分位"
    QUALITY_PCT_INDUSTRY = "行业内基本面百分位"
    VALUATION_PCT_INDUSTRY = "行业内估值百分位"
    NORTH_FLOW_PCT_INDUSTRY = "行业内北向资金百分位"
    TOP_TRADER_PCT_INDUSTRY = "行业内龙虎榜百分位"
    LIQUIDITY_PCT_INDUSTRY = "行业内流动性百分位"
    VOLATILITY_PCT_INDUSTRY = "行业内波动率百分位"
    MACRO_PCT_INDUSTRY = "行业内宏观百分位"
    FINANCIAL_FORWARD_PCT_INDUSTRY = "行业内财务前瞻百分位"
    EVENT_DRIVEN_PCT_INDUSTRY = "行业内事件驱动百分位"

    # ==================== 通用列名候选列表 ====================
    DATE_COLUMN_CANDIDATES = ["trade_date", "date", "日期", "datetime", "Date", "TRADE_DATE"]
    CODE_COLUMN_CANDIDATES = ["股票代码", "symbol", "code", "ts_code", "代码", "SECURITIES_CODE"]
    PRICE_COLUMN_CANDIDATES = ["最新价", "close", "收盘价", "当前价格", "price", "current_price"]

    # 研报相关
    RESEARCH_REPORT_COUNT = "研报买入次数"

    # ==================== 行业中性化列 ====================
    INDUSTRY_PERCENTILE = "行业内百分位"
    INDUSTRY_SIGNAL_SCORE = "行业信号评分"
    INDUSTRY_DEVIATION = "行业背离扣分"

    # ==================== 退出策略列 ====================
    STOP_LOSS = "止损价"
    T1_TARGET = "T1目标价"
    T2_TARGET = "T2目标价"
    TRAILING_STOP = "移动止损"
    EXIT_RRR = "盈亏比"

    # ==================== 背离位置列 ====================
    DIVERGENCE_DAYS = "背离距今"
    DIVERGENCE_PRICE = "背离位置"
    MACD_ZERO_AXIS_UP_DATE = "MACD上穿零轴时间"  # MACD上穿零轴日期

    # ==================== 多因子 Alpha 评分列 ====================
    FACTOR_QUALITY = "基本面评分"
    FACTOR_VALUATION = "估值评分"
    FACTOR_MOMENTUM = "动量评分"
    FACTOR_MONEYFLOW = "资金流评分"
    FACTOR_MACD = "MACD评分"

    # ==================== 仓位管理列 ====================
    SUGGESTED_POSITION = "建议仓位比例"
    POSITION_REASON = "仓位依据"
    TARGET_WEIGHT = "目标权重"

    # ==================== 流动性分析列 ====================
    AMOUNT = "成交额"
    AMOUNT_MA20 = "成交额MA20"


# ── 列顺序工具函数 ──────────────────────────────────────────────

def get_base_columns() -> list:
    return [
        ColumnNames.STOCK_CODE,
        ColumnNames.STOCK_NAME,
        ColumnNames.INDUSTRY,
        ColumnNames.INDUSTRY_SIGNAL,
        ColumnNames.LATEST_PRICE,
        ColumnNames.CHIP_95_PRICE,
        ColumnNames.MAIN_COST,
        ColumnNames.COST_POSITION,
    ]


def get_signal_columns() -> list:
    cols = [
        ColumnNames.STRONG_STOCK,
        ColumnNames.PRICE_VOLUME_RISE,
        "量价配合",
        ColumnNames.CONSECUTIVE_RISE_DAYS,
        ColumnNames.VOLUME_INCREASE_DAYS,
        ColumnNames.MACD_TREND,
        ColumnNames.MACD_CROSS,
        "柱状动能",
        "DIF斜率",
        ColumnNames.KDJ_SIGNAL,
        ColumnNames.CCI_SIGNAL,
        ColumnNames.RSI_SIGNAL,
        ColumnNames.BOLL_SIGNAL,
        ColumnNames.KLINE_PATTERN_SIGNAL,
        "10日均线价",
        "30日均线价",
        "60日均线价",
        "背离信号",
        ColumnNames.DIVERGENCE_DAYS,
        ColumnNames.DIVERGENCE_PRICE,
        ColumnNames.MACD_ZERO_AXIS_UP_DATE,
        ColumnNames.RISK_LEVEL,
        "宏观风险",
        ColumnNames.STOP_LOSS,
        ColumnNames.T1_TARGET,
        ColumnNames.T2_TARGET,
        ColumnNames.TRAILING_STOP,
        ColumnNames.SUGGESTED_POSITION,
        ColumnNames.TARGET_WEIGHT,
    ]
    return cols


def get_report_columns(fund_flow_periods: list = None) -> list:
    cols = [
        ColumnNames.BULL_TREND,
        ColumnNames.COMPREHENSIVE_ANALYSIS,
        ColumnNames.COMPREHENSIVE_SCORE,
        ColumnNames.COMPREHENSIVE_LEVEL,
        ColumnNames.FACTOR_QUALITY,
        ColumnNames.FACTOR_VALUATION,
        ColumnNames.FACTOR_MOMENTUM,
        ColumnNames.FACTOR_MONEYFLOW,
        ColumnNames.FACTOR_MACD,
        ColumnNames.RESEARCH_REPORT_COUNT,
        ColumnNames.FUND_MOMENTUM,
    ]
    if fund_flow_periods:
        period_map = {
            5: ColumnNames.FUND_FLOW_5D,
            10: ColumnNames.FUND_FLOW_10D,
            20: ColumnNames.FUND_FLOW_20D,
        }
        for period in fund_flow_periods:
            if period in period_map:
                cols.append(period_map[period])
    return cols


def get_all_technical_signal_columns() -> list:
    return [
        ColumnNames.MACD_TREND,
        ColumnNames.KDJ_SIGNAL,
        ColumnNames.CCI_SIGNAL,
        ColumnNames.RSI_SIGNAL,
        ColumnNames.BOLL_SIGNAL,
    ]


def get_final_column_order(fund_flow_periods: list = None) -> list:
    return get_base_columns() + get_signal_columns() + get_report_columns(fund_flow_periods) + [ColumnNames.STOCK_LINK]
