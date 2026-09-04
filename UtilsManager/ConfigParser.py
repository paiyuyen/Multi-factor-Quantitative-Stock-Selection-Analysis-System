# DataManager/Config.py
"""
配置管理模块（使用 Pydantic 重构）

负责读取和验证 config.ini 配置文件，提供全局配置访问接口。
支持以下配置分组：
- DATABASE: 数据库连接配置
- SYSTEM: 系统运行参数
- FUND_FLOW: 资金流分析配置
- TECHNICAL_INDICATORS: 技术指标参数
- DATA_SYNC: 数据同步配置

使用 Pydantic 带来的优势：
- 开箱即用的类型安全
- 自动类型转换（字符串→int/list/tuple）

版本: 2.0.0
- 优雅的数据校验（使用 @field_validator）
- 支持环境变量覆盖
"""

import os

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_signal_workers() -> int:
    return max(os.cpu_count() or 4, 2)


def parse_aliases(alias_str: str) -> dict[str, str]:
    """解析别名字符串为字典，格式：'别名1=目标,别名2=目标'"""
    aliases = {}
    for pair in alias_str.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            aliases[key.strip()] = value.strip()
    return aliases


class DatabaseConfig(BaseModel):
    """数据库配置模型"""

    user: str
    password: str
    host: str
    port: str
    db_name: str


class SystemConfig(BaseModel):
    """系统配置模型"""

    HOME_DIRECTORY: str = Field(default="~/Downloads/CoreNews_Reports")
    TEMP_DATA_DIR: str = Field(default=".")
    MAX_WORKERS: int = Field(default=15, ge=1)
    DATA_FETCH_RETRIES: int = Field(default=3, ge=1)
    DATA_FETCH_DELAY: int = Field(default=5, ge=1)
    STOCK_BASIC_INFO_EXPIRE_DAYS: int = Field(default=30, ge=1, le=365,
                                                description="股票基本信息缓存过期天数")
    SIGNAL_PROCESSING_PROCESSES: int = Field(default_factory=_default_signal_workers, ge=1)
    # ── D1 分片执行（shard）：大盘任务按 symbol 分批并行 + 失败片重跑 ──
    SHARD_MODE: str = Field(default="hybrid",
                            description="分片模式: off 单任务串行(回退) / symbol 按股票分片 / hybrid 全部已实现维度(v1=symbol, date 维度为 WFO path 扩展点)")
    SHARD_SYMBOL_BATCH_SIZE: int = Field(default=50, ge=1, le=10000,
                                         description="symbol 分片每批股票数（片粒度）")
    SHARD_MAX_ATTEMPTS: int = Field(default=2, ge=1, le=10,
                                    description="失败片最大尝试次数（含首跑；仅重跑失败片）")
    SHARD_MAX_WORKERS: int = Field(default=0, ge=0, le=64,
                                   description="分片并发 worker 数（0=自动=CPU 核数）")
    # ── Task E 幂等输出：分片输出按 (shard_id, key) upsert + 原子写 ──
    OUTPUT_WRITE_MODE: str = Field(default="upsert",
                                   description="输出写入模式: upsert 原子写+清单去重（重复运行不重复累加） / replace 直接替换写（禁用 upsert 回退）")

    @field_validator("SHARD_MODE")
    @classmethod
    def validate_shard_mode(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in ("off", "symbol", "hybrid"):
            msg = f"SHARD_MODE 必须为 off/symbol/hybrid，收到 {v}"
            raise ValueError(msg)
        return v_lower

    @field_validator("OUTPUT_WRITE_MODE")
    @classmethod
    def validate_output_write_mode(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in ("upsert", "replace"):
            msg = f"OUTPUT_WRITE_MODE 必须为 upsert/replace，收到 {v}"
            raise ValueError(msg)
        return v_lower

    @field_validator("HOME_DIRECTORY")
    @classmethod
    def expand_home(cls, v: str) -> str:
        return os.path.expanduser(v)


class LoggingConfig(BaseModel):
    """日志配置模型"""

    LOG_LEVEL: str = Field(default="INFO")
    LOG_DIR: str = Field(default="Logs")


class MultiHeadArrangementConfig(BaseModel):
    """多头排列评分系统配置"""

    FULL_BULL_THRESHOLD: int = Field(default=85, ge=0, le=100)
    TREND_ACCELERATION_THRESHOLD: int = Field(default=65, ge=0, le=100)
    TREND_OSCILLATION_THRESHOLD: int = Field(default=45, ge=0, le=100)
    MOVING_AVERAGE_PERIODS: list[int] = Field(default=[5, 10, 20, 30, 60])

    @field_validator("MOVING_AVERAGE_PERIODS", mode="before")
    @classmethod
    def parse_periods(cls, v: str | list[int]) -> list[int]:
        if isinstance(v, str):
            return [int(p.strip()) for p in v.split(",")]
        return v


class FilterRulesConfig(BaseModel):
    """弱势股过滤规则配置 + 流动性参数"""

    ENABLE_WEAK_STOCK_FILTER: bool = Field(default=True)
    EXEMPT_LEVELS: list[str] = Field(default=["完全主升", "趋势加速"])
    # 行业截面百分位过滤阈值（0-100）
    INDUSTRY_PCT_HARD: float = Field(default=10.0, ge=0.0, le=100.0)
    INDUSTRY_PCT_D: float = Field(default=30.0, ge=0.0, le=100.0)
    INDUSTRY_PCT_EXEMPT: float = Field(default=80.0, ge=0.0, le=100.0)

    @field_validator("EXEMPT_LEVELS", mode="before")
    @classmethod
    def parse_exempt_levels(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [level.strip() for level in v.split(",")]
        return v

    # ── 流动性参数 ────────────────────────────────────────────────────
    LIQ_VETO_RATIO: float = Field(default=0.05, ge=0.01, le=1.0)
    LIQ_W_SECTION: float = Field(default=0.4, ge=0.0, le=1.0)
    LIQ_W_TIMESERIES: float = Field(default=0.4, ge=0.0, le=1.0)
    LIQ_W_MARKETCAP: float = Field(default=0.2, ge=0.0, le=1.0)
    LIQ_MIN_DISCOUNT: float = Field(default=0.3, ge=0.0, le=1.0)


class FundFlowConfig(BaseModel):
    """资金流分析配置"""

    FUND_FLOW_PERIODS: list[int] = Field(default=[5, 10, 20])

    @field_validator("FUND_FLOW_PERIODS", mode="before")
    @classmethod
    def parse_periods(cls, v: str | list[int]) -> list[int]:
        if isinstance(v, str):
            return [int(p.strip()) for p in v.split(",")]
        return v

    @field_validator("FUND_FLOW_PERIODS")
    @classmethod
    def validate_periods(cls, v: list[int]) -> list[int]:
        VALID_FUND_FLOW_PERIODS = {3, 5, 10, 20}
        ALLOWED_COMBINATIONS = [
            (3, 5, 10),
            (3, 5, 20),
            (5, 10, 20),
            (3, 10, 20),
        ]

        if len(v) != 3:
            raise ValueError(
                f"错误：资金流周期必须设置为三个参数，当前设置了 {len(v)} 个。\n"
                f"允许的组合：\n"
                f"  - 3,5,10   （短中周期组合，推荐短线）\n"
                f"  - 3,5,20   （短长周期组合）\n"
                f"  - 5,10,20  （中长周期组合，默认，推荐中线）\n"
                f"  - 3,10,20  （分散周期组合）"
            )

        invalid_periods = [p for p in v if p not in VALID_FUND_FLOW_PERIODS]
        if invalid_periods:
            raise ValueError(
                f"错误：资金流周期包含无效值 {invalid_periods}。\n仅支持以下周期：{sorted(VALID_FUND_FLOW_PERIODS)}"
            )

        sorted_periods = tuple(sorted(v))
        if sorted_periods not in ALLOWED_COMBINATIONS:
            raise ValueError(
                f"错误：资金流周期组合 {v} 不被允许。\n"
                f"允许的组合（顺序不限）：\n"
                f"  - 3,5,10   （短中周期组合，推荐短线）\n"
                f"  - 3,5,20   （短长周期组合）\n"
                f"  - 5,10,20  （中长周期组合，默认，推荐中线）\n"
                f"  - 3,10,20  （分散周期组合）"
            )

        return v


class TechnicalIndicatorsConfig(BaseModel):
    """技术指标信号配置"""

    MACD_PARAMS: tuple[int, int, int] = Field(default=(12, 26, 9))

    @field_validator("MACD_PARAMS", mode="before")
    @classmethod
    def parse_macd_params(cls, v: str | tuple[int, int, int]) -> tuple[int, int, int]:
        if isinstance(v, str):
            return tuple(int(p.strip()) for p in v.split(","))
        return v

    @field_validator("MACD_PARAMS")
    @classmethod
    def validate_macd_params(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        fast, slow, signal = v
        if fast >= slow:
            raise ValueError(
                f"错误：MACD参数不合理（快线{fast} >= 慢线{slow}），"
                f"请确保快线 < 慢线。"
            )
        return v


class ColumnAliasesConfig(BaseModel):
    """列名别名配置"""

    code_aliases: str = Field(default="代码=股票代码,证券代码=股票代码,股票代码=股票代码")
    name_aliases: str = Field(default="名称=股票简称,股票名称=股票简称,股票简称=股票简称,简称=股票简称")
    price_aliases: str = Field(
        default="最新价=最新价,现价=最新价,当前价格=最新价,今收盘=最新价,收盘=最新价,收盘价=最新价"
    )


class FullBullScoringConfig(BaseModel):
    """MACD 完全多头评分维度权重 + 规则阈值"""

    WEIGHT_ZERO_AXIS: int = Field(default=20, ge=0, le=100)
    WEIGHT_STRATEGY_GOLDEN: int = Field(default=15, ge=0, le=100)
    WEIGHT_MOMENTUM: int = Field(default=15, ge=0, le=100)
    WEIGHT_DIF_SLOPE: int = Field(default=10, ge=0, le=100)
    WEIGHT_DIVERGENCE: int = Field(default=10, ge=0, le=100)
    WEIGHT_VOLUME_PRICE: int = Field(default=10, ge=0, le=100)
    WEIGHT_KLINE_PATTERN: int = Field(default=10, ge=0, le=100)
    CONCLUSION_FULL_BULL: int = Field(default=80, ge=0, le=100)
    CONCLUSION_BULLISH: int = Field(default=60, ge=0, le=100)
    CONCLUSION_OSCILLATE: int = Field(default=40, ge=0, le=100)
    # 规则阈值
    RULE_DIVERGENCE_THRESHOLD: float = Field(default=0.3, ge=0, le=1.0)
    RULE_WINNER_RATE_HIGH: int = Field(default=80, ge=0, le=100)
    RULE_WINNER_RATE_LOW: int = Field(default=15, ge=0, le=100)
    RULE_COST_RESISTANCE_RATIO: float = Field(default=0.95, ge=0, le=1.0)
    RULE_CHIP_CONCENTRATED_RATIO: float = Field(default=0.15, ge=0, le=1.0)
    RULE_PRICE_NEW_HIGH_DAYS: int = Field(default=20, ge=5, le=120)


class UserFocusStocksConfig(BaseModel):
    """用户关注股池配置"""

    USER_FOCUS_STOCKS: str = Field(default="")


class MultiFactorAlphaConfig(BaseModel):
    """多因子 Alpha 配置"""

    ENABLED: bool = Field(default=True, description="是否启用多因子 Alpha 评分")
    FINANCIAL_QUALITY_CACHE_DAYS: int = Field(default=90, ge=1, le=365,
                                               description="质量因子缓存天数")
    FINANCIAL_QUALITY_BATCH_SIZE: int = Field(default=100, ge=1, le=5000,
                                              description="质量因子每批采集股票数")
    FINANCIAL_QUALITY_BATCH_SLEEP: int = Field(default=10, ge=0, le=600,
                                               description="质量因子批间休眠秒数")
    FINANCIAL_QUALITY_FILE_CACHE_DAYS: int = Field(default=30, ge=1, le=365,
                                                   description="质量因子离线文件缓存天数")
    FUNDAMENTALS_RETRY: int = Field(default=3, ge=0, le=10,
                                     description="估值因子 API 重试次数")
    # 因子权重已迁移至 config/factor_registry.yaml


class AShareHubConfig(BaseModel):
    """AShareHub 筹码分布数据配置"""

    API_KEY: str = Field(default="")
    MONEYFLOW_RETRY: int = Field(default=3, ge=0, le=10,
                                   description="资金流向 API 429 重试次数")
    MONEYFLOW_PAGE_DELAY: float = Field(default=1.0, ge=0.0, le=30.0,
                                          description="资金流分页间隔秒数")
    ENABLE_FUNDAMENTALS: bool = Field(default=True,
                                       description="是否启用 AShareHub 估值因子同步")


class MacroFilterConfig(BaseModel):
    """宏观过滤器配置"""

    ENABLE_MACRO_FILTER: bool = Field(default=True)


class RegimeDetectionConfig(BaseModel):
    """市场状态分类参数"""

    BOLL_NARROW_RATIO: float = Field(default=0.8, ge=0.1, le=2.0,
                                      description="窄布林判定阈值：近期BOLL带宽/历史平均带宽 < 此值→震荡")
    OSCILLATION_HIST_STD_RATIO: float = Field(default=0.1, ge=0.01, le=1.0,
                                               description="震荡模式柱状图标准差比：abs(柱状图) < 此值×close.std()→震荡")
    TOP_RISK_MA20_DEVIATION: float = Field(default=0.15, ge=0.01, le=0.5,
                                            description="顶风险MA20偏离阈值：(close-MA20)/MA20 > 此值→顶部风险")
    OSCILLATION_MIN_BARS: int = Field(default=30, ge=10, le=120,
                                       description="震荡判定最小K线数")
    REVERSAL_LOOKBACK: int = Field(default=10, ge=5, le=60,
                                    description="反转检测回溯长度（根K线）")


class DivergenceConfig(BaseModel):
    """背离检测参数"""

    BASE_DISTANCE: int = Field(default=10, ge=5, le=60,
                                description="背离检测基础窗口（adaptive_distance的base_distance）")
    STRENGTH_THRESHOLD: float = Field(default=0.15, ge=0.01, le=1.0,
                                       description="背离有效强度门限，超过此值才生成信号")
    DECAY_HALF_LIFE: int = Field(default=8, ge=2, le=60,
                                  description="背离信号半衰期（天）")
    SLOPE_WINDOW: int = Field(default=5, ge=3, le=30,
                               description="DIF斜率线性回归窗口（根K线）")


class ScoringParamsConfig(BaseModel):
    """评分计算参数"""

    CROSS_DECAY_DAYS: int = Field(default=30, ge=5, le=120,
                                   description="金叉信号衰减半衰期（天）")
    CROSS_DECAY_MIN: float = Field(default=0.3, ge=0.1, le=1.0,
                                    description="金叉衰减下限（比例）")
    KLINE_DECAY_DAYS: int = Field(default=10, ge=2, le=60,
                                   description="K线形态衰减半衰期（天）")
    KLINE_DECAY_MIN: float = Field(default=0.2, ge=0.05, le=1.0,
                                    description="K线形态衰减下限（比例）")
    VOL_NORM_DENOMINATOR: float = Field(default=0.15, ge=0.01, le=1.0,
                                         description="金叉强度波动率归一化分母：(DIF-DEA)/ATR/此值→vol_factor")
    ATR_STOP_MULT: float = Field(default=1.5, ge=0.5, le=5.0,
                                  description="止损ATR倍数：止损价=close-ATR×此值")
    TRAILING_STOP_HIGH_RATIO: float = Field(default=0.98, ge=0.9, le=1.0,
                                              description="移动止损高位触发比：close≥近N日最高价×此值")
    TRAILING_STOP_LOOKBACK: int = Field(default=10, ge=5, le=60,
                                         description="移动止损回溯窗口（根K线）")
    TRAILING_STOP_HIGH_LOOKBACK: int = Field(default=20, ge=10, le=120,
                                              description="移动止损参考高点回溯窗口（根K线）")
    EXPECTED_RETURN_LOOKBACK: int = Field(default=20, ge=5, le=120,
                                           description="预期盈亏比计算回溯窗口（根K线）")
    GOLDEN_CROSS_BONUS: int = Field(default=10, ge=0, le=50,
                                     description="R04: 金叉量价确认加分")
    DIVERGENCE_PENALTY: int = Field(default=20, ge=0, le=50,
                                     description="R41: 顶背离量缩扣分")
    INDUSTRY_VALUATION_AGG_METHOD: str = Field(
        default="aggregate_profitable",
        description="行业估值聚合口径: aggregate_profitable(剔除亏损股, 中证口径) / "
                    "aggregate_full(整体法含负利润, 申万口径, 行业整体亏损时 PE 为负)",
    )
    # ── 波动率自适应退出参数（VAEO 学习产出）──
    LEARNED_T1_MULT: float = Field(default=3.0, ge=1.0, le=10.0,
                                    description="回测学习到的 T1 止盈 ATR 倍数（替代硬编码默认值）")
    LEARNED_T2_MULT: float = Field(default=5.0, ge=1.0, le=15.0,
                                    description="回测学习到的 T2 止盈 ATR 倍数（替代硬编码默认值）")


class TechnicalConstantsConfig(BaseModel):
    """标准技术指标参数"""

    ATR_LENGTH: int = Field(default=14, ge=5, le=60,
                             description="ATR计算周期（Wilder标准14）")
    ADX_LENGTH: int = Field(default=14, ge=5, le=60,
                             description="ADX计算周期（Wilder标准14）")
    RSI_LENGTH: int = Field(default=14, ge=5, le=60,
                             description="RSI计算周期（Wilder标准14）")
    BOLL_LENGTH: int = Field(default=20, ge=5, le=60,
                              description="BOLL计算周期（Bollinger标准20）")
    BOLL_STD: float = Field(default=2.0, ge=1.0, le=4.0,
                             description="BOLL标准差倍数（标准2）")
    STOCH_K: int = Field(default=9, ge=3, le=30,
                          description="Stoch %K周期（Lane标准9）")
    STOCH_D: int = Field(default=3, ge=2, le=15,
                          description="Stoch %D平滑周期（标准3）")
    KLINE_SCAN_WINDOW: int = Field(default=60, ge=20, le=200,
                                    description="K线形态扫描窗口（根K线）")


class BacktestConfig(BaseModel):
    """回测系统配置模型"""

    ENABLED: bool = True
    OPTIMIZE_FREQUENCY: str = "monthly"
    BACKTEST_START_DATE: str = Field(default="20200101", pattern=r"^\d{8}$")
    OUT_OF_SAMPLE_DAYS: int = Field(default=120, ge=20, le=504)
    HOLDOUT_RATIO: float = Field(default=0.20, ge=0.0, le=0.50,
                                  description="末段独立 holdout 占正式回测交易日比例（0.20=末段20%对WFO全程禁触，终验只在此段进行；0.0=禁用回退旧逻辑）")
    INITIAL_CASH: float = Field(default=1_000_000, gt=0)
    COMMISSION_RATE: float = Field(default=0.0003, ge=0, le=0.01)
    STAMP_TAX_RATE: float = Field(default=0.0005, ge=0, le=0.01,
                                     description="印花税费率（卖出收取，2023.8 起万五）")
    SLIPPAGE: float = Field(default=0.001, ge=0, le=0.01)
    TRANSFER_FEE_RATE: float = Field(default=0.00001, ge=0, le=0.001,
                                     description="过户费（双边，万0.1）")
    MIN_COMMISSION_PER_TRADE: float = Field(default=5.0, ge=0, le=100,
                                            description="A股每笔最低佣金（5元）")
    TIERED_COMMISSION_RATES: str = Field(
        default="",
        description="阶梯佣金费率（可选；空=统一COMMISSION_RATE）。格式: threshold1:rate1;threshold2:rate2;... 按成交额阈值升序匹配。例: 1000000:0.00025;5000000:0.0002"
    )
    # ── 冲击成本（未提供 AMOUNT_MA20 时的统一参数） ──
    IMPACT_BASE: float = Field(default=0.002, ge=0, le=0.05,
                               description="大单冲击成本基数（阈值处）")
    IMPACT_THRESHOLD: float = Field(default=0.01, ge=1e-5, le=1.0,
                                    description="冲击成本启用阈值（占ADV比例）")
    IMPACT_CAP: float = Field(default=0.05, ge=0, le=1.0,
                              description="冲击成本上限（防极端流动性下滑点>100%）")
    # ── 流动性分档冲击成本（按 AMOUNT_MA20 分档，业界做法） ──
    LIQUIDITY_TIER_EDGES: str = Field(default="5e6,2e7,1e8",
                                      description="流动性分档边界（AMOUNT_MA20 元，逗号分隔，档数=边界数+1）")
    LIQUIDITY_TIER_IMPACT_BASE: str = Field(default="0.008,0.003,0.0015,0.001",
                                            description="各档冲击成本基数（小票高、大票低）")
    LIQUIDITY_TIER_THRESHOLD: str = Field(default="0.005,0.01,0.01,0.02",
                                          description="各档冲击启用阈值（占ADV比例）")
    LIQUIDITY_TIER_CAP: str = Field(default="0.10,0.05,0.05,0.03",
                                    description="各档冲击成本上限")
    MAX_POSITION_PCT: float = Field(default=0.1, ge=0.01, le=1.0)
    PORTFOLIO_METHOD: str = Field(default="score_weighted")
    POINT_IN_TIME: bool = Field(default=True)
    # ── P1-5 单笔委托占成交量上限（按 ADV 成交额分档） ──
    MAX_ORDER_PCT: float = Field(default=0.30, ge=0.01, le=1.0,
                                 description="单笔委托占成交量上限（默认档，中流动性）")
    MAX_ORDER_PCT_HIGH: float = Field(default=0.20, ge=0.01, le=1.0,
                                      description="高流动性股单笔委托上限（日均成交额>1亿）")
    MAX_ORDER_PCT_LOW: float = Field(default=0.10, ge=0.01, le=1.0,
                                     description="低流动性股单笔委托上限（日均成交额<2000万）")
    # 0.1 成交时点模型（执行时序合规）：next_open 信号次日开盘（默认，A股T+1）/ vwap 信号次日VWAP
    # close 模式已移除（固有前视偏差：信号依赖当日收盘数据计算，以同日收盘价成交=先知交易）
    EXECUTION_MODEL: str = Field(default="next_open",
                                 description="成交时点模型: next_open 信号次日开盘成交（默认，符合A股T+1）/ vwap 信号次日VWAP成交")
    SIGNAL_PIPELINES: int = Field(default=3, ge=1, le=8)
    WFO_NUM_PATHS: int = Field(default=3, ge=1, le=10)

    # 贝叶斯优化预算
    BAYESIAN_N_INIT_SIGNAL: int = Field(default=15, ge=5, le=50)
    BAYESIAN_N_ITER_SIGNAL: int = Field(default=35, ge=10, le=200)
    BAYESIAN_N_INIT_PORTFOLIO: int = Field(default=50, ge=5, le=100)
    BAYESIAN_N_ITER_PORTFOLIO: int = Field(default=250, ge=20, le=500)
    # ── P2.4 预算制：总时间上限 + 连续无改进早停 ──
    BAYESIAN_TIME_BUDGET_SECONDS: int = Field(default=8 * 3600, ge=600, le=24 * 3600,
                                              description="WFO 总时间预算（秒），超时提前终止")
    BAYESIAN_MAX_NO_IMPROVE_WINDOWS: int = Field(default=3, ge=1, le=10,
                                                 description="连续无 OOS 改进窗口数，达到即提前终止本路径")
    # ── P2.1 CPCV：训练尾部净化（标签视界）+ 训练/OOS 禁运间隔 ──
    BAYESIAN_CPCV_PURGE_DAYS: int = Field(default=5, ge=0, le=60,
                                          description="CPCV 净化天数：训练窗口尾部剔除（前向收益标签视界）")
    BAYESIAN_CPCV_EMBARGO_DAYS: int = Field(default=3, ge=0, le=30,
                                            description="CPCV 禁运天数：训练结束与 OOS 开始之间的缓冲间隔")

    # 待寻优参数范围（逗号分隔：min,max,step）
    ATR_STOP_MULT_RANGE: str = "1.0,3.0,0.5"
    BOLL_NARROW_RATIO_RANGE: str = "0.6,1.2,0.1"
    CROSS_DECAY_DAYS_RANGE: str = "15,60,5"
    CONCLUSION_FULL_BULL_RANGE: str = "60,95,5"
    GOLDEN_CROSS_BONUS_RANGE: str = "5,20,5"
    DIVERGENCE_PENALTY_RANGE: str = "10,40,5"
    RISK_NONE_MULTIPLIER_RANGE: str = "0.5,2.0,0.25"
    BUY_THRESHOLD_RANGE: str = "5,30,5"
    MAX_HOLDINGS_RANGE: str = "3,20,1"
    # P0-7 ②：校准闭环 —— [BACKTEST_CALIBRATED] 覆写目标（默认取区间中位，
    # 与旧日频回退口径 int(17.5)=17 / int(11.5)=11 保持一致；校准后由
    # write_calibration_to_ini 写回、此处读取，供日频路径 EngineConfig 兜底使用）
    BUY_THRESHOLD: int = Field(default=12, ge=1, le=100,
                               description="买入评分阈值（校准覆写目标，WFO 未寻优时兜底）")
    MAX_HOLDINGS: int = Field(default=11, ge=0, le=100,
                              description="最大同时持仓数，0=不限制（校准覆写目标，WFO 未寻优时兜底）")

    @field_validator("OPTIMIZE_FREQUENCY")
    @classmethod
    def validate_frequency(cls, v: str) -> str:
        v_lower = v.lower().strip()
        if v_lower not in ("monthly", "quarterly", "initial"):
            msg = f"OPTIMIZE_FREQUENCY 必须为 monthly/quarterly/initial，收到 {v}"
            raise ValueError(msg)
        return v_lower

    @field_validator("CALENDAR_ALIGN_MODE")
    @classmethod
    def validate_calendar_align_mode(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in ("on", "off"):
            msg = f"CALENDAR_ALIGN_MODE 必须为 on/off，收到 {v}"
            raise ValueError(msg)
        return v_lower

    @field_validator("EXECUTION_MODEL")
    @classmethod
    def validate_execution_model(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower == "close":
            msg = f"EXECUTION_MODEL=close 已移除（固有前视偏差），请使用 next_open/vwap，收到 {v}"
            raise ValueError(msg)
        if v_lower not in ("next_open", "vwap"):
            msg = f"EXECUTION_MODEL 必须为 next_open/vwap，收到 {v}"
            raise ValueError(msg)
        return v_lower

    # ── A2 失败快照持久化 ──
    SNAPSHOT_ENABLED: bool = Field(default=True,
                                   description="窗口计算无效/异常时持久化失败快照（Task A2）")
    SNAPSHOT_MAX_ROWS: int = Field(default=200, ge=50, le=5000,
                                   description="快照 OHLCV 截断行数（最近 N 行）")
    SNAPSHOT_RETENTION_DAYS: int = Field(default=14, ge=1, le=365,
                                         description="失败快照保留天数，过期自动清理并告警")
    # ── 窗口预检与容错（指标计算前判断序列可计算性） ──
    PRECHECK_MODE: str = Field(default="RELAX",
                               description="窗口预检模式: STRICT 任何可疑一律SKIP / RELAX 硬失败SKIP+可修复填充+软问题放行 / OFF 关闭")
    # ── Task F 交易日历与停牌标志对齐 ──
    CALENDAR_ALIGN_MODE: str = Field(default="on",
                                     description="交易日历对齐: on 合并时按官方日历对齐（is_trading/is_suspended 标志 + 停牌比例日历口径 SKIP + 引擎日轴=交易所日历） / off 回退老版合并逻辑（无标志、启发式停牌检测、数据日轴）")
    CALENDAR_TTL_HOURS: float = Field(default=24.0, ge=1.0, le=24 * 30,
                                      description="官方交易日历本地缓存有效期（小时），过期后维护时重新拉取")

    # ── 指标计算降级（min_periods / 置信度标签） ──
    INDICATOR_DEGRADATION: str = Field(default="RELAX",
                                       description="指标降级模式: STRICT 原周期全窗计算(头部NaN,原行为) / RELAX 缩窗计算并标low_confidence / SKIP 标低置信由策略层跳过")
    INDICATOR_DEGRADATION_LOW_ACTION: str = Field(default="skip",
                                                  description="低置信度信号处理: skip 不下单 / low_weight 按系数降权")
    INDICATOR_DEGRADATION_LOW_WEIGHT: float = Field(default=0.5, ge=0.01, le=1.0,
                                                    description="低置信度信号降权系数")
    # ── 涨跌停撮合约束（可成交量规则，提升成交模拟真实度） ──
    SIMULATE_LIMIT_UP_DOWN: bool = Field(default=True,
                                         description="涨跌停撮合约束: true 触板日按可成交量比例部分成交/未成交 / false 回退简化撮合（触板一律禁止买卖）")
    LIMIT_SEAL_RATIO: float = Field(default=0.05, ge=0.0, le=1.0,
                                    description="一字板（开=收=限价）可成交量比例")
    LIMIT_TRADABLE_RATIO: float = Field(default=0.30, ge=0.0, le=1.0,
                                        description="开盘触板后炸板（open≥限价, close<限价）可成交量比例")
    LIMIT_INTRADAY_RATIO: float = Field(default=0.10, ge=0.0, le=1.0,
                                        description="盘中冲板（open<限价, high≥限价）可成交量比例")
    LIMIT_SEAL_DECAY: float = Field(default=0.5, ge=0.0, le=1.0,
                                    description="连续板每板可成交量衰减系数")
    # ── P0-6 ⑥：开盘集合竞价成交率分档（封单量/可成交量代理） ──
    AUCTION_FILL_RATIO: float = Field(default=0.12, ge=0.0, le=1.0,
                                      description="开盘价触板日集合竞价可成交量比例上限（= 当日量 × min(触板档, 该值)）")
    # ── 经验填充模型（技术债修复）：历史日线 V_t/V_prev 分位数替代固定比例 ──
    LIMIT_RATIO_MODE: str = Field(default="fixed",
                                  description="涨跌停可成交量比例来源: fixed 固定比例常量 / empirical_median 历史经验中位数（中性口径） / empirical_p10 历史经验10%分位（worst-case 保守口径）")
    LIMIT_CALIB_MIN_SAMPLES: int = Field(default=20, ge=1, le=1000000,
                                         description="经验填充模型单元格最少样本数，不足回退 fixed 档（防稀疏样本噪声）")
    LIMIT_STRESS_ENABLED: bool = Field(default=True,
                                       description="涨跌停专项压力测试（一字涨停/竞价触板/炸板高发窗口 worst-case 成本报告）")
    # ── P0-6 ⑤：市场状态仓位调节（客观状态变量，替代旧评分中位数口径） ──
    REGIME_RET20_FULL: float = Field(default=0.02, ge=-1.0, le=1.0,
                                     description="指数20日收益（全市场 ret_20d 中位数代理）≥ 此值 → 全仓倍率")
    REGIME_RET20_HALF: float = Field(default=-0.02, ge=-1.0, le=1.0,
                                     description="指数20日收益 ≥ 此值（且非高波动）→ 半仓倍率")
    REGIME_VOL_PCT_MAX: float = Field(default=0.8, ge=0.0, le=1.0,
                                      description="市场波动率分位（过去250交易日）> 此值 → 高波动，仓位压制到最低倍率")
    # ── 复牌跳空（0.6）：停牌后复牌日开盘大幅跳空（补涨兑现卖出 / 补跌标记 / 追高禁买） ──
    RESUME_GAP_UP: float = Field(default=0.05, ge=0.0, le=1.0,
                                 description="复牌高开≥该比例（相对停牌前收盘）→ 开盘兑现卖出 + 当日禁买（追高）；0=关闭")
    RESUME_GAP_DOWN: float = Field(default=0.05, ge=0.0, le=1.0,
                                   description="复牌低开≤-该比例 → 日志标记（风控卖出照常）；0=关闭")
    # ── 统一成本（单一来源 CostModel，覆盖原硬编码印花税分段） ──
    HANDLING_FEE_RATE: float = Field(default=0.0000341, ge=0.0, le=0.01,
                                     description="经手费（双边，0.00341%）")
    CSRC_FEE_RATE: float = Field(default=0.00002, ge=0.0, le=0.01,
                                 description="证管费（双边，0.002%）")
    STAMP_TAX_SEGMENTS: str = Field(
        default="2023-08-28:0.0005;2000-01-01:0.001",
        description="印花税日期分段表（date:rate;...，卖出单向；最晚≤交易日档生效）",
    )
    TRANSFER_FEE_SEGMENTS: str = Field(
        default="2022-04-29:0.00001;2000-01-01:0.00002",
        description="过户费日期分段表（date:rate;...，双边收取；2022-04-29 起万0.1，此前万0.2；最晚≤交易日档生效）",
    )

    def parse_range(self, key: str) -> tuple[float, float, float]:
        raw = getattr(self, key.upper(), "")
        if not raw or not raw.strip():
            raise ValueError(f"{key} 未配置，跳过")
        parts = [float(x.strip()) for x in raw.split(",")]
        if len(parts) != 3:
            msg = f"{key} 格式应为 min,max,step，收到 {raw!r}"
            raise ValueError(msg)
        return (parts[0], parts[1], parts[2])


class DistributionConfig(BaseModel):
    """东方财富API配置模型"""

    API_TOKEN: str = Field(default="", description="东方财富数据中心 API Token")


class TradingCostConfig(BaseModel):
    """A股交易成本配置模型（统一成本来源，供回测引擎 CostModel 与跟仓回测共用）"""

    COMMISSION_RATE: float = Field(default=0.0003, ge=0, le=0.01,
                                    description="佣金费率（默认万三）")
    STAMP_TAX_RATE: float = Field(default=0.001, ge=0, le=0.01,
                                    description="印花税费率（卖出收取，2023.8 起万五）")
    TRANSFER_FEE_RATE: float = Field(default=0.00001, ge=0, le=0.001,
                                      description="过户费率（双向，默认万0.1）")
    HANDLING_FEE_RATE: float = Field(default=0.0000341, ge=0, le=0.01,
                                     description="经手费（双边，0.00341%）")
    CSRC_FEE_RATE: float = Field(default=0.00002, ge=0, le=0.01,
                                 description="证管费（双边，0.002%）")
    STAMP_TAX_SEGMENTS: str = Field(
        default="2023-08-28:0.0005;2000-01-01:0.001",
        description="印花税日期分段表（date:rate;...，卖出单向）",
    )
    TRANSFER_FEE_SEGMENTS: str = Field(
        default="2022-04-29:0.00001;2000-01-01:0.00002",
        description="过户费日期分段表（date:rate;...，双边收取；2022-04-29 起万0.1，此前万0.2）",
    )


class PositionBacktestConfig(BaseModel):
    """跟仓回测配置模型"""

    POOL_FILE_PATH: str = Field(default="证券交割单.xlsx", description="证券交割单文件路径（XLSX格式）")


class PortfolioOptimizerConfig(BaseModel):
    """P4 组合优化器配置模型（数学规划驱动，替代 Top-K 等权）"""

    METHOD: str = Field(
        default="topk_equal",
        description="优化方法: mean_variance / min_variance / risk_parity / topk_equal(兼容旧版)",
    )
    RISK_AVERSION: float = Field(
        default=2.0, ge=0.01, le=20.0,
        description="风险厌恶系数 (γ, 越大越保守)",
    )
    TURNOVER_PENALTY: float = Field(
        default=0.001, ge=0.0, le=0.01,
        description="换手率惩罚系数 (λ_TC)",
    )
    MAX_WEIGHT: float = Field(
        default=0.10, ge=0.01, le=0.50,
        description="单票权重上限 (w_max)",
    )
    COV_LOOKBACK: int = Field(
        default=60, ge=10, le=250,
        description="协方差估计窗口 (交易日)",
    )
    SHRINKAGE: bool = Field(
        default=True,
        description="协方差收缩 (Ledoit-Wolf)",
    )
    INDUSTRY_NEUTRAL: bool = Field(
        default=False,
        description="行业中性约束",
    )
    INDUSTRY_DEVIATION: float = Field(
        default=0.05, ge=0.0, le=0.20,
        description="行业暴露偏离上限",
    )
    MAX_HOLDINGS: int = Field(
        default=0, ge=0, le=200,
        description="最大持仓数 (0=不限制)",
    )
    TARGET_CASH_RATIO: float = Field(
        default=0.0, ge=0.0, le=0.50,
        description="目标现金比例",
    )
    SOLVE_TIMEOUT: float = Field(
        default=5.0, ge=0.1, le=60.0,
        description="求解超时 (秒)",
    )
    # P3-3：优化器日志详细度（WFO路径下降噪）
    VERBOSE: bool = Field(
        default=False,
        description="优化器日志详细度（True=输出所有debug/info；False=仅warning以上）",
    )
    # ── 超参数寻优范围 ──
    OPTIMIZER_RISK_AVERSION_RANGE: str = Field(default="0.5,5.0,0.5", description="风险厌恶系数寻优范围")
    OPTIMIZER_TURNOVER_PENALTY_RANGE: str = Field(default="0.0005,0.003,0.0005", description="换手率惩罚寻优范围")
    OPTIMIZER_MAX_WEIGHT_RANGE: str = Field(default="0.05,0.15,0.01", description="单票权重上限寻优范围")
    OPTIMIZER_COV_LOOKBACK_RANGE: str = Field(default="30,120,10", description="协方差窗口寻优范围")

    def parse_range(self, key: str) -> tuple[float, float, float]:
        raw = getattr(self, key.upper(), "")
        if not raw or not raw.strip():
            raise ValueError(f"{key} 未配置，跳过")
        parts = [float(x.strip()) for x in raw.split(",")]
        if len(parts) != 3:
            msg = f"{key} 格式应为 min,max,step，收到 {raw!r}"
            raise ValueError(msg)
        return (parts[0], parts[1], parts[2])


class PositionSizingConfig(BaseModel):
    """仓位管理配置模型"""

    MAX_SINGLE_POSITION: float = Field(default=0.33, ge=0.0, le=1.0,
                                       description="最大单票仓位")
    KELLY_FRACTION: float = Field(default=0.25, ge=0.0, le=1.0,
                                  description="半凯利系数")
    DEFAULT_WIN_RATE: float = Field(default=0.55, ge=0.0, le=1.0,
                                    description="默认胜率假设")
    POSITION_A: float = Field(default=0.30, ge=0.0, le=1.0,
                              description="A级基础仓位")
    POSITION_B: float = Field(default=0.15, ge=0.0, le=1.0,
                              description="B级基础仓位")
    POSITION_C: float = Field(default=0.05, ge=0.0, le=1.0,
                              description="C级基础仓位")
    MAX_INDUSTRY_EXPOSURE: float = Field(default=0.30, ge=0.0, le=1.0,
                                         description="最大行业集中度")
    RISK_BUDGET: float = Field(default=0.02, ge=0.001, le=0.10,
                                description="波动率风险预算")
    MAX_DAY_TURNOVER: float = Field(default=0.20, ge=0.0, le=1.0,
                                    description="单日最大双边换手率")
    RISK_AVERSION: float = Field(default=1.0, ge=0.01, le=10.0,
                                 description="风险厌恶系数（越大越保守）")
    RISK_NONE_MULTIPLIER: float = Field(default=1.0, ge=0.1, le=5.0,
                                         description="NONE 风险等级仓位系数")


class ResearchReportFilterConfig(BaseModel):
    """研报过滤配置"""

    ENABLE_RESEARCH_REPORT_FILTER: bool = Field(default=False, description="是否启用研报过滤")
    RESEARCH_REPORT_MIN_COUNT: int = Field(default=1, ge=1, le=100,
                                           description="买入评级最低次数")


class ApiConfig(BaseModel):
    """API 服务配置模型"""

    ENABLED: bool = Field(default=False, description="是否启用 API 服务")
    ALERT_WEBHOOK_URL: str = Field(default="", description="告警 Webhook URL")
    ALERT_CHANNEL: str = Field(default="generic", description="告警渠道(generic/wecom/feishu/dingtalk)")
    ALERT_ON_FAILURE: bool = Field(default=True, description="失败时告警")
    ALERT_ON_SUCCESS: bool = Field(default=False, description="成功时告警")


class AppConfig(BaseSettings):
    """应用配置主模型"""

    model_config = SettingsConfigDict(
        env_prefix="BAISYS_", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    api: ApiConfig = ApiConfig()
    database: DatabaseConfig
    system: SystemConfig
    logging: LoggingConfig
    multi_head_arrangement: MultiHeadArrangementConfig
    filter_rules: FilterRulesConfig
    fund_flow: FundFlowConfig
    technical_indicators: TechnicalIndicatorsConfig
    column_aliases: ColumnAliasesConfig
    full_bull_scoring: FullBullScoringConfig
    user_focus_stocks: UserFocusStocksConfig
    asharehub: AShareHubConfig
    macro_filter: MacroFilterConfig
    regime_detection: RegimeDetectionConfig
    divergence: DivergenceConfig
    scoring_params: ScoringParamsConfig
    technical_constants: TechnicalConstantsConfig
    position_sizing: PositionSizingConfig
    backtest: BacktestConfig
    position_backtest: PositionBacktestConfig
    trading_cost: TradingCostConfig
    distribution: DistributionConfig
    multi_factor_alpha: MultiFactorAlphaConfig
    portfolio_optimizer: PortfolioOptimizerConfig
    research_report_filter: ResearchReportFilterConfig = ResearchReportFilterConfig()


class Config:
    """
    配置管理器（INI→Pydantic，自动类型转换）

    读取 config.ini 并委托 AppConfig（Pydantic）做类型校验与转换。
    所有历史平铺属性改为 @property 委托，数据归一在 app_config。
    """

    def __init__(self, config_file: str = "config.ini") -> None:
        self.config_file = config_file
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"配置文件未找到: {os.path.abspath(self.config_file)}")

        self._load_config()
        self._ensure_directories()

    # ── 加载：INI → Pydantic（单次操作，零手动类型转换） ─────────────────

    def _section_upper(self, name: str) -> dict[str, str]:
        """读取 INI 节并转大写 key，适配 Pydantic UPPER_CASE 字段。"""
        return {k.upper(): v for k, v in dict(self._raw_section(name)).items()}

    @staticmethod
    def _strip_inline(v: str) -> str:
        """递归剥离 # 和 ; inline comment（兼容 Python 3.12+ 默认 inline_comment_prefixes=()）。"""
        for sep in ("#", ";"):
            idx = v.find(sep)
            if idx >= 0 and (idx == 0 or v[idx - 1] in (" ", "\t")):
                v = v[:idx].rstrip()
        return v

    def _raw_section(self, name: str) -> dict[str, str]:
        """安全读取 INI 节，不存在时返回空 dict。"""
        try:
            return {k: self._strip_inline(v) for k, v in dict(self._cp[name]).items()}
        except KeyError:
            return {}

    def _load_config(self) -> None:
        import configparser

        self._cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        self._cp.read(self.config_file, encoding="utf-8")

        from UtilsManager.ConfigCipher import ConfigCipher

        # DATABASE（lowercase 字段名 + 敏感字段解密）
        key_path = self._cp["DATABASE"].get("encryption_key_path", fallback=None)
        if key_path:
            ConfigCipher.default_key_path = key_path
        db_raw = self._raw_section("DATABASE")
        for enc_key in ("password", "host", "port", "db_name"):
            db_raw[enc_key] = ConfigCipher.maybe_decrypt(db_raw.get(enc_key, ""))

        # COLUMN_ALIASES（lowercase 字段名）
        col_raw = self._raw_section("COLUMN_ALIASES")

        # ASHAREHUB（API_KEY 需解密）
        ah_raw = self._section_upper("ASHAREHUB")
        if ah_raw:
            ah_raw["API_KEY"] = ConfigCipher.maybe_decrypt(ah_raw.get("API_KEY", ""))

        # DISTRIBUTION（API_TOKEN 需解密）
        dist_raw = self._section_upper("DISTRIBUTION")
        if dist_raw:
            dist_raw["API_TOKEN"] = ConfigCipher.maybe_decrypt(dist_raw.get("API_TOKEN", ""))

        # 装配 AppConfig（Pydantic field_validator 自动处理逗号/bool/int/float 转换）
        self.app_config = AppConfig(
            database=DatabaseConfig(**db_raw),
            system=SystemConfig(**self._section_upper("SYSTEM")),
            logging=LoggingConfig(**self._section_upper("LOGGING")),
            multi_head_arrangement=MultiHeadArrangementConfig(**self._section_upper("MULTI_HEAD_ARRANGEMENT")),
            filter_rules=FilterRulesConfig(**self._section_upper("FILTER_RULES")),
            fund_flow=FundFlowConfig(**self._section_upper("FUND_FLOW")),
            technical_indicators=TechnicalIndicatorsConfig(**self._section_upper("TECHNICAL_INDICATORS")),
            column_aliases=ColumnAliasesConfig(**col_raw),
            full_bull_scoring=FullBullScoringConfig(**self._section_upper("FULL_BULL_SCORING")),
            user_focus_stocks=UserFocusStocksConfig(**self._section_upper("USER_FOCUS_STOCKS")),
            asharehub=AShareHubConfig(**ah_raw),
            macro_filter=MacroFilterConfig(**self._section_upper("MACRO_FILTER")),
            regime_detection=RegimeDetectionConfig(**self._section_upper("REGIME_DETECTION")),
            divergence=DivergenceConfig(**self._section_upper("DIVERGENCE")),
            scoring_params=ScoringParamsConfig(**self._section_upper("SCORING_PARAMS")),
            technical_constants=TechnicalConstantsConfig(**self._section_upper("TECHNICAL_CONSTANTS")),
            position_sizing=PositionSizingConfig(**self._section_upper("POSITION_SIZING")),
            backtest=BacktestConfig(**self._section_upper("BACKTEST")),
            position_backtest=PositionBacktestConfig(**self._section_upper("POSITION_BACKTEST")),
            trading_cost=TradingCostConfig(**self._section_upper("TRADING_COST")),
            distribution=DistributionConfig(**dist_raw),
            multi_factor_alpha=MultiFactorAlphaConfig(**self._section_upper("MULTI_FACTOR_ALPHA")),
            portfolio_optimizer=PortfolioOptimizerConfig(**self._section_upper("PORTFOLIO_OPTIMIZER")),
            research_report_filter=ResearchReportFilterConfig(**self._section_upper("RESEARCH_REPORT_FILTER")),
            api=ApiConfig(**self._section_upper("API")),
        )

        # ── 回测自动校准参数覆写 ──
        # [BACKTEST_CALIBRATED] 将校准参数（含 P0-7 ② 补齐的 buy_threshold/
        # max_holdings）覆写到各自子模型，与 BackTrading/calibration.py 的
        # CALIB_PARAM_MAP 保持一一对应，实现 INI 统一分组、写读闭环。
        bt_cal = self._section_upper("BACKTEST_CALIBRATED")
        if bt_cal:
            sc = self.app_config.scoring_params
            fr = self.app_config.filter_rules
            rd = self.app_config.regime_detection
            ps = self.app_config.position_sizing
            if "BOLL_NARROW_RATIO" in bt_cal:
                rd.BOLL_NARROW_RATIO = float(bt_cal["BOLL_NARROW_RATIO"])
            if "CROSS_DECAY_DAYS" in bt_cal:
                sc.CROSS_DECAY_DAYS = int(bt_cal["CROSS_DECAY_DAYS"])
            if "ATR_STOP_MULT" in bt_cal:
                sc.ATR_STOP_MULT = float(bt_cal["ATR_STOP_MULT"])
            # VAEO 波动率自适应退出参数覆写
            if "LEARNED_T1_MULT" in bt_cal:
                sc.LEARNED_T1_MULT = float(bt_cal["LEARNED_T1_MULT"])
            if "LEARNED_T2_MULT" in bt_cal:
                sc.LEARNED_T2_MULT = float(bt_cal["LEARNED_T2_MULT"])
            if "LIQ_VETO_RATIO" in bt_cal:
                fr.LIQ_VETO_RATIO = float(bt_cal["LIQ_VETO_RATIO"])
            if "KELLY_FRACTION" in bt_cal:
                ps.KELLY_FRACTION = float(bt_cal["KELLY_FRACTION"])
            if "POSITION_A" in bt_cal:
                ps.POSITION_A = float(bt_cal["POSITION_A"])
            if "POSITION_B" in bt_cal:
                ps.POSITION_B = float(bt_cal["POSITION_B"])
            if "POSITION_C" in bt_cal:
                ps.POSITION_C = float(bt_cal["POSITION_C"])
            if "CONCLUSION_FULL_BULL" in bt_cal:
                self.app_config.full_bull_scoring.CONCLUSION_FULL_BULL = int(bt_cal["CONCLUSION_FULL_BULL"])
            if "GOLDEN_CROSS_BONUS" in bt_cal:
                sc.GOLDEN_CROSS_BONUS = int(bt_cal["GOLDEN_CROSS_BONUS"])
            if "DIVERGENCE_PENALTY" in bt_cal:
                sc.DIVERGENCE_PENALTY = int(bt_cal["DIVERGENCE_PENALTY"])
            if "RISK_NONE_MULTIPLIER" in bt_cal:
                ps.RISK_NONE_MULTIPLIER = float(bt_cal["RISK_NONE_MULTIPLIER"])
            # P0-7 ②：校准闭环补齐 —— buy_threshold/max_holdings 曾只写不读，
            # 日频路径不生效；现覆写到 backtest 配置供 EngineConfig 兜底读取。
            # 注：历史版本可能以 "17.0" 浮点落盘，此处 int(float()) 容错兼容，
            # 新写入已由 calibration.py 类型断言保证整值。
            if "BUY_THRESHOLD" in bt_cal:
                self.app_config.backtest.BUY_THRESHOLD = int(float(bt_cal["BUY_THRESHOLD"]))
            if "MAX_HOLDINGS" in bt_cal:
                self.app_config.backtest.MAX_HOLDINGS = int(float(bt_cal["MAX_HOLDINGS"]))

    def reload(self) -> None:
        """热重载配置文件，保留 config_file 路径。"""
        self._load_config()
        self._ensure_directories()

    def watch(self, interval: float = 1.0, callback: callable = None) -> None:
        """
        启动配置文件监控（轮询模式，跨平台兼容）。

        Args:
            interval: 轮询间隔（秒）
            callback: 文件变更时的回调函数，接收 (config: Config) 参数

        Note:
            这是一个阻塞调用，通常在单独线程中运行。
            建议在主线程之外启动：threading.Thread(target=config.watch, daemon=True).start()
        """
        import time
        last_mtime = os.path.getmtime(self.config_file)
        while True:
            time.sleep(interval)
            try:
                current_mtime = os.path.getmtime(self.config_file)
                if current_mtime > last_mtime:
                    last_mtime = current_mtime
                    self.reload()
                    if callback:
                        callback(self)
            except FileNotFoundError:
                pass
            except Exception:
                pass

    # ── 向后兼容属性（只读委托至 app_config） ──────────────────────────

    # 数据库
    @property
    def DB_USER(self) -> str: return self.app_config.database.user

    @property
    def DB_PASSWORD(self) -> str: return self.app_config.database.password

    @property
    def DB_HOST(self) -> str: return self.app_config.database.host

    @property
    def DB_PORT(self) -> str: return self.app_config.database.port

    @property
    def DB_NAME(self) -> str: return self.app_config.database.db_name

    # 系统
    @property
    def HOME_DIRECTORY(self) -> str: return self.app_config.system.HOME_DIRECTORY

    @property
    def CACHE_DIRECTORY(self) -> str:
        return os.path.join(self.HOME_DIRECTORY, "cache")

    @property
    def TEMP_DATA_DIRECTORY(self) -> str:
        return os.path.join(self.app_config.system.HOME_DIRECTORY, self.app_config.system.TEMP_DATA_DIR)

    @property
    def MAX_WORKERS(self) -> int: return self.app_config.system.MAX_WORKERS

    # D1 分片执行（shard）
    @property
    def SHARD_MODE(self) -> str: return self.app_config.system.SHARD_MODE

    @property
    def SHARD_SYMBOL_BATCH_SIZE(self) -> int: return self.app_config.system.SHARD_SYMBOL_BATCH_SIZE

    @property
    def SHARD_MAX_ATTEMPTS(self) -> int: return self.app_config.system.SHARD_MAX_ATTEMPTS

    @property
    def SHARD_MAX_WORKERS(self) -> int: return self.app_config.system.SHARD_MAX_WORKERS

    # Task E 幂等输出
    @property
    def OUTPUT_WRITE_MODE(self) -> str: return self.app_config.system.OUTPUT_WRITE_MODE

    @property
    def DATA_FETCH_RETRIES(self) -> int: return self.app_config.system.DATA_FETCH_RETRIES

    @property
    def DATA_FETCH_DELAY(self) -> int: return self.app_config.system.DATA_FETCH_DELAY

    @property
    def STOCK_BASIC_INFO_EXPIRE_DAYS(self) -> int: return self.app_config.system.STOCK_BASIC_INFO_EXPIRE_DAYS

    @property
    def SIGNAL_PROCESSING_PROCESSES(self) -> int: return self.app_config.system.SIGNAL_PROCESSING_PROCESSES

    # 日志
    @property
    def LOG_LEVEL(self) -> str: return self.app_config.logging.LOG_LEVEL

    @property
    def LOG_DIR(self) -> str:
        return os.path.join(self.app_config.system.HOME_DIRECTORY, self.app_config.logging.LOG_DIR)

    # 多头排列
    @property
    def FULL_BULL_THRESHOLD(self) -> int: return self.app_config.multi_head_arrangement.FULL_BULL_THRESHOLD

    @property
    def TREND_ACCELERATION_THRESHOLD(self) -> int: return self.app_config.multi_head_arrangement.TREND_ACCELERATION_THRESHOLD

    @property
    def TREND_OSCILLATION_THRESHOLD(self) -> int: return self.app_config.multi_head_arrangement.TREND_OSCILLATION_THRESHOLD


    @property
    def MOVING_AVERAGE_PERIODS(self) -> list[int]: return self.app_config.multi_head_arrangement.MOVING_AVERAGE_PERIODS

    # 过滤规则
    @property
    def ENABLE_WEAK_STOCK_FILTER(self) -> bool: return self.app_config.filter_rules.ENABLE_WEAK_STOCK_FILTER

    @property
    def EXEMPT_LEVELS(self) -> list[str]: return self.app_config.filter_rules.EXEMPT_LEVELS

    @property
    def FILTER_PCT_HARD(self) -> float: return self.app_config.filter_rules.INDUSTRY_PCT_HARD

    @property
    def FILTER_PCT_D(self) -> float: return self.app_config.filter_rules.INDUSTRY_PCT_D

    @property
    def FILTER_PCT_EXEMPT(self) -> float: return self.app_config.filter_rules.INDUSTRY_PCT_EXEMPT

    # 资金流
    @property
    def FUND_FLOW_PERIODS(self) -> list[int]: return self.app_config.fund_flow.FUND_FLOW_PERIODS

    # 技术指标
    @property
    def MACD_PARAMS(self) -> tuple[int, int, int]: return self.app_config.technical_indicators.MACD_PARAMS

    # 列名别名（需 parse_aliases 解析）
    @property
    def CODE_ALIASES(self) -> dict[str, str]: return parse_aliases(self.app_config.column_aliases.code_aliases)

    @property
    def NAME_ALIASES(self) -> dict[str, str]: return parse_aliases(self.app_config.column_aliases.name_aliases)

    @property
    def PRICE_ALIASES(self) -> dict[str, str]: return parse_aliases(self.app_config.column_aliases.price_aliases)

    # 研报
    @property
    def ENABLE_RESEARCH_REPORT_FILTER(self) -> bool: return self.app_config.research_report_filter.ENABLE_RESEARCH_REPORT_FILTER

    @property
    def RESEARCH_REPORT_MIN_COUNT(self) -> int: return self.app_config.research_report_filter.RESEARCH_REPORT_MIN_COUNT

    # 自选股
    @property
    def USER_FOCUS_STOCKS(self) -> str: return self.app_config.user_focus_stocks.USER_FOCUS_STOCKS

    # AShareHub
    @property
    def ASHAREHUB_API_KEY(self) -> str: return self.app_config.asharehub.API_KEY

    @property
    def MONEYFLOW_RETRY(self) -> int: return self.app_config.asharehub.MONEYFLOW_RETRY

    @property
    def MONEYFLOW_PAGE_DELAY(self) -> float: return self.app_config.asharehub.MONEYFLOW_PAGE_DELAY

    # 宏观过滤
    @property
    def ENABLE_MACRO_FILTER(self) -> bool: return self.app_config.macro_filter.ENABLE_MACRO_FILTER

    # 回测
    @property
    def BACKTEST_START_DATE(self) -> str: return self.app_config.backtest.BACKTEST_START_DATE

    @property
    def OUT_OF_SAMPLE_DAYS(self) -> int: return self.app_config.backtest.OUT_OF_SAMPLE_DAYS

    @property
    def HOLDOUT_RATIO(self) -> float: return self.app_config.backtest.HOLDOUT_RATIO

    @property
    def SIGNAL_PIPELINES(self) -> int: return self.app_config.backtest.SIGNAL_PIPELINES

    # Task F 交易日历与停牌标志对齐
    @property
    def CALENDAR_ALIGN_MODE(self) -> str: return self.app_config.backtest.CALENDAR_ALIGN_MODE

    @property
    def CALENDAR_TTL_HOURS(self) -> float: return self.app_config.backtest.CALENDAR_TTL_HOURS

    # 0.1 成交时点模型（执行时序合规）
    @property
    def EXECUTION_MODEL(self) -> str: return self.app_config.backtest.EXECUTION_MODEL

    # 跟仓回测
    @property
    def POOL_FILE_PATH(self) -> str: return self.app_config.position_backtest.POOL_FILE_PATH

    # 东方财富 API
    @property
    def DISTRIBUTION_API_TOKEN(self) -> str: return self.app_config.distribution.API_TOKEN

    # 交易成本
    @property
    def TRADING_COST_PARAMS(self) -> dict:
        t = self.app_config.trading_cost
        return {
            "commission_rate": t.COMMISSION_RATE,
            "stamp_tax_rate": t.STAMP_TAX_RATE,
            "transfer_fee_rate": t.TRANSFER_FEE_RATE,
            "handling_fee_rate": t.HANDLING_FEE_RATE,
            "csrc_fee_rate": t.CSRC_FEE_RATE,
            "stamp_tax_segments": t.STAMP_TAX_SEGMENTS,
            "transfer_fee_segments": t.TRANSFER_FEE_SEGMENTS,
        }

    # 多因子 Alpha
    @property
    def MULTI_FACTOR_ALPHA_ENABLED(self) -> bool:
        return self.app_config.multi_factor_alpha.ENABLED

    @property
    def FINANCIAL_QUALITY_CACHE_DAYS(self) -> int:
        return self.app_config.multi_factor_alpha.FINANCIAL_QUALITY_CACHE_DAYS

    @property
    def FINANCIAL_QUALITY_BATCH_SIZE(self) -> int:
        return self.app_config.multi_factor_alpha.FINANCIAL_QUALITY_BATCH_SIZE

    @property
    def FINANCIAL_QUALITY_BATCH_SLEEP(self) -> int:
        return self.app_config.multi_factor_alpha.FINANCIAL_QUALITY_BATCH_SLEEP

    @property
    def FINANCIAL_QUALITY_FILE_CACHE_DAYS(self) -> int:
        return self.app_config.multi_factor_alpha.FINANCIAL_QUALITY_FILE_CACHE_DAYS

    @property
    def FUNDAMENTALS_RETRY(self) -> int:
        return self.app_config.multi_factor_alpha.FUNDAMENTALS_RETRY

    @property
    def ENABLE_FUNDAMENTALS(self) -> bool:
        return self.app_config.asharehub.ENABLE_FUNDAMENTALS

    # ── Dict 聚合属性（供 SignalManager / DataProcessingService 等使用） ──

    @property
    def FULL_BULL_WEIGHTS(self) -> dict:
        f = self.app_config.full_bull_scoring
        return {
            "MACD趋势": f.WEIGHT_ZERO_AXIS,
            "金叉信号": f.WEIGHT_STRATEGY_GOLDEN,
            "柱状动能": f.WEIGHT_MOMENTUM,
            "DIF斜率": f.WEIGHT_DIF_SLOPE,
            "背离信号": f.WEIGHT_DIVERGENCE,
            "量价配合": f.WEIGHT_VOLUME_PRICE,
            "K线形态": f.WEIGHT_KLINE_PATTERN,
        }

    @property
    def FULL_BULL_THRESHOLDS(self) -> dict:
        f = self.app_config.full_bull_scoring
        return {
            "fully_bull": f.CONCLUSION_FULL_BULL,
            "bullish": f.CONCLUSION_BULLISH,
            "oscillate": f.CONCLUSION_OSCILLATE,
        }

    @property
    def RULE_THRESHOLDS(self) -> dict:
        f = self.app_config.full_bull_scoring
        return {
            "divergence": f.RULE_DIVERGENCE_THRESHOLD,
            "winner_rate_high": f.RULE_WINNER_RATE_HIGH,
            "winner_rate_low": f.RULE_WINNER_RATE_LOW,
            "cost_resistance_ratio": f.RULE_COST_RESISTANCE_RATIO,
            "chip_concentrated_ratio": f.RULE_CHIP_CONCENTRATED_RATIO,
            "price_new_high_days": f.RULE_PRICE_NEW_HIGH_DAYS,
            "liq_veto_ratio": self.app_config.filter_rules.LIQ_VETO_RATIO,
        }

    @property
    def REGIME_DETECTION(self) -> dict:
        r = self.app_config.regime_detection
        return {
            "boll_narrow_ratio": r.BOLL_NARROW_RATIO,
            "oscillation_hist_std_ratio": r.OSCILLATION_HIST_STD_RATIO,
            "top_risk_ma20_deviation": r.TOP_RISK_MA20_DEVIATION,
            "oscillation_min_bars": r.OSCILLATION_MIN_BARS,
            "reversal_lookback": r.REVERSAL_LOOKBACK,
        }

    @property
    def DIVERGENCE_PARAMS(self) -> dict:
        d = self.app_config.divergence
        return {
            "base_distance": d.BASE_DISTANCE,
            "strength_threshold": d.STRENGTH_THRESHOLD,
            "decay_half_life": d.DECAY_HALF_LIFE,
            "slope_window": d.SLOPE_WINDOW,
        }

    @property
    def SCORING_PARAMS(self) -> dict:
        s = self.app_config.scoring_params
        return {
            "cross_decay_days": s.CROSS_DECAY_DAYS,
            "cross_decay_min": s.CROSS_DECAY_MIN,
            "kline_decay_days": s.KLINE_DECAY_DAYS,
            "kline_decay_min": s.KLINE_DECAY_MIN,
            "vol_norm_denominator": s.VOL_NORM_DENOMINATOR,
            "atr_stop_mult": s.ATR_STOP_MULT,
            "trailing_stop_high_ratio": s.TRAILING_STOP_HIGH_RATIO,
            "trailing_stop_lookback": s.TRAILING_STOP_LOOKBACK,
            "trailing_stop_high_lookback": s.TRAILING_STOP_HIGH_LOOKBACK,
            "expected_return_lookback": s.EXPECTED_RETURN_LOOKBACK,
            "golden_cross_bonus": s.GOLDEN_CROSS_BONUS,
            "divergence_penalty": s.DIVERGENCE_PENALTY,
            # VAEO 波动率自适应退出参数
            "learned_t1_mult": s.LEARNED_T1_MULT,
            "learned_t2_mult": s.LEARNED_T2_MULT,
        }

    @property
    def TECHNICAL_CONSTANTS(self) -> dict:
        t = self.app_config.technical_constants
        return {
            "atr_length": t.ATR_LENGTH,
            "adx_length": t.ADX_LENGTH,
            "rsi_length": t.RSI_LENGTH,
            "boll_length": t.BOLL_LENGTH,
            "boll_std": t.BOLL_STD,
            "stoch_k": t.STOCH_K,
            "stoch_d": t.STOCH_D,
            "kline_scan_window": t.KLINE_SCAN_WINDOW,
        }

    @property
    def POSITION_SIZING(self) -> dict:
        p = self.app_config.position_sizing
        f = self.app_config.filter_rules
        return {
            "max_single_position": p.MAX_SINGLE_POSITION,
            "kelly_fraction": p.KELLY_FRACTION,
            "default_win_rate": p.DEFAULT_WIN_RATE,
            "position_a": p.POSITION_A,
            "position_b": p.POSITION_B,
            "position_c": p.POSITION_C,
            "risk_none_multiplier": p.RISK_NONE_MULTIPLIER,
            "max_industry_exposure": p.MAX_INDUSTRY_EXPOSURE,
            "risk_budget": p.RISK_BUDGET,
            "max_day_turnover": p.MAX_DAY_TURNOVER,
            "risk_aversion": p.RISK_AVERSION,
            "liq_w_section": f.LIQ_W_SECTION,
            "liq_w_timeseries": f.LIQ_W_TIMESERIES,
            "liq_w_marketcap": f.LIQ_W_MARKETCAP,
            "liq_min_discount": f.LIQ_MIN_DISCOUNT,
        }

    # ── P4 组合优化器配置 ──────────────────────────────────────────────

    @property
    def PORTFOLIO_OPTIMIZER(self) -> dict:
        o = self.app_config.portfolio_optimizer
        return {
            "method": o.METHOD,
            "risk_aversion": o.RISK_AVERSION,
            "turnover_penalty": o.TURNOVER_PENALTY,
            "max_weight": o.MAX_WEIGHT,
            "cov_lookback": o.COV_LOOKBACK,
            "shrinkage": o.SHRINKAGE,
            "industry_neutral": o.INDUSTRY_NEUTRAL,
            "industry_deviation": o.INDUSTRY_DEVIATION,
            "max_holdings": o.MAX_HOLDINGS,
            "target_cash_ratio": o.TARGET_CASH_RATIO,
            "solve_timeout": o.SOLVE_TIMEOUT,
            "verbose": o.VERBOSE,
        }

    # ── 工具方法 ────────────────────────────────────────────────────────

    def _ensure_directories(self) -> None:
        for d in (self.HOME_DIRECTORY, self.CACHE_DIRECTORY, self.TEMP_DATA_DIRECTORY, self.LOG_DIR):
            os.makedirs(d, exist_ok=True)
