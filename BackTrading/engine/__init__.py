from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from UtilsManager.ConfigParser import Config
from BackTrading.domain.models import CostModel, DEFAULT_TRANSFER_FEE_SEGMENTS


# ═══════════════════════════════════════════════════════════
# P2.6 子配置类 — 按功能域分组，EngineConfig 通过 @property
# 暴露子视图；直接字段访问 engine_cfg.xxx 保持向后兼容。
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CostConfig:
    """交易成本子配置 — 仅展示用，实际费用走 CostModel"""
    commission_rate: float
    stamp_tax_rate: float
    slippage: float
    transfer_fee_rate: float
    transfer_fee_segments: tuple[tuple[str, float], ...]
    min_commission_per_trade: float

    @property
    def buy_fee_simple(self) -> float:
        return self.commission_rate + self.transfer_fee_rate

    @property
    def sell_fee_simple(self) -> float:
        return self.commission_rate + self.transfer_fee_rate + self.stamp_tax_rate


@dataclass(frozen=True)
class ExecutionConfig:
    """成交撮合子配置"""
    execution_model: str
    simulate_limit_up_down: bool
    limit_seal_sell_ratio: float
    limit_seal_buy_ratio: float
    limit_tradable_up_ratio: float
    limit_tradable_down_ratio: float
    limit_intraday_ratio: float
    limit_seal_decay: float
    auction_fill_ratio: float
    limit_ratio_mode: str
    limit_calib_min_samples: int
    resume_gap_up: float
    resume_gap_down: float
    resume_auction_fill_ratio: float
    resume_impact_multiplier: float
    max_order_pct: float
    max_order_pct_high: float
    max_order_pct_low: float
    adv_amount_threshold_high: float
    adv_amount_threshold_low: float
    order_expiry_days: int
    strict_listing_days: bool


@dataclass(frozen=True)
class PositionConfig:
    """仓位控制子配置"""
    initial_cash: float
    max_position_pct: float
    portfolio_method: str
    atr_stop_mult: float
    kelly_fraction: float
    position_a: float
    boll_narrow_ratio: float
    cross_decay_days: int
    risk_none_multiplier: float
    max_holdings: int
    buy_threshold: int
    risk_per_trade: float
    top_k: int


@dataclass(frozen=True)
class RegimeConfig:
    """市场状态仓位调节子配置"""
    regime_ret20_full: float
    regime_ret20_half: float
    regime_vol_pct_max: float
    regime_full_multiplier: float
    regime_half_multiplier: float
    regime_min_multiplier: float


@dataclass(frozen=True)
class SuspensionConfig:
    """停牌盯市子配置"""
    susp_decay_start_days: int
    susp_daily_decay_rate: float
    susp_max_discount: float


@dataclass(frozen=True)
class OptimizerConfig:
    """组合优化器子配置"""
    optimizer_method: str
    optimizer_risk_aversion: float
    optimizer_turnover_penalty: float
    optimizer_max_weight: float
    optimizer_cov_lookback: int
    optimizer_shrinkage: bool
    optimizer_industry_neutral: bool
    optimizer_industry_deviation: float
    optimizer_max_holdings: int
    optimizer_target_cash: float
    optimizer_solve_timeout: float
    optimizer_verbose: bool


@dataclass(frozen=True)
class MarketFilterConfig:
    """市场过滤器子配置"""
    market_filter_enabled: bool
    market_filter_bull_ratio: float
    market_filter_min_stocks: int


@dataclass
class EngineConfig:
    """回测引擎配置 - 纯数据容器，无业务逻辑"""

    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    slippage: float = 0.001
    transfer_fee_rate: float = 0.00001  # 过户费 0.001% 双边
    # 过户费日期分段表（2022-04-29 前后费率不同，双边收取）；
    # fallback 构造 CostModel 时未显式传入则用 CostModel 默认值。
    transfer_fee_segments: tuple[tuple[str, float], ...] = DEFAULT_TRANSFER_FEE_SEGMENTS
    max_position_pct: float = 0.1
    portfolio_method: str = "score_weighted"
    point_in_time: bool = True
    atr_stop_mult: float = 2.5
    kelly_fraction: float = 0.25
    position_a: float = 0.3
    boll_narrow_ratio: float = 0.8
    cross_decay_days: int = 30
    risk_none_multiplier: float = 1.0
    max_holdings: int = 0  # 0=不限制
    buy_threshold: int = 15  # 买入评分阈值
    min_commission_per_trade: float = 5.0  # A股每笔最低佣金 5 元
    cost_model: Any = None  # CostModel | None — forward ref to avoid circular import
    # ── 市场过滤器（大盘风控开关） ──
    market_filter_enabled: bool = False
    market_filter_bull_ratio: float = 0.55  # >55%标的站上MA20视为牛市
    market_filter_min_stocks: int = 10  # 最少有效标的数
    
    # ── ATR 风险驱动仓位控制（A4） ──
    risk_per_trade: float = 0.02  # 单笔风险占总资金比例（默认2%）
    # ── 成交时点模型（0.1 执行时序合规） ──
    # next_open=信号次日开盘价成交（默认，符合A股T+1）
    # vwap=信号次日VWAP（成交额/成交量，后复权）成交。next_open/vwap 下单挂至次日开盘撮合，
    # 并与 simulate_limit_up_down 联动：次日一字涨停不可买入、一字跌停不可卖出。
    # close 模式已移除（固有前视偏差：信号依赖当日收盘数据计算，以同日收盘价成交=先知交易）
    execution_model: str = "next_open"  # next_open / vwap
    # ── 涨跌停撮合约束（simulate_limit_up_down=true 开启可成交量模型） ──
    simulate_limit_up_down: bool = True  # false=回退简化撮合（触板一律禁买/禁卖）
    limit_seal_ratio: float = 0.05  # [deprecated] 保留兼容
    # P1-2 修复：一字板封死方向流动性不对称——
    # 涨停板卖出相对容易（排队少，提供流动性），跌停板买入极难（恐慌情绪，逆势挂单）。
    limit_seal_sell_ratio: float = 0.05  # 一字涨停/跌停封板时卖出可成交量比例（提供流动性，相对容易）
    limit_seal_buy_ratio: float = 0.02   # 一字涨停/跌停封板时买入可成交量比例（逆势排队，极难）
    # P1-2 修复：涨跌停方向流动性不对称——涨停开盘买方排队深，买方成交难；
    # 跌停开盘恐慌抛压，卖方成交困难但炸板后流动性通常好于涨停封板。
    limit_tradable_up_ratio: float = 0.30   # 涨停开盘触板可成交量比例（买方保守）
    limit_tradable_down_ratio: float = 0.30  # 跌停开盘触板可成交量比例
    limit_tradable_ratio: float = 0.30  # [deprecated] 保留兼容，不再被核心路径消费
    limit_intraday_ratio: float = 0.10  # 盘中冲板（open<限价, high≥限价）可成交量比例
    limit_seal_decay: float = 0.5  # 连续板每板可成交量衰减系数
    # ── P0-6 ⑥ 开盘集合竞价成交率分档（封单量/可成交量代理） ──
    # 开盘价触板日（open≥涨停价/≤跌停价、未一字封死）集合竞价可成交量上限 =
    # 当日成交量 × min(触板档比例, auction_fill_ratio)。假设文档化：
    # 成交价=开盘价（集合竞价价），开盘后向限价收敛的盘中成交不单独建模。
    auction_fill_ratio: float = 0.12
    # ── 经验填充模型（limit_calibration.py）：用历史日线 V_t/V_prev 分位数
    # 替代固定比例常量（技术债：固定比例缺经验依据）。分钟/tick/盘口数据
    # 不在数据源内，经验分位是可行的统计替代：
    #   fixed            = 旧行为（固定比例常量）
    #   empirical_median = 经验中位数（中性口径，竞价触板日可成交量=前日量×p50）
    #   empirical_p10    = 经验 10% 分位（worst-case 保守口径，暴露流动性枯竭）
    # 单元格样本不足 limit_calib_min_samples → 回退 fixed 档；校准表全样本静态
    # 统计（静态参数选择，应用时不带前视——竞价可成交量 = 前日量 × 校准分位）。
    limit_ratio_mode: str = "fixed"
    limit_calib_min_samples: int = 20
    # ── 0.6 复牌跳空（停牌后复牌日开盘大幅跳空：补涨兑现 / 补跌标记） ──
    resume_gap_up: float = 0.05  # 复牌高开≥该比例（相对停牌前收盘）→ 开盘兑现卖出 + 当日禁买
    resume_gap_down: float = 0.05  # 复牌低开≤-该比例 → 日志标记（风控卖出照常）
    # P2-5 修复：复牌跳空买入集合竞价成交率限制
    # 复牌跳空高开日集合竞价买入可成交比例（复牌日买方抢筹激烈，实际成交受限）
    resume_auction_fill_ratio: float = 0.15
    # P1-6 修复：复牌跳空卖出流动性冲击放大系数
    # 复牌日成交量放大但实际流动性不足，冲击成本高于正常日
    resume_impact_multiplier: float = 2.0
    # ── 交易参数（P1-2：从 core.py 硬编码提升为配置驱动） ──
    max_order_pct: float = 0.30  # 默认单笔订单上限（占 ADV 股数比例）
    max_order_pct_high: float = 0.20  # 高流动性股上限 (ADV成交额>adv_amount_threshold_high)
    max_order_pct_low: float = 0.10   # 低流动性股上限 (ADV成交额<adv_amount_threshold_low)
    adv_amount_threshold_high: float = 1e8  # 高流动性阈值（元）
    adv_amount_threshold_low: float = 2e7   # 低流动性阈值（元）
    top_k: int = 20  # 每日最大候选买入数（集中资金，避免每只分到极小额度 < 1 手）
    # ── P3-3（审计）挂单过期天数（P0-6 ②：A股订单当日有效，信号次日未成交即撤销） ──
    # 从 core.py 硬编码提升为配置项，便于研究（如"连续重挂 N 日"实验）；默认 1 保持原语义。
    order_expiry_days: int = 1
    # ── P3-4（审计）上市日表严格模式：params._listing_days 缺失时 fail-fast ──
    # 默认 False 兼容旧行为（告警后停用新股豁免）；True 与数据质量门禁联动，
    # 杜绝"表缺失 → 静默停用豁免 → 结果口径改变"的低风险漂移。
    strict_listing_days: bool = False
    # ── 市场状态仓位调节（P0-6 ⑤：客观状态变量，替代评分中位数口径） ──
    # 指数 20 日收益（全市场后复权收盘 ret_20d 中位数代理）+ 市场波动率分位
    # （横截面日收益 std 在过去 250 交易日分位）。评分口径字段（regime_full_
    # threshold 等）已弃用，仅保留以兼容旧配置。
    regime_ret20_full: float = 0.02  # 指数20日收益 ≥ 此值 → 全仓倍率
    regime_ret20_half: float = -0.02  # 指数20日收益 ≥ 此值（且非高波）→ 半仓倍率
    regime_vol_pct_max: float = 0.8  # 波动率分位 > 此值 → 高波动，压制到最低倍率
    # ── 以下为旧评分口径（P0-6 ⑤ 弃用，仅兼容保留） ──
    regime_full_threshold: int = 30  # [已弃用] 中位数评分 ≥ 此值 → 全仓倍率
    regime_half_threshold: int = 15  # [已弃用] 中位数评分 ≥ 此值 → 半仓倍率
    regime_full_multiplier: float = 1.0  # 全仓倍率
    regime_half_multiplier: float = 0.5  # 半仓倍率
    regime_min_multiplier: float = 0.25  # 最低仓位倍率

    # ── P2-1 停牌盯市：停牌天数保守衰减折扣（无行业指数数据时的务实替代方案） ──
    # 停牌天数 > susp_decay_start_days 时，盯市价按 (1 - susp_daily_decay_rate)^(天数 - 起始天数) 折扣
    # 目的：防止长期停牌期间净值虚高（重大事项不确定性）；复牌日按实际成交价结算
    susp_decay_start_days: int = 10       # 停牌超过此天数开始衰减
    susp_daily_decay_rate: float = 0.002  # 每日衰减率 0.2%（年化 ~47% 保守折扣）
    susp_max_discount: float = 0.30       # 最大折扣上限 30%（防止极端停牌过度压低净值）

    # ── 组合优化器配置（P4 数学规划驱动） ──
    # 优化方法: mean_variance / min_variance / risk_parity / topk_equal(兼容旧版 Top-K 等权)
    optimizer_method: str = "topk_equal"
    # 风险厌恶系数 (γ, 越大越保守; 2.0 = 平衡, 5.0 = 保守)
    optimizer_risk_aversion: float = 2.0
    # 换手率惩罚系数 (λ_TC, 建议取年化交易成本 0.5~1.5 倍; 0.001 = 千1 轻度惩罚)
    optimizer_turnover_penalty: float = 0.001
    # 单票权重上限 (w_max, 默认 10%)
    optimizer_max_weight: float = 0.10
    # 协方差估计窗口 (交易日, P1-3 修复：A股市场结构变化快，60天不足，提升至120天)
    optimizer_cov_lookback: int = 120
    # 协方差收缩 (Ledoit-Wolf, True = 小样本稳健)
    optimizer_shrinkage: bool = True
    # 行业中性约束 (是否启用)
    optimizer_industry_neutral: bool = False
    # 行业暴露偏离上限 (绝对值)
    optimizer_industry_deviation: float = 0.05
    # 最大持仓数 (0 = 不限制)
    optimizer_max_holdings: int = 0
    # 目标现金比例
    optimizer_target_cash: float = 0.0
    # 求解超时 (秒)
    optimizer_solve_timeout: float = 5.0
    # P3-3 优化：优化器日志详细度控制（WFO 路径下大量重复日志降噪）
    # True = 输出所有 debug/info 日志；False = 仅输出 warning 以上级别
    optimizer_verbose: bool = False

    # ── FIX(P1) Subtask-9：强制持有期限上限 ──
    # 无止损保护（stop_col=NaN）时股票可能被永久持有，导致资金效率低。
    # 持仓超过 max_hold_days 交易日后重新评估：若当前 buy_score 低于 buy_threshold → 卖出。
    # 默认 60 个交易日（约一季度）；设为 0 表示关闭此限制。
    max_hold_days: int = 60

    # ── P2.6 子配置视图属性（不破坏 engine_cfg.xxx 直接访问兼容性） ──
    @property
    def cost(self) -> CostConfig:
        """交易成本子配置视图"""
        return CostConfig(
            commission_rate=self.commission_rate,
            stamp_tax_rate=self.stamp_tax_rate,
            slippage=self.slippage,
            transfer_fee_rate=self.transfer_fee_rate,
            transfer_fee_segments=self.transfer_fee_segments,
            min_commission_per_trade=self.min_commission_per_trade,
        )

    @property
    def execution(self) -> ExecutionConfig:
        """成交撮合子配置视图"""
        return ExecutionConfig(
            execution_model=self.execution_model,
            simulate_limit_up_down=self.simulate_limit_up_down,
            limit_seal_sell_ratio=self.limit_seal_sell_ratio,
            limit_seal_buy_ratio=self.limit_seal_buy_ratio,
            limit_tradable_up_ratio=self.limit_tradable_up_ratio,
            limit_tradable_down_ratio=self.limit_tradable_down_ratio,
            limit_intraday_ratio=self.limit_intraday_ratio,
            limit_seal_decay=self.limit_seal_decay,
            auction_fill_ratio=self.auction_fill_ratio,
            limit_ratio_mode=self.limit_ratio_mode,
            limit_calib_min_samples=self.limit_calib_min_samples,
            resume_gap_up=self.resume_gap_up,
            resume_gap_down=self.resume_gap_down,
            resume_auction_fill_ratio=self.resume_auction_fill_ratio,
            resume_impact_multiplier=self.resume_impact_multiplier,
            max_order_pct=self.max_order_pct,
            max_order_pct_high=self.max_order_pct_high,
            max_order_pct_low=self.max_order_pct_low,
            adv_amount_threshold_high=self.adv_amount_threshold_high,
            adv_amount_threshold_low=self.adv_amount_threshold_low,
            order_expiry_days=self.order_expiry_days,
            strict_listing_days=self.strict_listing_days,
        )

    @property
    def position(self) -> PositionConfig:
        """仓位控制子配置视图"""
        return PositionConfig(
            initial_cash=self.initial_cash,
            max_position_pct=self.max_position_pct,
            portfolio_method=self.portfolio_method,
            atr_stop_mult=self.atr_stop_mult,
            kelly_fraction=self.kelly_fraction,
            position_a=self.position_a,
            boll_narrow_ratio=self.boll_narrow_ratio,
            cross_decay_days=self.cross_decay_days,
            risk_none_multiplier=self.risk_none_multiplier,
            max_holdings=self.max_holdings,
            buy_threshold=self.buy_threshold,
            risk_per_trade=self.risk_per_trade,
            top_k=self.top_k,
        )

    @property
    def regime(self) -> RegimeConfig:
        """市场状态仓位调节子配置视图"""
        return RegimeConfig(
            regime_ret20_full=self.regime_ret20_full,
            regime_ret20_half=self.regime_ret20_half,
            regime_vol_pct_max=self.regime_vol_pct_max,
            regime_full_multiplier=self.regime_full_multiplier,
            regime_half_multiplier=self.regime_half_multiplier,
            regime_min_multiplier=self.regime_min_multiplier,
        )

    @property
    def suspension(self) -> SuspensionConfig:
        """停牌盯市子配置视图"""
        return SuspensionConfig(
            susp_decay_start_days=self.susp_decay_start_days,
            susp_daily_decay_rate=self.susp_daily_decay_rate,
            susp_max_discount=self.susp_max_discount,
        )

    @property
    def optimizer(self) -> OptimizerConfig:
        """组合优化器子配置视图"""
        return OptimizerConfig(
            optimizer_method=self.optimizer_method,
            optimizer_risk_aversion=self.optimizer_risk_aversion,
            optimizer_turnover_penalty=self.optimizer_turnover_penalty,
            optimizer_max_weight=self.optimizer_max_weight,
            optimizer_cov_lookback=self.optimizer_cov_lookback,
            optimizer_shrinkage=self.optimizer_shrinkage,
            optimizer_industry_neutral=self.optimizer_industry_neutral,
            optimizer_industry_deviation=self.optimizer_industry_deviation,
            optimizer_max_holdings=self.optimizer_max_holdings,
            optimizer_target_cash=self.optimizer_target_cash,
            optimizer_solve_timeout=self.optimizer_solve_timeout,
            optimizer_verbose=self.optimizer_verbose,
        )

    @property
    def market_filter(self) -> MarketFilterConfig:
        """市场过滤器子配置视图"""
        return MarketFilterConfig(
            market_filter_enabled=self.market_filter_enabled,
            market_filter_bull_ratio=self.market_filter_bull_ratio,
            market_filter_min_stocks=self.market_filter_min_stocks,
        )

    # ── P2.6 防御性验证 ──
    def validate(self) -> list[str]:
        """验证配置一致性，返回问题列表（空列表 = 通过）"""
        issues = []
        if self.initial_cash <= 0:
            issues.append("initial_cash 必须为正数")
        if not (0 < self.max_position_pct <= 1.0):
            issues.append("max_position_pct 应在 (0, 1] 范围")
        if self.buy_threshold < 0:
            issues.append("buy_threshold 不应为负数")
        if self.simulate_limit_up_down and self.execution_model == "close":
            issues.append("close 执行模式已移除（前视偏差），请改用 next_open/vwap")
        if self.optimizer_cov_lookback < 30:
            issues.append("optimizer_cov_lookback < 30 天，协方差估计极不可靠")
        if self.susp_daily_decay_rate < 0 or self.susp_daily_decay_rate > 0.05:
            issues.append("susp_daily_decay_rate 异常（建议范围 0~0.05）")
        if issues:
            logger.warning("[P2.6] EngineConfig 验证发现 {} 个问题: {}", len(issues), issues)
        return issues

    @property
    def buy_fee_rate(self) -> float:
        """买入费率（不含滑点）：佣金 + 过户费。

        ⚠️ 仅供展示/兼容：简单相加不含逐笔最低佣金（min_commission_per_trade）
        与历史费率分段表，不能用于费用核算。引擎实际费用一律经
        engine_cfg.cost_model（CostModel.buy_cost_breakdown，逐笔强制下限）。
        """
        return self.commission_rate + self.transfer_fee_rate

    @property
    def sell_fee_rate(self) -> float:
        """卖出费率（不含滑点）：佣金 + 过户费 + 印花税。

        ⚠️ 仅供展示/兼容：简单相加不含逐笔最低佣金（min_commission_per_trade）
        与历史费率分段表，不能用于费用核算。引擎实际费用一律经
        engine_cfg.cost_model（CostModel.sell_cost_breakdown，逐笔强制下限）。
        """
        return self.commission_rate + self.transfer_fee_rate + self.stamp_tax_rate


# ── 引擎公共 API re-export ──
# core.py 依赖上方已定义的 EngineConfig；此处放在类定义之后以避免循环 import：
#   __init__.py 先定义 EngineConfig → 再 import core → core 反向 import EngineConfig（已就绪）
from BackTrading.engine.core import (  # noqa: E402
    _MIN_SLIPPAGE_FLOOR,
    _run_single_backtest,
    run_full_backtest,
)

# ── P1.1 重构：拆分模块导出（向后兼容） ──
from BackTrading.engine.position_manager import (  # noqa: E402
    PositionState,
)
from BackTrading.engine.cost_calculator import (  # noqa: E402
    CostAccum,
    CostCalculator,
)
from BackTrading.engine.execution_engine import (  # noqa: E402
    AuctionFillConfig,
    ExecutionEngine,
)

__all__ = [
    "EngineConfig",
    "run_full_backtest",
    "_run_single_backtest",
    "_MIN_SLIPPAGE_FLOOR",
    # P1.1 拆分模块
    "PositionState",
    "CostAccum",
    "CostCalculator",
    "AuctionFillConfig",
    "ExecutionEngine",
]
