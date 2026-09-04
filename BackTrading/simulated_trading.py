from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm

from BackTrading.engine import EngineConfig, _run_single_backtest
from BackTrading.prepare import _build_params, merge_best_params_into_structured, prepare_backtest_data
from LogicAnalyzer.backtest_metrics import compute_risk_metrics


@dataclass
class SimTradeVerdict:
    """模拟交易验证结果与决策。"""

    sim_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    sim_sortino: float = 0.0
    oos_sortino: float = 0.0
    sharpe_degradation: float = 0.0     # 1 - sim/oos，负值表示 sim 优于 oos
    sortino_degradation: float = 0.0
    promote: bool = False
    reason: str = ""

    # 审计增强：统计与样本量元数据
    sim_sample_days: int = 0
    sim_trade_count: int = 0
    stat_p_value: float = 1.0

    # 兼容性：保留旧字段名（degradation = sharpe_degradation）
    @property
    def degradation(self) -> float:
        return self.sharpe_degradation


# 衰减容忍度：30%（审计要求，原 50% 过松）
_DECAY_THRESHOLD = 0.30
# warmup 天数（sim 段前预留交易日 buffer，让引擎建立 ADV/仓位后再进入 sim 段，
# 避免 20 天 sim 窗口下 ADV 永不满载导致流动性约束失效）
_SIM_WARMUP_DAYS = 30
# ── 审计增强：硬性统计门槛 ──
_MIN_SIM_DAYS = 20          # 模拟期最少交易日（低于此值统计噪声过大）
_MIN_SIM_TRADES = 3         # 模拟期最少交易次数（少于3笔无法评估滑点/冲击成本）
_MIN_OOS_SHARPE = 0.20      # OOS Sharpe 最低门槛（低于此值样本外信号极弱，拒绝自引用放行）


def _cost_model_from_config(config: Any) -> Any:
    """从 Config 构建 CostModel（含流动性分档冲击成本）。

    审计（成本外部化）：fail-fast——构建失败直接抛出，禁止静默回落"统一成本"。
    统一成本口径缺失逐笔最低佣金/历史费率分段表/流动性分档，验证与主回测费用
    口径分裂。无 Config（研究模式）时显式返回默认 CostModel()（仍含逐笔最低
    佣金与日期分段表，仅费率取默认值）。
    """
    from BackTrading.domain.models import CostModel

    if config is None:
        return CostModel()
    return CostModel.from_backtest_config(
        config.app_config.backtest,
        trading_cost=config.app_config.trading_cost,
    )


_ANN_FACTOR = 244  # A股实际年化交易日数均值（非美股252）


def _hac_sharpe_se_from_returns(
    returns: np.ndarray,
    risk_free_rate: float = 0.03,
    lag: int | None = None,
) -> tuple[float, int, float]:
    """从日收益序列估计年化 Sharpe 的 Newey-West HAC 标准误（delta 方法）。

    替代 Lo(2002) iid 近似 sqrt((SR²+0.5)/n)：该近似忽略自相关与
    波动聚集（ARCH），且未按年化口径换算（低估 SE 约 √252 倍，t 统计量被
    同比例放大，显著性结论失真）。

    矩条件 z_t = [r_t - μ, (r_t - μ)² - σ²]，梯度 g = [1/σ, -μ/(2σ³)]，
    se(SR_d) = sqrt(gᵀ V_HAC g)，再按 √252 年化。V_HAC 为 Newey-West
    Bartlett 核 2×2 协方差矩阵（计入自相关与波动聚集），滞后阶数默认
    floor(4(n/100)^(2/9))，与 overfitting.compute_dm_test 口径一致。

    Args:
        returns: 日收益序列（可含 NaN，自动剔除）。
        risk_free_rate: 年化无风险利率（与 compute_risk_metrics 默认一致）。
        lag: Newey-West 滞后阶数，None 时按样本量自动选取。

    Returns:
        (se_annualized, n_returns, sharpe_annualized)；
        样本不足或方差退化（如测试桩常数权益曲线）时返回 (nan, n, 0.0)，
        由调用方回退 Lo(2002) iid 年化近似。
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    if n < 10:
        return math.nan, n, 0.0

    excess = returns - risk_free_rate / _ANN_FACTOR
    mu = float(excess.mean())
    sigma2 = float(np.mean((excess - mu) ** 2))
    if sigma2 <= 1e-12:
        return math.nan, n, 0.0
    sigma = math.sqrt(sigma2)
    sharpe_ann = mu / sigma * math.sqrt(_ANN_FACTOR)

    if lag is None:
        lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(lag, n - 2))

    dev = excess - mu
    z = np.column_stack([dev, dev**2 - sigma2])
    v = z.T @ z / n
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        gj = z[:-j].T @ z[j:] / n
        v += w * (gj + gj.T)

    grad = np.array([1.0 / sigma, -mu / (2.0 * sigma**3)])
    se_daily = math.sqrt(max(float(grad @ v @ grad), 0.0) / n)
    return se_daily * math.sqrt(_ANN_FACTOR), n, sharpe_ann


def _hac_sharpe_se(
    equity_curve: list[dict[str, Any]],
    risk_free_rate: float = 0.03,
    lag: int | None = None,
) -> tuple[float, int, float]:
    """从权益曲线估计年化 Sharpe 的 Newey-West HAC 标准误（delta 方法）。

    提取 portfolio_value 序列后委托 _hac_sharpe_se_from_returns。
    """
    if not equity_curve:
        return math.nan, 0, 0.0
    vals = np.asarray(
        [float(r["portfolio_value"]) for r in equity_curve if r.get("portfolio_value") is not None],
        dtype=float,
    )
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3 or vals[0] <= 0:
        return math.nan, 0, 0.0
    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    returns = returns[np.isfinite(returns)]
    return _hac_sharpe_se_from_returns(returns, risk_free_rate=risk_free_rate, lag=lag)


def validate_params(
    kline_df: pd.DataFrame,
    best_params: dict[str, float],
    oos_sharpe: float,
    sim_days: int = 20,
    config: Any | None = None,
    engine_cfg: EngineConfig | None = None,
    oos_sortino: float = 0.0,  # 审计新增：样本外 Sortino
    validation_dates: set[str] | None = None,  # P0-10 ④：独立验证集（与选参区间无交集）
    oos_sample_days: int = 60,  # 审计增强：OOS 样本量（用于统计检验 SE 估算）
    oos_returns: np.ndarray | None = None,  # 审计增强：OOS 日收益序列（HAC 修正 SE）
) -> SimTradeVerdict:
    """用独立验证集验证 best_params 的稳定性。

    P0-10 ④：原实现取"最近 sim_days 个交易日"做模拟验证，该区间与 WFO 的
    holdout/OOS 选参区间重叠 → 自引用验证（sim 段正是选参评估段）。
    现优先使用调用方传入的独立验证集（如末段 holdout，WFO 全程禁触）；
    未提供时才回退最近 N 日并告警（自引用回退，与主流程口径一致）。

    Args:
        kline_df: 全量 K 线数据（含信号列或原始数据均可）。
        best_params: WFO 选出的最佳参数（flat dict，至少含 atr_stop_mult）。
            P1-18：调用方需确保 best_params 包含以下夹具键，否则 ST 5% 涨跌幅
            与次新股豁免逻辑在模拟验证路径将被停用：
            - best_params["_st_history"]: ST/退市逐日状态
            - best_params["_listing_days"]: 上市日期映射
            - best_params["_exclude_st"]: 是否启用 ST 剔除
            （runner.py 已自动注入；独立调用 verify_strategy 时需手动传入）
        oos_sharpe: WFO 在样本外窗口上的 Sharpe。
        sim_days: 验证集交易日数。
        config: Config 实例（可选，用于构建结构化 params）。
        engine_cfg: EngineConfig 实例（可选，构建最终回测引擎）。
        oos_sortino: WFO 在样本外窗口上的 Sortino（审计新增，默认 0=跳过 Sortino 校验）。
        validation_dates: 独立验证集日期集合（与选参区间无交集）；
            None 时回退"最近 sim_days 日"（自引用，告警）。
        oos_sample_days: OOS 样本交易日数（用于统计检验 SE 估算，默认 60）。
        oos_returns: 样本外日收益序列（WFO rank-1 组合 OOS 段拼接）；
            提供时 OOS 侧 Sharpe SE 使用 Newey-West HAC 修正（与 sim 侧同口径），
            未提供时回退 Lo(2002) iid 年化近似并告警。

    Returns:
        SimTradeVerdict 包含决策、统计指标与拒绝原因。
    """
    if best_params is None or oos_sharpe is None:
        return SimTradeVerdict(promote=False, reason="WFO 结果为空，跳过模拟验证")

    # 取验证段交易日（独立验证集优先；否则最近 sim_days 日自引用回退）
    dates = sorted(kline_df["trade_date"].unique())
    if validation_dates:
        _v_dates = sorted({str(d) for d in dates} & {str(d) for d in validation_dates})
        if len(_v_dates) < max(10, sim_days // 2):
            # 审计：样本不足一律拒绝自动放行（promote=False），需人工复核。
            # 原实现 promote=True 直接放行——独立验证集不足时策略被静默跳过
            # 验证直接上线，合规与风控漏洞；调用方（runner）遇 promote=False
            # 不写入校准结果，即"禁用自动上线"。
            return SimTradeVerdict(
                promote=False,
                reason=f"独立验证集交易日仅 {len(_v_dates)} 天 < {max(10, sim_days // 2)}，"
                       f"样本不足需人工确认，拒绝自动放行",
            )
        sim_dates_sorted = _v_dates[-sim_days:]
    else:
        logger.warning(
            "[模拟验证] 未提供独立验证集（holdout 未激活），"
            "回退最近 N 个交易日做模拟验证（自引用，结果仅供参考）"
        )
        if len(dates) < sim_days + 20:
            return SimTradeVerdict(
                promote=False,
                reason=f"数据不足（{len(dates)} 个交易日），无法做模拟验证，"
                       f"样本不足需人工确认，拒绝自动放行",
            )
        sim_dates_sorted = dates[-sim_days:]

    # 准备信号 + 止损价
    # 统一使用 _build_params 构建结构化 params，消除 config is None 时
    # 直接把扁平 best_params 传给 prepare_backtest_data 导致的参数不一致：
    # 旧代码的 is_flat 转换白名单不完整（遗漏 atr_stop_mult, expected_return_lookback,
    # conclusion_bullish/oscillate 等），导致 prepare/engine 参数口径不一致。
    from UtilsManager.ConfigParser import Config as _Cfg

    cfg = config if config is not None and isinstance(config, _Cfg) else _Cfg()
    structured = _build_params(cfg)

    # 将 best_params（WFO 产出的扁平 dict）合并进结构化 params 的对应分区
    # 统一使用 merge_best_params_into_structured，与 runner.py/prepare.py 保持同一路由逻辑
    merge_best_params_into_structured(best_params, structured)

    prepared = prepare_backtest_data(
        kline_df, params=structured, compute_exit_strategy=False, vectorized=True,
    )

    # ── P1-1：warmup buffer 避免 ADV 冷启动 ──
    _prep = prepared
    if pd.api.types.is_datetime64_any_dtype(_prep["trade_date"]):
        _prep = _prep.copy()
        _prep["trade_date"] = _prep["trade_date"].dt.strftime("%Y-%m-%d")
    _date_range = _prep["trade_date"].astype(str)
    _unique_dates = sorted(_date_range.unique())
    # P0-10 ④：验证段按 sim_dates_sorted 定位（独立验证集可能不在数据末尾），
    # 回退模式下即末尾 sim_days 日
    _sim_n = min(len(sim_dates_sorted), len(_unique_dates))
    _sim_dates_str = {d for d in sim_dates_sorted if d in _unique_dates}
    sim_dates_str = _sim_dates_str
    if sim_dates_sorted and sim_dates_sorted[0] in _unique_dates:
        _sim_start_pos = _unique_dates.index(sim_dates_sorted[0])
    else:
        _sim_start_pos = len(_unique_dates) - _sim_n
    # 扩展段 [warmup_start, sim_end]
    _warmup_pos = max(0, _sim_start_pos - _SIM_WARMUP_DAYS)
    _warmup_start = _unique_dates[_warmup_pos]
    _sim_end = sim_dates_sorted[-1] if sim_dates_sorted else _unique_dates[-1]
    mask_ext = (_date_range >= _warmup_start) & (_date_range <= _sim_end)
    ext_data = _prep[mask_ext].copy()
    if ext_data.empty:
        return SimTradeVerdict(
            promote=False,
            reason="模拟期数据为空，样本不足需人工确认，拒绝自动放行",
        )

    # 按 best_params 的 atr_stop_mult 计算止损价
    stop_mult = best_params.get("atr_stop_mult")
    if stop_mult is not None and "ATR" in ext_data.columns:
        # P0-1：止损价与引擎比较基准统一到后复权空间（指标 ATR 亦为后复权）
        _stop_close = ext_data["close_normal"] if "close_normal" in ext_data.columns else ext_data["close"]
        ext_data["止损价"] = _stop_close - ext_data["ATR"] * stop_mult
    elif "止损价" not in ext_data.columns:
        ext_data["止损价"] = 0.0

    if engine_cfg is None:
        # ── 口径一致性（审计）：统一"选参→验证"引擎参数口径 ──
        # 旧实现仅注入 atr_stop_mult，buy_threshold/max_holdings 落回 EngineConfig
        # 默认（15/0），与主流程 runner.py:491-493（best_params → engine_cfg，回退
        # 校准覆写 BUY_THRESHOLD=17/MAX_HOLDINGS=11）口径分裂：WFO 寻优出的
        # buy_threshold/max_holdings 在模拟验证回测中不生效。现与主流程对齐。
        _sc_cfg = config.app_config.scoring_params if config is not None else None
        _bt_cfg = config.app_config.backtest if config is not None else None
        engine_cfg = EngineConfig(
            atr_stop_mult=float(best_params.get(
                "atr_stop_mult",
                getattr(_sc_cfg, "ATR_STOP_MULT", 1.5) if _sc_cfg is not None else 1.5,
            )),
            buy_threshold=int(best_params.get(
                "buy_threshold",
                getattr(_bt_cfg, "BUY_THRESHOLD", 17) if _bt_cfg is not None else 17,
            )),
            max_holdings=int(best_params.get(
                "max_holdings",
                getattr(_bt_cfg, "MAX_HOLDINGS", 11) if _bt_cfg is not None else 11,
            )),
            cost_model=_cost_model_from_config(config),
        )

    # ── 口径一致性自检：打印实际生效的关键参数；best_params 与 engine_cfg 冲突时告警 ──
    _eff_params = {
        "atr_stop_mult": engine_cfg.atr_stop_mult,
        "buy_threshold": engine_cfg.buy_threshold,
        "max_holdings": engine_cfg.max_holdings,
        "cross_decay_days": structured["scoring"].get("cross_decay_days"),
        "boll_narrow_ratio": structured["regime"].get("boll_narrow_ratio"),
        "conclusion_full_bull": structured["thresholds"].get("fully_bull"),
    }
    logger.info(f"[模拟验证] 生效关键参数: {_eff_params}")
    for _k, _v in (
        ("atr_stop_mult", engine_cfg.atr_stop_mult),
        ("buy_threshold", engine_cfg.buy_threshold),
        ("max_holdings", engine_cfg.max_holdings),
    ):
        if _k in best_params and best_params[_k] != _v:
            logger.warning(
                f"[模拟验证] 口径不一致: best_params[{_k}]={best_params[_k]} "
                f"≠ engine_cfg.{_k}={_v}，模拟验证与选参口径分裂"
            )

    tl: list[dict[str, Any]] = []
    ec: list[dict[str, Any]] = []
    _run_single_backtest(ext_data, best_params, engine_cfg, tl, ec)

    # 仅保留 sim 段权益曲线（warmup 段不入指标）
    ec_sim = [row for row in ec if str(row.get("time", ""))[:10] in sim_dates_str]
    if not ec_sim:
        return SimTradeVerdict(
            promote=False,
            reason="模拟期权益数据为空，样本不足需人工确认，拒绝自动放行",
        )

    risk = compute_risk_metrics(ec_sim) or {}
    sim_sharpe = risk.get("sharpe_ratio", 0.0) or 0.0
    sim_sortino = risk.get("sortino_ratio")
    if sim_sortino is None or not math.isfinite(sim_sortino):
        sim_sortino = 0.0

    # ── 统计元数据采集 ──
    n_sim = len(ec_sim)
    sim_trade_count = len([
        t for t in tl
        if str(t.get("trade_date", t.get("time", "")))[:10] in sim_dates_str
    ])

    # ── 硬性门槛校验（审计增强：拒绝统计噪声与弱信号自引用） ──
    min_sample_ok = n_sim >= _MIN_SIM_DAYS
    min_trades_ok = sim_trade_count >= _MIN_SIM_TRADES
    oos_robust = oos_sharpe > _MIN_OOS_SHARPE

    # OOS_Sharp ≤ 0（负/零）单独处理：样本外收益为负/零 = 策略信号失效，
    # 拒绝自动上线 + 触发人工审核。不落入下方通用"样本不足"拒绝（避免
    # 负值与"接近零但为正"同化为单一常数语义），reason 上报实际数值供审计。
    if oos_sharpe <= 0:
        return SimTradeVerdict(
            sim_sharpe=sim_sharpe, oos_sharpe=oos_sharpe,
            sim_sample_days=n_sim, sim_trade_count=sim_trade_count,
            promote=False,
            reason=(
                f"拒绝: OOS_Sharp={oos_sharpe:.4f}≤0（样本外收益为负/零，策略信号失效），"
                f"拒绝自动上线，需人工审核"
            ),
        )

    if not (min_sample_ok and min_trades_ok and oos_robust):
        return SimTradeVerdict(
            sim_sharpe=sim_sharpe, oos_sharpe=oos_sharpe,
            sim_sample_days=n_sim, sim_trade_count=sim_trade_count,
            promote=False,
            reason=(
                f"拒绝: 样本量({n_sim}d<{_MIN_SIM_DAYS})或交易数({sim_trade_count}<{_MIN_SIM_TRADES})不足，"
                f"或OOS_Sharp({oos_sharpe:.4f})过弱(<{_MIN_OOS_SHARPE})，统计效力不足"
            ),
        )

    # ── Sharpe/Sortino 衰减校验 ──
    # 旧实现：oos_sharpe/oos_sortino ≤ 0.01 时用单一常数 1.0 替代（标记 100% 衰减），
    # 无法区分负值/接近零/未提供的语义：
    #   - OOS_Sharp ≤ 0 已在硬门槛单列拒绝（见上），此处执行时必 > 0（硬门槛
    #     _MIN_OOS_SHARPE=0.20 保证），sharpe_deg 正常按比例计算；
    #   - oos_sortino ≤ 0 语义为"未提供 Sortino"（docstring：默认 0=跳过 Sortino
    #     校验）→ 跳过，不参与 degrade_ok 判定。修复旧实现把"未提供"同化为
    #     100% 衰减导致 runner 未传 oos_sortino 时模拟验证恒拒绝的 bug。
    sharpe_deg = 1.0 - (sim_sharpe / oos_sharpe)
    sortino_deg = None
    if oos_sortino > 0:
        sortino_deg = 1.0 - (sim_sortino / oos_sortino)

    # ── 统计显著性检验（Newey-West HAC 校正 SE，双侧 t 检验） ──
    # 旧实现缺陷：① se = sqrt((SR²+0.5)/n) 为 iid 近似且未按年化口径换算，
    # 忽略自相关与波动聚集（SE 低估约 √252 倍）；② norm.cdf(t) 为单侧 p-value。
    # 现改为：sim 侧用 delta 方法 + Newey-West HAC 2×2 协方差矩阵（自动计入
    # 自相关/ARCH，滞后阶数 floor(4(n/100)^(2/9))，与 compute_dm_test 同口径）；
    # OOS 侧优先使用调用方传入的 OOS 收益序列（oos_returns）做同口径 HAC 修正，
    # 仅标量输入时回退 Lo(2002) iid 近似（年化口径 se = sqrt((252 + SR²/2)/n)，
    # 为可用信息下的下限估计，并告警）；
    # p 值统一使用双侧检验 p = 2·Φ(-|t|)。
    se_sim, _n_hac, _sr_hac = _hac_sharpe_se(ec_sim)
    if not math.isfinite(se_sim) or se_sim <= 0:
        se_sim = math.sqrt((_ANN_FACTOR + sim_sharpe**2 / 2.0) / max(n_sim, 1))
    # OOS 侧：优先用调用方提供的 OOS 收益序列做 HAC 修正；仅标量输入时
    # 回退 Lo(2002) iid 年化近似（下限估计）并明确告警，不再静默使用近似。
    if oos_returns is not None:
        _se_oos_hac, _n_oos_hac, _sr_oos_hac = _hac_sharpe_se_from_returns(oos_returns)
        if math.isfinite(_se_oos_hac) and _se_oos_hac > 0:
            se_oos = _se_oos_hac
            logger.info(
                f"OOS 侧 Sharpe SE 使用 Newey-West HAC 修正: "
                f"SE={se_oos:.3f} (n={_n_oos_hac}, SR={_sr_oos_hac:.2f}, iid 近似 "
                f"SE={math.sqrt((_ANN_FACTOR + _sr_oos_hac**2 / 2.0) / max(_n_oos_hac, 1)):.3f})"
            )
        else:
            se_oos = math.sqrt((_ANN_FACTOR + oos_sharpe**2 / 2.0) / max(oos_sample_days, 1))
            logger.warning(
                f"OOS 收益序列样本不足，回退 iid 近似 SE={se_oos:.3f}（n={oos_sample_days}）"
            )
    else:
        se_oos = math.sqrt((_ANN_FACTOR + oos_sharpe**2 / 2.0) / max(oos_sample_days, 1))
        logger.warning(
            f"未提供 OOS 收益序列，OOS 侧 SE 用 Lo(2002) iid 近似 {se_oos:.3f}；"
            f"提供 oos_returns 可启用 Newey-West HAC 修正"
        )
    se_diff = math.sqrt(se_sim**2 + se_oos**2)
    t_stat = (sim_sharpe - oos_sharpe) / se_diff if se_diff > 1e-9 else -99.0
    p_value = float(2.0 * norm.sf(abs(t_stat)))

    # ── 综合判定 ──
    degrade_ok = sharpe_deg < _DECAY_THRESHOLD and (
        sortino_deg is None or sortino_deg < _DECAY_THRESHOLD
    )
    stat_ok = p_value > 0.05  # 5% 显著性水平（双侧）拒绝显著差异
    positive_ok = sim_sharpe > 0.1

    promote = positive_ok and degrade_ok and stat_ok

    if promote:
        reason = (
            f"通过 | sim_SR={sim_sharpe:.2f}(n={n_sim}, trades={sim_trade_count}) / oos_SR={oos_sharpe:.2f} "
            f"| 衰减 {sharpe_deg:.0%} | p={p_value:.3f}"
        )
    else:
        fail_parts = []
        if not positive_ok: fail_parts.append("sim_SR≤0.1")
        if not degrade_ok:
            _worst_deg = sharpe_deg if sortino_deg is None else max(sharpe_deg, sortino_deg)
            fail_parts.append(f"衰减{_worst_deg:.0%}≥{_DECAY_THRESHOLD:.0%}")
        if not stat_ok: fail_parts.append(f"双侧p={p_value:.3f}≤0.05显著差异")
        reason = (
            f"拒绝 | sim_SR={sim_sharpe:.2f}(n={n_sim}, trades={sim_trade_count}) / oos_SR={oos_sharpe:.2f} "
            f"| {'; '.join(fail_parts)}"
        )

    logger.info(f"  [模拟验证] {reason}")
    return SimTradeVerdict(
        sim_sharpe=sim_sharpe,
        oos_sharpe=oos_sharpe,
        sim_sortino=sim_sortino,
        oos_sortino=oos_sortino,
        sharpe_degradation=sharpe_deg,
        sortino_degradation=sortino_deg,
        promote=promote,
        reason=reason,
        sim_sample_days=n_sim,
        sim_trade_count=sim_trade_count,
        stat_p_value=p_value,
    )
