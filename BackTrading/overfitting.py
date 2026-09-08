from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


def _ann_factor() -> int:
    return 244  # A股实际年化交易日数均值（非美股252）


def _compute_sharpe_from_equity(
    equity_curve: list[dict[str, Any]] | pd.DataFrame,
    risk_free_rate: float = 0.03,
) -> float:
    """统一 Sharpe 计算入口（超额收益口径）。

    所有下游函数（DSR / OOS 衰减 / Sortino）均应调用此函数，避免
    各模块各自实现导致口径漂移。

    Args:
        equity_curve: 权益曲线（list of dict 或 DataFrame，含 portfolio_value 列）。
        risk_free_rate: 年化无风险利率（默认 3%）。

    Returns:
        年化超额收益 Sharpe；数据不足时返回 0.0。
    """
    if isinstance(equity_curve, pd.DataFrame):
        vals = equity_curve["portfolio_value"].values.astype(float)
    else:
        vals = np.array(
            [e.get("portfolio_value", 0) for e in equity_curve], dtype=float
        )

    finite_mask = np.isfinite(vals)
    if finite_mask.sum() < 2:
        return 0.0
    vals = vals[finite_mask]
    if vals[0] <= 0:
        return 0.0

    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0

    ann_factor = _ann_factor()
    mu = returns.mean() * ann_factor
    excess_mu = mu - risk_free_rate
    sigma = returns.std(ddof=1) * math.sqrt(ann_factor)
    return float(excess_mu / sigma) if sigma > 0 else 0.0


# OOS 最少交易日数：低于此样本量时 Sharpe 估计噪声过大，衰减比不可判定
_MIN_OOS_DAYS = 10

# ── 三态分类：区分"策略无效" vs "真实过拟合" vs "策略偏弱" ──
# 背景：原系统将 IS_Sharpe ≤ 0 报为"过拟合"，严重误导排障方向。
# 三态逻辑：
#   INVALID   → IS_Sharpe ≤ 0，策略在样本内都无超额收益，不是过拟合
#   OVERFITTED → IS_Sharpe > 0 但衰减 > 50%（或 OOS Sharpe ≤ 0），真实过拟合
#   WEAK      → IS_Sharpe > 0 且衰减 > 30% 但 ≤ 50%，策略信号偏弱
class OverfitType(enum.Enum):
    VALID = "VALID"           # 通过校验
    INVALID = "INVALID"       # 策略无效：IS Sharpe ≤ 0，非过拟合
    OVERFITTED = "OVERFITTED"  # 真实过拟合：IS 正但 OOS 衰减 > 50%
    WEAK = "WEAK"             # 策略偏弱：IS 正但衰减 30%-50%


def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    target_sr: float = 0.0,
) -> float:
    if n_obs <= 1:
        return 0.5

    sd = sharpe / math.sqrt(_ann_factor())
    td = target_sr / math.sqrt(_ann_factor())

    num = (sd - td) * math.sqrt(n_obs - 1)
    den = math.sqrt(1.0 - skew * sd + ((kurt - 1.0) / 4.0) * sd * sd)
    if den <= 1e-12:
        return 0.5

    return float(stats.norm.cdf(num / den))


def compute_dm_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    lag: int | None = None,
) -> tuple[float, float]:
    """Diebold-Mariano 检验（Newey-West HAC 方差）。

    损失函数取负收益（loss = -return，等价于 d_t = a_t - b_t），
    H0: 两组日收益均值相等（E[d] = 0）。
    用于判定"寻优最佳参数在 OOS 上是否显著优于基准（配置中位数参数）"，
    防止调参只是噪声拟合：若 p ≥ 0.05，最佳参数的相对优势不显著，
    最终参数提取应退回稳健中位数主路径，而不是采信 Sharpe 尖峰。

    Args:
        returns_a: 策略 A 日收益序列。
        returns_b: 基准 B 日收益序列。
        lag: Newey-West 滞后阶数，None 时取 floor(4*(n/100)^(2/9))。

    Returns:
        (dm_stat, p_value)；dm_stat > 0 表示 A 平均收益高于 B（A 更优）。
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    n = min(len(a), len(b))
    if n < 10:
        return 0.0, 1.0

    d = a[:n] - b[:n]
    mean_d = float(d.mean())
    d_centered = d - mean_d

    if lag is None:
        lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(lag, n - 2))

    gamma0 = float(np.mean(d_centered ** 2))
    var_hac = gamma0
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        gamma_j = float(np.mean(d_centered[:-j] * d_centered[j:]))
        var_hac += 2.0 * w * gamma_j

    var_hac = max(var_hac, 1e-12)
    se = math.sqrt(var_hac / n)
    dm_stat = mean_d / se
    p_value = float(2.0 * stats.norm.cdf(-abs(dm_stat)))
    return dm_stat, p_value


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    num_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    if num_trials <= 1:
        return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurt, 0.0)

    sigma_sr = 1.0 / math.sqrt(n_obs - 1) if n_obs > 1 else 1.0
    # sigma_sr = 1/sqrt(T) (Bailey & López de Prado 2014),
    # T ≡ 单次回测的独立收益观测数（n_obs），非试验次数
    gamma_euler = 0.5772156649

    inv_n = 1.0 / num_trials
    try:
        z1 = float(stats.norm.ppf(1.0 - inv_n))
        z2 = float(stats.norm.ppf(1.0 - inv_n / math.e))
    except Exception:
        return 0.5

    e_max_sr = sigma_sr * ((1.0 - gamma_euler) * z1 + gamma_euler * z2)
    return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurt, e_max_sr)


def compute_pbo(
    window_results: list[dict[Any, Any]],
    top_m: int = 5,
) -> float:
    if not window_results:
        return 0.5

    violations = 0
    total_windows = 0

    for w in window_results:
        ocs = w.get("oos_combos", [])
        if len(ocs) < 2:
            continue

        oos_sharpes = [
            c["oos_sharpe"]
            for c in ocs
            if c.get("oos_sharpe") is not None and not (isinstance(c["oos_sharpe"], float) and math.isnan(c["oos_sharpe"]))
        ]
        if len(oos_sharpes) < 2:
            continue

        median_oos = float(np.median(oos_sharpes))
        rank1 = next((c for c in ocs if c.get("is_rank") == 1), None)
        if rank1 is not None:
            sr = rank1.get("oos_sharpe")
            if sr is not None and not (isinstance(sr, float) and math.isnan(sr)):
                total_windows += 1
                if sr < median_oos:
                    violations += 1

    if total_windows == 0:
        return 0.5
    return violations / total_windows


def compute_dsr_from_equity_curve(
    equity_curve: list[dict[Any, Any]],
    num_trials: int,
    risk_free_rate: float = 0.03,
) -> float:
    """计算 DSR，Sharpe 口径通过 _compute_sharpe_from_equity 保证一致（超额收益）。

    Args:
        equity_curve: 权益曲线。
        num_trials: 寻优评估次数。
        risk_free_rate: 年化无风险利率（默认 3%）。
    """
    if len(equity_curve) < 2:
        return 0.5

    sharpe = _compute_sharpe_from_equity(equity_curve, risk_free_rate)

    vals = pd.Series([e.get("portfolio_value", 0) for e in equity_curve]).values.astype(float)
    if len(vals) < 2:
        return 0.5

    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    n = len(returns)
    if n < 2:
        return 0.5

    skew = float(pd.Series(returns).skew())  # type: ignore[arg-type]
    kurt = float(pd.Series(returns).kurtosis()) + 3.0  # type: ignore[arg-type]

    return deflated_sharpe_ratio(sharpe, n, num_trials, skew, kurt)


# ──────────────────────────────────────────────────────────────
# Out-of-Sample 衰减校验（López de Prado 2018 准则 #1）
# ──────────────────────────────────────────────────────────────

@dataclass
class OOSDecayReport:
    """样本外衰减校验报告（三态分类）。

    业务定义：区分三种失效场景，避免将"策略无效"误报为"过拟合"。
      - INVALID:   IS_Sharpe ≤ 0，策略在样本内无超额收益，不是过拟合
      - OVERFITTED: IS_Sharpe > 0 但衰减 > 50%（或 OOS Sharpe ≤ 0），真实过拟合
      - WEAK:      IS_Sharpe > 0 但衰减 30%-50%，策略信号偏弱
    """

    # ── 输入 ──
    is_sharpe: float = 0.0
    oos_sharpe: float = 0.0
    is_sortino: float = 0.0
    oos_sortino: float = 0.0

    # ── 计算 ──
    sharpe_decay: float = 0.0        # 1 - oos / is，正值表示衰减
    sortino_decay: float = 0.0
    sharpe_ratio: float = 1.0        # oos / is，>0.7 表示未超阈

    # ── 判定 ──
    overfit_type: OverfitType = OverfitType.VALID
    passed: bool = False
    reason: str = ""
    details: list[str] = field(default_factory=list)
    is_sample_days: int = 0
    oos_sample_days: int = 0

    # ── 序列化 ──
    def to_dict(self) -> dict[str, Any]:
        return {
            "is_sharpe": round(self.is_sharpe, 4),
            "oos_sharpe": round(self.oos_sharpe, 4),
            "sharpe_decay_pct": f"{self.sharpe_decay:.1%}",
            "is_sortino": round(self.is_sortino, 4),
            "oos_sortino": round(self.oos_sortino, 4),
            "sortino_decay_pct": f"{self.sortino_decay:.1%}",
            "overfit_type": self.overfit_type.value,
            "passed": "PASS" if self.passed else "FAIL",
            "reason": self.reason,
            "is_days": self.is_sample_days,
            "oos_days": self.oos_sample_days,
        }

    def log(self) -> None:
        ftype = self.overfit_type.value
        status = "PASS" if self.passed else f"FAIL({ftype})"
        logger.info(
            f"[OOS衰减校验] {status} | IS_Sharpe={self.is_sharpe:.2f} → "
            f"OOS_Sharpe={self.oos_sharpe:.2f} "
            f"(衰减 {self.sharpe_decay:.1%}) | "
            f"IS_Sortino={self.is_sortino:.2f} → OOS_Sortino={self.oos_sortino:.2f} "
            f"(衰减 {self.sortino_decay:.1%}) | {self.reason}"
        )


def _compute_risk_from_curve(
    equity_curve: list[dict[str, Any]] | pd.DataFrame,
    risk_free_rate: float = 0.03,
) -> tuple[float, float]:
    """从净值曲线计算年化 Sharpe 和 Sortino。

    Sharpe 通过 _compute_sharpe_from_equity 统一入口计算，
    Sortino 在此函数计算（仅 OOS 衰减校验需要 pair 返回）。

    Returns:
        (sharpe, sortino)
    """
    # Sharpe 走统一入口，避免重复实现导致口径漂移
    sharpe = _compute_sharpe_from_equity(equity_curve, risk_free_rate)

    # Sortino 需要 returns 序列，此处保留实现
    if isinstance(equity_curve, pd.DataFrame):
        vals = equity_curve["portfolio_value"].values.astype(float)
    else:
        vals = np.array([e.get("portfolio_value", 0) for e in equity_curve], dtype=float)

    finite_mask = np.isfinite(vals)
    if finite_mask.sum() < 2:
        return 0.0, 0.0
    vals = vals[finite_mask]
    if vals[0] <= 0:
        return 0.0, 0.0

    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return 0.0, 0.0

    ann_factor = _ann_factor()
    mu = returns.mean() * ann_factor
    excess_mu = mu - risk_free_rate

    downside = returns[returns < 0]
    _SORTINO_CEILING = 100.0
    if len(downside) == 0:
        sortino = _SORTINO_CEILING if excess_mu > 0 else 0.0
    else:
        downside_std = downside.std(ddof=1) * math.sqrt(ann_factor)
        raw_sortino = excess_mu / downside_std if downside_std > 0 else (_SORTINO_CEILING if excess_mu > 0 else 0.0)
        sortino = min(raw_sortino, _SORTINO_CEILING)

    return float(sharpe), float(sortino)


def validate_oos_decay(
    is_equity_curve: list[dict[str, Any]] | pd.DataFrame,
    oos_equity_curve: list[dict[str, Any]] | pd.DataFrame,
    *,
    decay_threshold: float = 0.30,
    weak_threshold: float = 0.50,
    is_days: int = 0,
    oos_days: int = 0,
) -> OOSDecayReport:
    """样本外衰减校验（三态分类）—— 核心 gate。

    三态判定逻辑：
      - INVALID:   IS_Sharpe ≤ 0，策略在样本内无超额收益，不是过拟合
      - OVERFITTED: IS_Sharpe > 0 但 OOS Sharpe ≤ 0（100% 衰减）或 Sharpe 衰减 > 50%
      - WEAK:      IS_Sharpe > 0 且衰减 30%-50%，策略信号偏弱

    Args:
        is_equity_curve: 样本内（训练集）净值曲线。
        oos_equity_curve: 样本外（独立测试集）净值曲线。
        decay_threshold: 弱信号容忍度，默认 30%。
        weak_threshold: 弱信号 vs 过拟合分界，默认 50%。
        is_days: 样本内交易日数（报告用）。
        oos_days: 样本外交易日数（报告用）。

    Returns:
        OOSDecayReport —— passed=False 时结果应直接废弃，overfit_type 指示原因。
    """
    report = OOSDecayReport(
        is_sample_days=is_days,
        oos_sample_days=oos_days,
    )

    is_sharpe, is_sortino = _compute_risk_from_curve(is_equity_curve)
    oos_sharpe, oos_sortino = _compute_risk_from_curve(oos_equity_curve)

    report.is_sharpe = is_sharpe
    report.is_sortino = is_sortino
    report.oos_sharpe = oos_sharpe
    report.oos_sortino = oos_sortino

    # ── Gate 1: IS Sharpe ≤ 0 → 策略无效（不是过拟合） ──
    if is_sharpe <= 0:
        report.passed = False
        report.overfit_type = OverfitType.INVALID
        report.reason = (
            f"策略无效：样本内 Sharpe = {is_sharpe:.2f} ≤ 0，"
            f"模型在训练窗口本身无超额收益信号，非过拟合问题"
        )
        report.log()
        return report

    # ── Gate 2: OOS Sharpe ≤ 0 → 真实过拟合（IS 正但 OOS 完全失效） ──
    if oos_sharpe <= 0:
        report.sharpe_decay = 1.0
        report.sortino_decay = 1.0
        report.passed = False
        report.overfit_type = OverfitType.OVERFITTED
        report.reason = (
            f"过拟合：IS_Sharpe={is_sharpe:.2f} 但 OOS_Sharpe={oos_sharpe:.2f} ≤ 0，"
            f"衰减 100%，模型在样本外完全失效"
        )
        report.log()
        return report

    # ── Gate 3: OOS 样本量不足 ──
    if oos_days and oos_days < _MIN_OOS_DAYS:
        report.passed = False
        report.overfit_type = OverfitType.WEAK
        report.reason = (
            f"样本外交易日仅 {oos_days} 天（<{_MIN_OOS_DAYS}），"
            f"Sharpe 估计噪声过大，无法可靠判定衰减"
        )
        report.log()
        return report

    # ── 衰减计算 ──
    sharpe_decay = 1.0 - (oos_sharpe / is_sharpe)
    report.sharpe_decay = sharpe_decay
    report.sharpe_ratio = oos_sharpe / is_sharpe

    if math.isfinite(is_sortino) and 0 < is_sortino < 900:
        sortino_decay = 1.0 - (oos_sortino / is_sortino)
    else:
        # IS 无下行样本（Sortino 截断为 999 大数）或不可计算时，
        # Sortino 比无可比性，衰减记为 0 不参与 gate（由 Sharpe 独立判定）。
        sortino_decay = 0.0
    report.sortino_decay = max(sortino_decay, 0.0)  # Sortino 衰减下限 0（OOS 更好不报错）

    # ── 三态判定 ──
    if sharpe_decay > weak_threshold or sortino_decay > weak_threshold:
        # 衰减 > 50% → 真实过拟合
        report.passed = False
        report.overfit_type = OverfitType.OVERFITTED
        report.reason = (
            f"过拟合：IS_Sharpe={is_sharpe:.2f} → OOS_Sharpe={oos_sharpe:.2f}，"
            f"Sharpe 衰减 {sharpe_decay:.1%} > {weak_threshold:.0%}，"
            f"模型在样本外泛化能力严重不足，结果废弃"
        )
    elif sharpe_decay > decay_threshold or sortino_decay > decay_threshold:
        # 衰减 30%-50% → 策略偏弱（告警但不直接废弃）
        report.passed = False
        report.overfit_type = OverfitType.WEAK
        report.reason = (
            f"策略偏弱：IS_Sharpe={is_sharpe:.2f} → OOS_Sharpe={oos_sharpe:.2f}，"
            f"Sharpe 衰减 {sharpe_decay:.1%}（>{decay_threshold:.0%}）但 ≤ {weak_threshold:.0%}，"
            f"策略信号偏弱，结果仅供参考"
        )
    else:
        # 衰减 ≤ 30% → 通过
        report.passed = True
        report.overfit_type = OverfitType.VALID
        report.reason = (
            f"泛化通过：Sharpe 衰减 {sharpe_decay:.1%} ≤ {decay_threshold:.0%}，"
            f"Sortino 衰减 {sortino_decay:.1%} ≤ {decay_threshold:.0%}，"
            f"模型泛化性通过"
        )

    report.log()
    return report
