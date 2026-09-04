from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


def _ann_factor() -> int:
    return 244  # A股实际年化交易日数均值（非美股252）


# OOS 最少交易日数：低于此样本量时 Sharpe 估计噪声过大，衰减比不可判定
_MIN_OOS_DAYS = 10


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
) -> float:
    if len(equity_curve) < 2:
        return 0.5

    vals = pd.Series([e.get("portfolio_value", 0) for e in equity_curve]).values.astype(float)
    if len(vals) < 2:
        return 0.5

    returns = (vals[1:] - vals[:-1]) / vals[:-1]
    n = len(returns)
    if n < 2:
        return 0.5

    sharpe = float(returns.mean() / returns.std()) * math.sqrt(_ann_factor()) if returns.std() > 0 else 0.0
    skew = float(pd.Series(returns).skew())  # type: ignore[arg-type]
    kurt = float(pd.Series(returns).kurtosis()) + 3.0  # type: ignore[arg-type]

    return deflated_sharpe_ratio(sharpe, n, num_trials, skew, kurt)


# ──────────────────────────────────────────────────────────────
# Out-of-Sample 衰减校验（López de Prado 2018 准则 #1）
# ──────────────────────────────────────────────────────────────

@dataclass
class OOSDecayReport:
    """样本外衰减校验报告。

    业务定义：检验模型是否在历史特征上发生过度拟合（Overfitting）。
    样本外夏普比率相对于样本内夏普比率的衰减幅度不得超过 30%。
    若超过此阈值，判定模型对 XGBoost 等超参数进行了过度网格搜索，
    或特征工程存在隐性泄露，结果直接废弃。
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
            "passed": "PASS" if self.passed else "FAIL",
            "reason": self.reason,
            "is_days": self.is_sample_days,
            "oos_days": self.oos_sample_days,
        }

    def log(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        logger.info(
            f"[OOS衰减校验] {status} | IS_Sharpe={self.is_sharpe:.2f} → "
            f"OOS_Sharpe={self.oos_sharpe:.2f} (衰减 {self.sharpe_decay:.1%}) | "
            f"IS_Sortino={self.is_sortino:.2f} → OOS_Sortino={self.oos_sortino:.2f} "
            f"(衰减 {self.sortino_decay:.1%}) | {self.reason}"
        )


def _compute_risk_from_curve(
    equity_curve: list[dict[str, Any]] | pd.DataFrame,
    risk_free_rate: float = 0.03,
) -> tuple[float, float]:
    """从净值曲线计算年化 Sharpe 和 Sortino。

    P3 审计修复：新增 risk_free_rate 参数，与 backtest_metrics.compute_risk_metrics
    保持一致（超额收益口径），默认 3% 年化。

    Returns:
        (sharpe, sortino)
    """
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
    n = len(returns)
    if n < 2:
        return 0.0, 0.0

    ann_factor = _ann_factor()
    mu = returns.mean() * ann_factor
    excess_mu = mu - risk_free_rate  # P3 超额收益
    sigma = returns.std(ddof=1) * math.sqrt(ann_factor)
    sharpe = excess_mu / sigma if sigma > 0 else 0.0

    downside = returns[returns < 0]
    # P1/P3 对齐：无穷大 Sortino 截断为 100.0，使用超额收益
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
    is_days: int = 0,
    oos_days: int = 0,
) -> OOSDecayReport:
    """样本外衰减校验 —— 核心 gate。

    Args:
        is_equity_curve: 样本内（训练集）净值曲线。
        oos_equity_curve: 样本外（独立测试集）净值曲线。
        decay_threshold: 衰减容忍度，默认 30%。
        is_days: 样本内交易日数（报告用）。
        oos_days: 样本外交易日数（报告用）。

    Returns:
        OOSDecayReport —— passed=False 时结果应直接废弃。

    Raises:
        ValueError: 当样本内 Sharpe ≤ 0 时，拒绝计算衰减比（模型本身无信号）。
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

    # ── Guard: IS Sharpe ≤ 0 → 无信号，直接 FAIL ──
    if is_sharpe <= 0:
        report.passed = False
        report.reason = "样本内 Sharpe ≤ 0，模型本身无超额收益信号，拒绝衰减计算"
        report.log()
        return report

    # ── Guard: OOS Sharpe ≤ 0 → 样本外完全失效，直接 FAIL ──
    if oos_sharpe <= 0:
        report.sharpe_decay = 1.0  # 100% 衰减
        report.sortino_decay = 1.0
        report.passed = False
        report.reason = "样本外 Sharpe ≤ 0，模型在样本外完全失效"
        report.log()
        return report

    # ── Guard: OOS 样本量不足 → Sharpe 估计噪声过大，衰减比不可判定，拒绝通过 ──
    if oos_days and oos_days < _MIN_OOS_DAYS:
        report.passed = False
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

    # ── 判定：任一指标衰减超限则 FAIL ──
    if sharpe_decay > decay_threshold:
        report.passed = False
        report.reason = (
            f"Sharpe 衰减 {sharpe_decay:.1%} > {decay_threshold:.0%}，"
            f"疑似超参数过度网格搜索或特征工程隐性泄露，结果废弃"
        )
    elif sortino_decay > decay_threshold:
        report.passed = False
        report.reason = (
            f"Sortino 衰减 {sortino_decay:.1%} > {decay_threshold:.0%}，"
            f"下行风险在样本外显著恶化，结果废弃"
        )
    else:
        report.passed = True
        report.reason = (
            f"Sharpe 衰减 {sharpe_decay:.1%} ≤ {decay_threshold:.0%}，"
            f"Sortino 衰减 {sortino_decay:.1%} ≤ {decay_threshold:.0%}，"
            f"模型泛化性通过"
        )

    report.log()
    return report
