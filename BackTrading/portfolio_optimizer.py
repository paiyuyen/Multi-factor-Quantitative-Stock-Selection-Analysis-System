"""
组合优化器 — CVXPY 数学规划驱动

替代 Top-K 等权分配，将回测引擎从"规则驱动"升级为"数学规划驱动"。

核心目标函数：
    max_w  wᵀμ - (γ/2)wᵀΣw - λ_TC · ‖w - w₀‖₁

约束：
    Σwᵢ = 1 - cash_ratio          (控仓)
    0 ≤ wᵢ ≤ w_max                (禁止做空 + 单票上限)
    |Σ_{i∈Ind} wᵢ - w_bench_Ind| ≤ ε  (行业中性)
    ‖w‖₀ ≤ max_holdings           (最大持仓数, 通过 L1 正则近似)

作者: Baisys Quant
版本: 1.0.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# ── CVXPY 求解器（DCP 凸优化，自动选择最优求解器） ──
# Windows 下 cvxpy DLL 加载可能直接崩溃进程（非 ImportError），用子进程隔离检测
def _check_cvxpy() -> bool:
    """在子进程中检测 cvxpy 可用性，避免 DLL 崩溃影响主进程"""
    import subprocess
    import sys
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import cvxpy; print('OK')"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and "OK" in result.stdout
    except Exception:
        return False


HAVE_CVXPY = _check_cvxpy()
cp = None
if HAVE_CVXPY:
    try:
        import cvxpy as cp
    except Exception:  # 子进程检测通过但主进程加载失败
        logger.warning(
            "[PortfolioOptimizer] cvxpy 主进程导入失败，将回退至 scipy SLSQP。"
        )
        HAVE_CVXPY = False
        cp = None
if not HAVE_CVXPY:
    logger.warning(
        "[PortfolioOptimizer] cvxpy 不可用（DLL 加载失败或未安装），将回退至 scipy SLSQP。"
    )
    logger.warning(
        "[PortfolioOptimizer] ★★★ 求解器降级告警 ★★★\n"
        "  SLSQP 为局部搜索求解器，L1 正则化下目标函数非凸 → 结果可能非全局最优。\n"
        "  建议安装 cvxpy（pip install cvxpy）或确认 Windows 系统级 VC++ Redistributable 已安装。\n"
        "  本次优化结果仅供参考，不保证组合风险预算约束被精确满足。"
    )


# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    """组合优化器配置"""

    # 优化方法: mean_variance / min_variance / risk_parity / topk_equal(兼容)
    method: str = "mean_variance"

    # 风险厌恶系数 (γ, 越大越保守; 2.0 = 平衡, 5.0 = 保守)
    risk_aversion: float = 2.0

    # 换手率惩罚系数 (λ_TC, 建议取年化交易成本的 0.5~1.5 倍)
    # 典型值: 0.0005 (万5, 轻度惩罚) ~ 0.002 (千2, 强惩罚)
    turnover_penalty: float = 0.001

    # 单票权重上限
    max_weight: float = 0.10

    # 行业暴露偏离上限 (对基准, 绝对值)
    max_industry_deviation: float = 0.05

    # P1-4 修复：A股融券限制——默认禁止做空（A股融券标的有限且成本高）
    short_allowed: bool = False
    # 融券成本（年化，默认7%），做空时叠加到目标函数中作为持有成本惩罚
    short_cost_annual: float = 0.07

    # 协方差估计窗口 (交易日)
    # P1-3 修复：A股市场结构变化快，60天不足以捕捉稳定协方差结构，提升至120天
    cov_lookback: int = 120

    # EWMA 衰减因子 (RiskMetrics 标准 0.94)
    ewma_lambda: float = 0.94

    # 协方差收缩 (Ledoit-Wolf, 小样本稳健)
    shrinkage: bool = True

    # P0.7 修复：样本量低于 min_samples 时使用 Ledoit-Wolf 收缩，
    #       低于 min_samples/3 时 fallback 等权并标记协方差降级。
    cov_shrink: float = 0.5
    cov_shrink_threshold: int = 30  # 低于此行数强制收缩协方差

    # 最大持仓数 (0 = 不限制)
    max_holdings: int = 0

    # 目标现金比例
    target_cash_ratio: float = 0.0

    # 是否用上期权重做 warm start
    warm_start: bool = True

    # P3-3：日志详细度控制（WFO 路径下降噪）
    # True = 输出所有 debug/info；False = 仅输出 warning 以上
    verbose: bool = False

    # 求解超时 (秒) — 超过则回退
    solve_timeout: float = 5.0

    # CVXPY 求解器回退链 (自动按优先级尝试)
    solver_chain: tuple[str, ...] = (
        "CLARABEL",    # 首选: 快速、稳定、纯 Python
        "ECOS",        # 备选 1
        "SCS",         # 备选 2 (近似求解器, 速度快)
        "OSQP",        # 备选 3 (QP 专用)
    )


# ─────────────────────────────────────────────────────────────
# 协方差矩阵估计
# ─────────────────────────────────────────────────────────────

class CovarianceEstimator:
    """协方差矩阵估计器 — 支持 EWMA / Ledoit-Wolf / Sample"""

    @staticmethod
    def ewma(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
        """指数加权协方差矩阵 (RiskMetrics 标准 λ=0.94)

        近期数据权重更高，对结构性变化响应更快。
        """
        T, N = returns.shape
        weights = np.array([(1 - lam) * lam ** (T - 1 - t) for t in range(T)])
        weights /= weights.sum()

        mean = np.average(returns, axis=0, weights=weights)
        centered = returns - mean

        cov = np.zeros((N, N))
        for t in range(T):
            cov += weights[t] * np.outer(centered[t], centered[t])

        return cov

    @staticmethod
    def sample(returns: np.ndarray) -> np.ndarray:
        """样本协方差矩阵 (MLE)"""
        return np.cov(returns, rowvar=False)

    @staticmethod
    def ledoit_wolf(
        returns: np.ndarray,
        delta: float | None = None,
    ) -> np.ndarray:
        """Ledoit-Wolf 收缩估计器 (小样本稳健)

        收缩目标: 对角阵 (各资产独立)
        收缩强度: 自动选择最优 (最小化 MSE)

        适用于候选资产多、样本少时的矩阵条件数爆炸问题。
        """
        T, N = returns.shape
        sample_cov = np.cov(returns, rowvar=False)

        # 收缩目标：方差对角阵
        target = np.diag(np.diag(sample_cov))

        if delta is None:
            # 自动计算最优收缩强度
            mean_ret = np.mean(returns, axis=0)
            X = returns - mean_ret

            # 上界: 样本协方差与收缩目标之间的平方距离
            Phi = np.mean(X[:, :, None] * X[:, None, :], axis=0)
            upper_bound = np.sum((sample_cov - target) ** 2)

            if upper_bound < 1e-20:
                return target.copy()

            # 下界: 逐元素方差估计
            lower_bound = 0.0
            for i in range(N):
                for j in range(N):
                    var_ij = np.mean((X[:, i] * X[:, j] - sample_cov[i, j]) ** 2)
                    lower_bound += var_ij

            delta = min(1.0, max(0.0, lower_bound / upper_bound))

        # 收缩协方差
        cov = delta * target + (1 - delta) * sample_cov
        logger.debug(f"[CovarianceEstimator] Ledoit-Wolf δ = {delta:.4f}")

        return cov

    @classmethod
    def estimate(
        cls,
        returns: np.ndarray,
        method: str = "ewma",
        lam: float = 0.94,
        shrinkage: bool = False,
    ) -> np.ndarray:
        """统一入口

        Args:
            returns: (T, N) 收益率矩阵
            method: 'ewma' / 'ledoit_wolf' / 'sample'
            lam: EWMA 衰减因子
            shrinkage: 是否启用 Ledoit-Wolf 收缩

        Returns:
            (N, N) 协方差矩阵 (确保正定)
        """
        T, N = returns.shape
        # P0.7 修复：小样本协方差估计守卫 + 自动收缩
        if T < N:
            logger.warning(
                f"[CovarianceEstimator] 样本量严重不足 (T={T} < N={N})，"
                f"协方差矩阵必然奇异 → 强制使用 Ledoit-Wolf 收缩估计"
            )
            cov = cls.ledoit_wolf(returns)
        elif T < 60:
            # 样本量偏低（低于默认 cov_lookback=120），启用自动收缩
            logger.warning(
                f"[CovarianceEstimator] 样本量偏低 (T={T} < 60)，"
                f"协方差估计不稳定 → 启用 Ledoit-Wolf 收缩加固"
            )
            cov = cls.ledoit_wolf(returns)
            # 再用基础估计做加权混合（收缩为主）
            base_cov = cls.sample(returns)
            cov = 0.4 * base_cov + 0.6 * cov
        else:
            # P1-3 修复：EWMA 与 Ledoit-Wolf 收缩组合使用
            # 先用 method 计算基础协方差，再可选叠加收缩增强稳健性
            if method == "ewma":
                cov = cls.ewma(returns, lam)
            elif method == "ledoit_wolf":
                cov = cls.ledoit_wolf(returns)
            else:
                cov = cls.sample(returns)

            # 若启用收缩且未使用 ledoit_wolf，则对基础估计再叠加收缩
            if shrinkage and method != "ledoit_wolf":
                shrunk = cls.ledoit_wolf(returns)
                # 以基础估计为主，收缩估计为稳健锚，加权平均
                cov = 0.7 * cov + 0.3 * shrunk

        # 确保正定 (最小特征值平移)
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 0:
            min_shift = abs(eigvals.min()) + 1e-8
            cov += min_shift * np.eye(cov.shape[0])
            logger.debug(
                f"[CovarianceEstimator] 修正非正定: min_eig = {eigvals.min():.2e} → +{min_shift:.2e}"
            )

        return cov


# ─────────────────────────────────────────────────────────────
# 组合优化器 (CVXPY + SLSQP 双引擎)
# ─────────────────────────────────────────────────────────────

class PortfolioOptimizer:
    """组合优化器 — 支持均值方差 / 最小方差 / 风险平价

    使用 CVXPY 声明式凸优化建模，支持 DCP 规则验证。
    CVXPY 不可用时回退 scipy SLSQP。

    目标函数：
        min_w  -wᵀμ + (γ/2)wᵀΣw + λ_TC · ‖w - w₀‖₁
        (等价于 max_w  wᵀμ - (γ/2)wᵀΣw - λ_TC · ‖w - w₀‖₁)

    约束：
        Σwᵢ = 1 - cash_ratio
        0 ≤ wᵢ ≤ w_max
        |Σ_{i∈Ind} wᵢ - w_bench_Ind| ≤ ε  (行业中性)
    """

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.cfg = config or OptimizerConfig()
        self._last_risk_model_fallback: bool = False  # P1.18: scipy fallback 可追踪标记

    # ── 主入口 ──────────────────────────────────────────────

    def optimize(
        self,
        candidate_symbols: list[str],
        alpha_signals: np.ndarray,          # (n,) 预期收益 / 评分
        returns_history: pd.DataFrame,       # (T, n) 历史收益率, columns = symbols
        current_weights: dict[str, float],   # 当前持仓权重
        industry_map: dict[str, str] | None = None,
        benchmark_industry_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """执行组合优化，返回目标权重。

        Args:
            candidate_symbols: 候选股票列表
            alpha_signals: 每个候选的预期收益/评分 (n,)
            returns_history: 历史收益率 DataFrame (T × n)
            current_weights: 当前持仓权重 {symbol: weight}
            industry_map: 股票→行业映射 {symbol: industry_name}
            benchmark_industry_weights: 基准行业权重 {industry: weight}

        Returns:
            {symbol: weight} 目标权重字典
        """
        n = len(candidate_symbols)
        if n == 0:
            return {}

        # 准备收益率矩阵
        ret_matrix = self._prepare_returns(returns_history, candidate_symbols)
        if ret_matrix is None:
            logger.debug(
                f"[Optimizer] 收益率数据不足 (n={n}), 回退等权"
            )
            return self._fallback_equal_weight(candidate_symbols)

        # 协方差估计
        try:
            cov = CovarianceEstimator.estimate(
                ret_matrix,
                method="ewma" if not self.cfg.shrinkage else "ledoit_wolf",
                lam=self.cfg.ewma_lambda,
                shrinkage=self.cfg.shrinkage,
            )
        except Exception as e:
            logger.warning(f"[Optimizer] 协方差估计失败: {e}，回退等权")
            return self._fallback_equal_weight(candidate_symbols)

        # 上一期权重向量
        w0 = np.array([current_weights.get(s, 0.0) for s in candidate_symbols])

        # Alpha 归一化 (将评分映射为合理量级的预期收益)
        mu = self._normalize_alpha(alpha_signals, ret_matrix)

        # 行业分组
        industry_groups = self._build_industry_groups(
            candidate_symbols, industry_map
        )

        # 按方法路由
        method = self.cfg.method
        if method == "mean_variance":
            return self._mean_variance_opt(
                candidate_symbols, mu, cov, w0,
                industry_groups, benchmark_industry_weights,
            )
        elif method == "min_variance":
            return self._min_variance_opt(
                candidate_symbols, cov, w0,
                industry_groups, benchmark_industry_weights,
            )
        elif method == "risk_parity":
            return self._risk_parity_opt(
                candidate_symbols, cov,
            )
        elif method == "topk_equal":
            return self._fallback_equal_weight(candidate_symbols)
        else:
            logger.warning(f"[Optimizer] 未知方法 '{method}'，回退等权")
            return self._fallback_equal_weight(candidate_symbols)

    # ── 均值方差优化 (CVXPY 声明式) ───────────────────────

    def _mean_variance_opt(
        self,
        symbols: list[str],
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
        industry_groups: list[list[int]] | None,
        benchmark_weights: dict[str, float] | None,
    ) -> dict[str, float]:
        """均值方差优化 — CVXPY 声明式建模 + SLSQP 回退"""
        n = len(symbols)
        target_sum = 1.0 - self.cfg.target_cash_ratio

        if HAVE_CVXPY:
            result = self._mean_variance_cvxpy(
                symbols, mu, cov, w0, target_sum,
                industry_groups, benchmark_weights,
            )
            if result is not None:
                return result

        # CVXPY 失败或未安装 → scipy 回退
        if not HAVE_CVXPY and n > 10:
            logger.warning(
                f"[Optimizer] CVXPY 不可用 ({n} 只候选)，回退 scipy SLSQP；"
                f"高维约束求解可靠性降低，建议使用凸规划求解器"
            )
        # P1.15 修复：CVXPY 失败回退时显式 WARN 告警（含行业中性约束静默放宽风险）
        # 原 logger.debug 掩盖了 SLSQP 局部最优导致行业中性约束可能失效的问题
        logger.warning(
            f"[Optimizer] CVXPY 不可用或失败，回退 scipy SLSQP（{n} 只候选）→ "
            f"行业中性约束可能未严格满足（局部最优近似）"
        )
        # P1.18 修复：标记 scipy fallback 供下游追踪验证
        self._last_risk_model_fallback = True
        return self._mean_variance_scipy(
            symbols, mu, cov, w0, target_sum,
            industry_groups, benchmark_weights,
        )

    def _mean_variance_cvxpy(
        self,
        symbols: list[str],
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
        target_sum: float,
        industry_groups: list[list[int]] | None,
        benchmark_weights: dict[str, float] | None,
    ) -> dict[str, float] | None:
        """CVXPY 声明式均值方差优化"""
        n = len(symbols)
        # 延迟导入 cvxpy (避免 DLL 崩溃)
        import cvxpy as cp
        w = cp.Variable(n)

        # 目标函数:  min  -wᵀμ + (γ/2)wᵀΣw + λ_TC · ‖w - w₀‖₁ + 融券成本
        risk_term = (self.cfg.risk_aversion / 2.0) * cp.quad_form(w, cov)
        return_term = -mu.T @ w
        turnover_term = self.cfg.turnover_penalty * cp.norm(w - w0, 1)

        # P1-4 修复：融券成本项 — 对负权重部分加收年化融券成本（折算为日成本）
        short_cost_term = cp.Constant(0.0)
        if self.cfg.short_allowed:
            short_cost_daily = self.cfg.short_cost_annual / 244.0  # A股年化交易日244
            w_neg = cp.maximum(-w, 0)
            short_cost_term = short_cost_daily * cp.sum(w_neg)

        objective = cp.Minimize(return_term + risk_term + turnover_term + short_cost_term)

        # 约束
        constraints = [
            cp.sum(w) == target_sum,
        ]

        # P1-4 修复：A股默认禁止做空；short_allowed=False 时禁止负权重
        if self.cfg.short_allowed:
            constraints.append(w >= -self.cfg.max_weight)
        else:
            constraints.append(0 <= w)
            constraints.append(w <= self.cfg.max_weight)

        # 行业中性约束
        if industry_groups is not None and benchmark_weights is not None:
            eps = self.cfg.max_industry_deviation
            for grp in industry_groups:
                # 取行业名
                ind_name = None
                for idx in grp:
                    s = symbols[idx] if idx < len(symbols) else None
                    if s and hasattr(self, '_industry_map') and self._industry_map:
                        ind_name = self._industry_map.get(s)
                        break

                if ind_name is None:
                    # 尝试从行业分组推断
                    ind_name = f"_group_{len(constraints)}"

                bench_w = benchmark_weights.get(ind_name, 0.0)
                ind_weight = cp.sum(w[grp])

                constraints.append(ind_weight - bench_w <= eps)
                constraints.append(bench_w - ind_weight <= eps)
        elif industry_groups is not None:
            # 无基准时的行业上限 (3 * max_weight 作为上限)
            for grp in industry_groups:
                constraints.append(cp.sum(w[grp]) <= self.cfg.max_weight * 3)

        prob = cp.Problem(objective, constraints)

        # 求解器链 (尝试优先级)
        t0 = time.time()
        for solver in self.cfg.solver_chain:
            try:
                prob.solve(
                    solver=solver,
                    verbose=False,
                    max_iters=2000,
                    time_limit=self.cfg.solve_timeout,
                )
                elapsed = time.time() - t0
                if prob.status in ("optimal", "optimal_inaccurate"):
                    logger.debug(
                        f"[Optimizer] CVXPY 求解成功 (solver={solver}, "
                        f"status={prob.status}, time={elapsed:.3f}s)"
                    )
                    w_opt = np.clip(w.value, 0, None)
                    if w_opt.sum() > 0:
                        w_opt = w_opt / w_opt.sum() * target_sum
                    # P1-17 修复：清理微小权重后重归一化，避免 Σw 偏离约束
                    w_opt[w_opt < 0.001] = 0
                    if w_opt.sum() > 0:
                        w_opt = w_opt / w_opt.sum() * target_sum
                    return dict(zip(symbols, w_opt))
                elif prob.status in ("infeasible", "unbounded"):
                    logger.debug(f"[Optimizer] CVXPY {solver}: {prob.status}")
                    continue
            except Exception as e:
                logger.debug(f"[Optimizer] CVXPY {solver} 失败: {e}")
                continue

        elapsed = time.time() - t0
        if elapsed >= self.cfg.solve_timeout:
            logger.debug(f"[Optimizer] CVXPY 超时 ({elapsed:.1f}s)")

        return None

    def _mean_variance_scipy(
        self,
        symbols: list[str],
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
        target_sum: float,
        industry_groups: list[list[int]] | None,
        benchmark_weights: dict[str, float] | None,
    ) -> dict[str, float]:
        """scipy SLSQP 回退求解"""
        from scipy.optimize import minimize

        n = len(symbols)

        def objective(w: np.ndarray) -> float:
            return_term = -w @ mu
            risk_term = (self.cfg.risk_aversion / 2.0) * w @ cov @ w
            turnover = self.cfg.turnover_penalty * np.sum(np.abs(w - w0))
            if self.cfg.short_allowed:
                short_cost_daily = self.cfg.short_cost_annual / 244.0  # A股年化交易日244
                short_pos = np.minimum(w, 0)
                return return_term + risk_term + turnover + short_cost_daily * np.sum(np.abs(short_pos))
            return return_term + risk_term + turnover

        constraints: list[dict[str, Any]] = [
            {"type": "eq", "fun": lambda w: np.sum(w) - target_sum},
        ]

        if industry_groups is not None and benchmark_weights is not None:
            eps = self.cfg.max_industry_deviation
            for grp in industry_groups:
                bench_w = benchmark_weights.get(
                    symbols[grp[0]] if grp else "", 0.0
                )
                constraints.extend([
                    {
                        "type": "ineq",
                        "fun": lambda w, g=grp, b=bench_w, e=eps:
                            e - (np.sum(w[g]) - b),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda w, g=grp, b=bench_w, e=eps:
                            e + (np.sum(w[g]) - b),
                    },
                ])
        elif industry_groups is not None:
            for grp in industry_groups:
                constraints.append({
                    "type": "ineq",
                    "fun": lambda w, g=grp:
                        self.cfg.max_weight * 3 - np.sum(w[g]),
                })

        if self.cfg.short_allowed:
            bounds = [(-self.cfg.max_weight, self.cfg.max_weight)] * n
        else:
            bounds = [(0, self.cfg.max_weight)] * n

        # Warm start
        if self.cfg.warm_start and w0.sum() > 0:
            if self.cfg.short_allowed:
                x0 = np.clip(w0, -self.cfg.max_weight, self.cfg.max_weight)
            else:
                x0 = np.clip(w0, 0, self.cfg.max_weight)
            if x0.sum() > 0:
                x0 = x0 / x0.sum() * target_sum
            else:
                x0 = np.ones(n) / n
        else:
            x0 = np.ones(n) / n

        result = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if result.success:
            w_opt = np.clip(result.x, 0, None)
            if w_opt.sum() > 0:
                w_opt = w_opt / w_opt.sum() * target_sum
            w_opt[w_opt < 0.001] = 0
            if w_opt.sum() > 0:
                w_opt = w_opt / w_opt.sum() * target_sum

            # 回退模式权重后约束验证
            if np.any(w_opt < -1e-9):
                logger.warning(
                    f"[Optimizer] SLSQP回退产生负权重，强制截断 "
                    f"(min_weight={w_opt.min():.6f})"
                )
                w_opt = np.clip(w_opt, 0, None)
                if w_opt.sum() > 0:
                    w_opt = w_opt / w_opt.sum() * target_sum
            abs_sum_err = abs(w_opt.sum() - target_sum)
            if abs_sum_err > 1e-4:
                logger.warning(
                    f"[Optimizer] SLSQP回退权重Σw={w_opt.sum():.6f}偏离目标{target_sum} "
                    f"(Δ={abs_sum_err:.6f})，重新归一化"
                )
                w_opt = w_opt / w_opt.sum() * target_sum

            return dict(zip(symbols, w_opt))

        logger.warning(
            f"[Optimizer] scipy 求解失败: {result.message}，回退等权"
        )
        return self._fallback_equal_weight(symbols)

    # ── 最小方差组合 ───────────────────────────────────────

    def _min_variance_opt(
        self,
        symbols: list[str],
        cov: np.ndarray,
        w0: np.ndarray,
        industry_groups: list[list[int]] | None,
        benchmark_weights: dict[str, float] | None,
    ) -> dict[str, float]:
        """最小方差组合 — 最小化组合风险"""
        n = len(symbols)

        if HAVE_CVXPY:
            w = cp.Variable(n)
            objective = cp.Minimize(
                cp.quad_form(w, cov)
                + self.cfg.turnover_penalty * cp.norm(w - w0, 1)
            )
            constraints = [
                cp.sum(w) == 1.0,
                0 <= w,
                w <= self.cfg.max_weight,
            ]

            if industry_groups:
                eps = self.cfg.max_industry_deviation
                for grp in industry_groups:
                    ind_weight = cp.sum(w[grp])
                    if benchmark_weights:
                        bench_w = benchmark_weights.get(
                            symbols[grp[0]] if grp else "", 0.0
                        )
                        constraints.extend([
                            ind_weight - bench_w <= eps,
                            bench_w - ind_weight <= eps,
                        ])
                    else:
                        constraints.append(ind_weight <= self.cfg.max_weight * 3)

            prob = cp.Problem(objective, constraints)
            for solver in self.cfg.solver_chain:
                try:
                    prob.solve(solver=solver, verbose=False)
                    if prob.status in ("optimal", "optimal_inaccurate"):
                        w_opt = np.clip(w.value, 0, None)
                        # P1-17 修复：清零微小权重后重归一化
                        w_opt[w_opt < 0.001] = 0
                        if w_opt.sum() > 0:
                            w_opt /= w_opt.sum()
                        return dict(zip(symbols, w_opt))
                except Exception:
                    continue

        # scipy 回退
        from scipy.optimize import minimize

        # P1.18 修复：标记 scipy fallback 供下游验证
        self._last_risk_model_fallback = True
        logger.debug(
            f"[Optimizer] 最小方差 scipy SLSQP 回退（{n} 只候选）→ "
            f"risk_model_fallback=True"
        )

        def obj_func(w_vec: np.ndarray) -> float:
            return (w_vec @ cov @ w_vec
                    + self.cfg.turnover_penalty * np.sum(np.abs(w_vec - w0)))

        result = minimize(
            obj_func, np.ones(n) / n, method="SLSQP",
            bounds=[(0, self.cfg.max_weight)] * n,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
            options={"maxiter": 1000},
        )

        if result.success:
            w_opt = np.clip(result.x, 0, None)
            # P1-17 修复：清零微小权重后重归一化
            w_opt[w_opt < 0.001] = 0
            if w_opt.sum() > 0:
                w_opt /= w_opt.sum()
            return dict(zip(symbols, w_opt))

        return self._fallback_equal_weight(symbols)

    # ── 风险平价 ───────────────────────────────────────────

    def _risk_parity_opt(
        self,
        symbols: list[str],
        cov: np.ndarray,
    ) -> dict[str, float]:
        """风险平价 — 每项资产对组合风险的贡献相等

        通过梯度迭代求解，不依赖求解器。
        """
        n = len(symbols)
        max_w = self.cfg.max_weight

        if n <= 1:
            return {symbols[0]: min(1.0, max_w)}

        # CVXPY 版本 (精确求解)
        if HAVE_CVXPY:
            try:
                w = cp.Variable(n)
                sigma_p = cp.sqrt(cp.quad_form(w, cov))
                # Marginal risk contribution
                mrc = cov @ w / sigma_p
                # 目标: 各资产风险贡献相等
                target_risk = cp.Constant(1.0 / n)
                risk_contrib = w * mrc
                objective = cp.Minimize(
                    cp.sum_squares(risk_contrib - 1.0 / n)
                )
                constraints = [
                    cp.sum(w) == 1.0,
                    w >= 1e-6,
                    w <= max_w,
                ]
                prob = cp.Problem(objective, constraints)
                for solver in self.cfg.solver_chain:
                    try:
                        prob.solve(solver=solver, verbose=False)
                        if prob.status in ("optimal", "optimal_inaccurate"):
                            w_opt = np.clip(w.value, 0, None)
                            if w_opt.sum() > 0:
                                w_opt /= w_opt.sum()
                            return dict(zip(symbols, w_opt))
                    except Exception:
                        continue
            except Exception:
                pass

        # 梯度迭代法 (经典实现) — CVXPY 不可用时回退
        # P1.18 修复：标记 risk_model_fallback 供下游追踪验证
        self._last_risk_model_fallback = True
        logger.debug(
            f"[Optimizer] 风险平价梯度迭代回退（{n} 只候选）→ "
            f"risk_model_fallback=True，无凸规划保障"
        )
        x = np.ones(n) / n
        for _ in range(200):
            sigma = np.sqrt(x @ cov @ x)
            if sigma < 1e-12:
                break
            mrc = cov @ x / sigma
            target = np.mean(mrc)
            # 避免除以零
            safe_mrc = np.clip(mrc, 1e-12, None)
            x = x * (target / safe_mrc)
            x = np.clip(x, 0, max_w)
            x = x / x.sum()

        return dict(zip(symbols, x))

    # ── 辅助方法 ───────────────────────────────────────────

    def _normalize_alpha(
        self,
        alpha: np.ndarray,
        returns: np.ndarray,
    ) -> np.ndarray:
        """将 Alpha 评分归一化为与协方差同量级的预期收益

        问题: buy_score (0~100) 与日收益率 (~0.01) 量级不同，
              直接相减会导致优化器被收益项主导。
        解决: 将 Alpha 线性映射到历史收益率范围。
        P1-17 修复：映射到 [-ret_std, +ret_std] 范围而非全正值，
              避免优化器因全正 alpha 盲目重仓。
        """
        if len(alpha) == 0:
            return alpha

        # 取历史收益率的均值和标准差作为参考尺度
        ret_mean = np.mean(np.abs(returns))
        if ret_mean < 1e-10:
            return np.zeros_like(alpha)

        # 将 Alpha 归一化到 [-1, 1]（而非 [0,1]），保留正负区分
        a_min, a_max = alpha.min(), alpha.max()
        if a_max - a_min < 1e-10:
            normalized = np.zeros_like(alpha)  # 全相同分 → 中性预期
        else:
            normalized = 2.0 * (alpha - a_min) / (a_max - a_min) - 1.0  # [-1, 1]

        # 映射到收益率尺度 (均值 ± 标准化偏差范围)
        ret_std = np.std(returns)
        mu = ret_mean + normalized * ret_std

        return mu

    def _prepare_returns(
        self,
        returns_history: pd.DataFrame,
        symbols: list[str],
    ) -> np.ndarray | None:
        """准备收益率矩阵 (T, n)

        对齐可用列、处理缺失、填充不可用资产。
        """
        available = [s for s in symbols if s in returns_history.columns]
        if len(available) < self.cfg.min_candidates:
            return None

        ret = returns_history[available].tail(self.cfg.cov_lookback).dropna(how="all")

        # 对齐到所有候选
        if len(available) < len(symbols):
            missing = [s for s in symbols if s not in available]
            if len(ret) > 0:
                market_ret = ret.mean(axis=1)
                for s in missing:
                    ret[s] = market_ret

        if len(symbols) > ret.shape[1]:
            return None

        ret = ret[symbols].dropna().values

        if (ret.shape[0] < self.cfg.min_samples
                or ret.shape[1] < self.cfg.min_candidates):
            return None

        return ret

    def _build_industry_groups(
        self,
        symbols: list[str],
        industry_map: dict[str, str] | None,
    ) -> list[list[int]] | None:
        """构建行业分组索引列表 [[idx1, idx2], [idx3, idx4], ...]"""
        if industry_map is None:
            return None

        groups: dict[str, list[int]] = {}
        for i, sym in enumerate(symbols):
            ind = industry_map.get(sym)
            if ind:
                groups.setdefault(ind, []).append(i)

        # 过滤掉单只股票的行业 (无约束意义)
        multi_groups = [v for v in groups.values() if len(v) >= 2]
        return multi_groups if multi_groups else None

    def _fallback_equal_weight(self, symbols: list[str]) -> dict[str, float]:
        """回退：等权分配"""
        n = len(symbols)
        if n == 0:
            return {}
        # 先取上限，再归一化确保 Σw = 1。
        w_raw = min(1.0 / n, self.cfg.max_weight)
        w_sum = w_raw * n
        if w_sum < 1.0 - 1e-12:
            w_raw = 1.0 / n  # 回退等权，忽略 max_weight（fallback 场景无优化器约束）
            logger.debug(
                f"[Optimizer] fallback_equal_weight 权重和不足(Σw={w_sum:.4f})，"
                f"回退等权 1/n={1.0/n:.4f}"
            )
        return {s: w_raw for s in symbols}
