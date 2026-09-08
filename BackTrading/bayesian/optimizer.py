from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

from BackTrading.engine import EngineConfig
from BackTrading.bayesian.cost_model import FidelityController
from BackTrading.bayesian.kernel import (
    GPState,
    build_gp,
    restore_gp_state,
    save_gp_state,
)
from BackTrading.bayesian.acquisition import mixed_acquisition, optimize_acquisition
from BackTrading.bayesian.space import ParamSpace, split_by_cost, _get_fixed_params


# ── 参数归一化工具 ──

def _to_normalized(params: dict[str, float], spaces: dict[str, ParamSpace]) -> np.ndarray:
    """将参数 dict 映射为 [0,1]^d 向量（按 spaces 顺序）。"""
    arr = []
    for name, sp in spaces.items():
        val = params[name]
        if sp.high - sp.low < 1e-12:
            arr.append(0.5)
        else:
            arr.append((val - sp.low) / (sp.high - sp.low))
    return np.clip(arr, 0.0, 1.0)


def _from_normalized(
    x: np.ndarray, spaces: dict[str, ParamSpace]
) -> dict[str, float]:
    """将 [0,1]^d 向量还原为参数 dict，含离散化取整。"""
    names = list(spaces.keys())
    out: dict[str, float] = {}
    for i, name in enumerate(names):
        sp = spaces[name]
        raw = float(x[i]) * (sp.high - sp.low) + sp.low
        if sp.step is not None and sp.step > 0:
            n = int(round((sp.high - sp.low) / sp.step))
            ticks = sp.low + np.arange(n + 1) * sp.step
            raw = float(ticks[np.argmin(np.abs(ticks - raw))])
        out[name] = float(np.clip(raw, sp.low, sp.high))
    return out


# ── Sobol 初始采样 ──

def _sobol_samples(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Sobol 准随机序列 in [0,1]^d。向上取整到 2 的幂次再截断。"""
    n_pow2 = 1
    while n_pow2 < n:
        n_pow2 <<= 1
    sampler = Sobol(d, seed=seed, scramble=True)
    return sampler.random(n_pow2)[:n]


# ── 去重与 GP 去退化 ──

def _unique_x_agg(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按坐标去重聚合：重复点取 y 均值，修复 GP 退化（同 x 异 y → 表面平坦）。

    归一化坐标由同一管线生成（确定性），重复坐标必然对应同一参数组合；
    去重后 GP 拟合的观测集才是真唯一的参数点集合。
    """
    Xa = np.asarray(X, dtype=float)
    Ya = np.asarray(Y, dtype=float)
    if len(Xa) == 0:
        return Xa, Ya
    Xr = np.round(Xa, 10)
    _, idx, inv = np.unique(Xr, axis=0, return_index=True, return_inverse=True)
    if len(idx) == len(Xa):
        return Xa, Ya
    Y_agg = np.zeros(len(idx))
    np.add.at(Y_agg, inv, Ya)
    counts = np.bincount(inv)
    Y_agg /= np.maximum(counts, 1)
    return Xa[idx], Y_agg


# ── 局部随机探索（替代 L-BFGS-B 爬山） ──
# L-BFGS-B 在 GP 代理上会走向虚假夏普峰值（核函数平滑伪极值），
# 改为 ε-邻域 Sobol 采样：在已知最优附近随机扰动，不信任代理曲率。

def _local_refine(
    base_x: np.ndarray,
    gp_signal,  # GaussianProcessRegressor（仅用于获取 mu/sigma）
    signal_bounds: np.ndarray,
    n_samples: int = 20,
    epsilon: float = 0.1,
    seed: int = 42,
    n_random: int = 8,
) -> np.ndarray:
    """在已知最优的 ε-邻域内做 Sobol 采样，替代 L-BFGS-B。

    问题：L-BFGS-B 驱动参数走向 GP 核函数平滑产生的"虚假夏普峰值"，
    真实函数在该区域平坦或负 Sharpe。随机扰动不信任代理曲率，
    只在已验证最优附近探索，并用风险调整评分（mu - λ·sigma）。
    """
    rng = np.random.RandomState(seed)
    d = len(base_x)

    # 方案 1：Sobol 采样 ε-邻域
    n_pow2 = 1
    while n_pow2 < n_samples:
        n_pow2 <<= 1
    sampler = Sobol(d, seed=seed, scramble=True)
    sobol_pts = sampler.random(n_pow2)[:n_samples]

    candidates: list[np.ndarray] = []
    for s in sobol_pts:
        perturbed = np.clip(base_x + (s - 0.5) * 2 * epsilon, 0.0, 1.0)
        candidates.append(perturbed)

    # 方案 2：少量纯随机探索（防止局部最优）
    for _ in range(n_random):
        rand_x = rng.uniform(0, 1, d)
        blended = 0.7 * base_x + 0.3 * rand_x
        candidates.append(blended)

    # 用 GP 代理评分，取最优（风险调整：高不确定区域降分）
    best_x = base_x.copy()
    best_val = -1e10
    for cx in candidates:
        cx_reshaped = cx.reshape(1, -1)
        try:
            mu, sigma = gp_signal.predict(cx_reshaped, return_std=True)
            adjusted = mu[0] - 0.1 * sigma[0]  # 风险调整
            if adjusted > best_val:
                best_val = adjusted
                best_x = cx.copy()
        except Exception:
            continue

    return best_x


# ── 主优化函数 ──

def optimize_window(
    kline_df: "pd.DataFrame",
    engine_cfg: EngineConfig,
    spaces: dict[str, ParamSpace],
    *,
    n_init_signal: int = 15,
    n_iter_signal: int = 35,
    n_init_portfolio: int = 20,
    n_iter_portfolio: int = 150,
    n_refine_top: int = 3,
    seed: int = 42,
    previous_gp_state: GPState | None = None,
    progress_cb: Any = None,
    compute_exit_strategy: bool = True,
    eval_start_date: str | None = None,
    st_history: dict | None = None,
    exclude_st: bool = False,  # FIX(P0): 回测默认不排除 ST
    data_version: str | None = None,
    listing_days: dict | None = None,
    db_engine: Any = None,
) -> tuple[dict[str, float], GPState | None, list[dict[str, float]], float]:
    """单窗口贝叶斯优化（4 阶段）。

    Args:
        kline_df: 窗口内训练数据。
        engine_cfg: 基座 EngineConfig（params 叠加在其上）。
        spaces: 全参数空间。
        n_init_signal: Phase 1 Sobol 采样数。
        n_iter_signal: Phase 2 贝叶斯迭代数。
        n_init_portfolio: Phase 3 初始采样数（固定信号参数后）。
        n_iter_portfolio: Phase 3 贝叶斯迭代数。
        n_refine_top: Phase 4 精细化候选数。
        seed: 随机种子。
        previous_gp_state: 前一窗口的 GPState（warm-start）。
        progress_cb: 可选回调 progress_cb(phase, i, n, sharpe)。
        compute_exit_strategy: 传给 FidelityController。

    Returns:
        (best_params, gp_state_for_next_window, top_k_params_for_oos, best_sharpe, best_is_equity)
        best_is_equity: 最优候选在训练集(IS)上的净值曲线，供 OOS 衰减校验使用。
    """
    import pandas as pd  # type: ignore[import]

    # P4-Fix: 注入被固定的低敏感参数（space.py 已排除出搜索空间，
    # 但 backtest 引擎仍需要这些字段）
    _fixed_params = _get_fixed_params()

    signal_sp, portfolio_sp = split_by_cost(spaces)
    signal_names = list(signal_sp.keys())
    portfolio_names = list(portfolio_sp.keys())

    n_signal = len(signal_sp)
    n_total = len(spaces)

    controller = FidelityController(kline_df, engine_cfg, compute_exit_strategy, vectorized=True, eval_start_date=eval_start_date, st_history=st_history, exclude_st=exclude_st, data_version=data_version, listing_days=listing_days, db_engine=db_engine)
    _opt_t0 = time.time()

    best_sharpe_local = -1e10
    best_params_local: dict[str, float] = {}
    best_equity_local: Any = None
    X_hist: list[np.ndarray] = []
    Y_hist: list[float] = []          # GP 训练目标（DSR）
    Sharpe_hist: list[float] = []    # 审计修复：原始 Sharpe 用于 Top-K 排序
    params_hist: list[dict[str, float]] = []

    # ── 评估去重：同一参数组合只算一次 ──
    # GP 在重复点（Phase1 信号坐标恒为默认值、Phase2 组合坐标冻结在 best）下
    # 表面平坦，EI 会反复提议已评估点；这里全局兜底 + 连续重复转随机探索。
    _seen: dict[tuple[tuple[str, float], ...], dict[str, Any]] = {}
    _consecutive_skips = 0

    # ── DSR 惩罚（P4-Fix）：用 deflated_sharpe_ratio 替代原始 Sharpe 作为 GP 目标 ──
    # 随评估次数增加，DSR 惩罚加大，自动抑制过度搜索同一区域。
    # 原始 Sharpe 仍保留用于 best_sharpe_local 比较（业务口径）。
    # 审计修复：n_obs 应为唯一交易日数，非多股票面板总行数。
    # 原 len(kline_df) 可能为 N_stocks × N_days（数万），导致 sigma_sr 严重失真。
    _n_days_train = int(kline_df["trade_date"].dropna().unique().__len__())
    _eval_count = 0

    from BackTrading.overfitting import deflated_sharpe_ratio as _dsr_calc

    def _dsr_of(sharpe: float) -> float:
        """计算 deflated sharpe：评估次数越多，惩罚越重，抑制网格搜索式遍历。"""
        _eval_count_ref = max(_eval_count, 1)
        return _dsr_calc(sharpe, _n_days_train, _eval_count_ref)

    def _params_key(p: dict[str, float]) -> tuple[tuple[str, float], ...]:
        return tuple(sorted((k, round(float(v), 8)) for k, v in p.items()))

    def _eval_once(params: dict[str, float], fidelity: int) -> dict[str, Any]:
        # P4-Fix: 注入被固定的低敏感参数（space.py 已排除出搜索空间，
        # 但 backtest 引擎仍需要这些字段）。固定参数优先级低于搜索参数。
        merged = {**_fixed_params, **params}
        key = _params_key(merged)
        hit = _seen.get(key)
        if hit is not None:
            logger.debug(f"  [去重] 参数已评估过，复用结果: sharpe={hit['sharpe']:.4f}")
            return hit
        result = controller.evaluate(merged, fidelity=fidelity)
        _seen[key] = result
        return result

    def _track(sharpe: float, params: dict[str, float], x: np.ndarray, equity: Any = None) -> None:
        nonlocal best_sharpe_local, best_params_local, best_equity_local, _eval_count
        _eval_count += 1
        # DSR 惩罚：随评估次数增加，deflated sharpe 越来越低，
        # 引导 GP 探索新区域而非过度搜索已知峰值。
        dsr_value = _dsr_of(sharpe)
        X_hist.append(x)
        Y_hist.append(dsr_value)  # GP 用 DSR 值训练，非原始 Sharpe
        Sharpe_hist.append(sharpe)  # 审计修复：同步记录原始 Sharpe，供 Top-K 按真实值排序
        params_hist.append(params.copy())
        # raw Sharpe 仍用于业务口径 best 比较（日志和 OOS 用原始值）
        if sharpe > best_sharpe_local:
            best_sharpe_local = sharpe
            best_params_local = params.copy()
            best_equity_local = list(equity) if equity else None

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Sobol + 预热缓存（组合空间 × Level2，信号用默认参数）
    # ═══════════════════════════════════════════════════════════
    n_init = n_init_signal if n_total > 0 else 3
    n_portfolio = len(portfolio_sp)

    # 预热信号缓存：用信号参数中点值，确保 Phase 1 的 Level 2 始终命中
    default_signal = {k: (sp.low + sp.high) / 2.0 for k, sp in signal_sp.items()}
    if default_signal:
        controller.warm_cache(default_signal)

    if n_portfolio > 0:
        logger.info(f"[Phase 1/4] Sobol init: {n_init} 组 (组合空间 Level2, 信号=默认)")
        sobol_x = _sobol_samples(n_init, n_portfolio, seed=seed)
        _consecutive_skips = 0
        for i in range(n_init):
            port_params = _from_normalized(sobol_x[i], portfolio_sp)
            params = {**default_signal, **port_params}
            x_norm = _to_normalized(params, spaces)
            result = _eval_once(params, fidelity=0)
            sharpe = result["sharpe"]
            _track(sharpe, params, x_norm, result.get("equity"))
            if progress_cb:
                progress_cb(1, i, n_init, sharpe)
            logger.debug(f"  Sobol[{i}] sharpe={sharpe:.4f}")
    else:
        logger.info(f"[Phase 1/4] Sobol init: {n_init} 组 (全空间 Level2)")
        sobol_x = _sobol_samples(n_init, n_total, seed=seed)
        _consecutive_skips = 0
        for i in range(n_init):
            params = _from_normalized(sobol_x[i], spaces)
            x_norm = sobol_x[i].reshape(1, -1)
            result = _eval_once(params, fidelity=0)
            sharpe = result["sharpe"]
            _track(sharpe, params, x_norm.ravel(), result.get("equity"))
            if progress_cb:
                progress_cb(1, i, n_init, sharpe)
            logger.debug(f"  Sobol[{i}] sharpe={sharpe:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: 贝叶斯 Level 1（全空间 GP + 组合参数冻结，仅优化信号维度）
    # ═══════════════════════════════════════════════════════════
    if n_signal > 0 and n_iter_signal > 0:
        logger.info(f"[Phase 2/4] Bayes signal: {n_iter_signal} 轮 (全空间GP)")
        X_all, Y_arr = _unique_x_agg(np.array(X_hist), np.array(Y_hist))
        # 信号 + 组合参数边界
        signal_bounds = np.array([[0.0, 1.0]] * n_signal)
        _consecutive_skips = 0

        for i in range(n_iter_signal):
            gp = build_gp(
                X_all, Y_arr,
                previous_state=restore_gp_state(previous_gp_state, n_total),
                n_restarts=5,
            )
            best_f = float(max(Y_hist))

            # 当前最佳组合参数（归一化）
            best_port_x = _to_normalized(
                {k: best_params_local.get(k, (sp.low + sp.high) / 2.0)
                 for k, sp in portfolio_sp.items()},
                portfolio_sp,
            )

            # 多起点 L-BFGS-B 优化混合采集函数（仅信号维度）
            best_acq = -1e10
            best_sig_x = None
            rng = np.random.RandomState(seed + i)
            for restart in range(10):
                x0 = rng.uniform(0, 1, n_signal)
                res = minimize(
                    lambda xs: -float(mixed_acquisition(
                        np.concatenate([xs, best_port_x]).reshape(1, -1),
                        gp, best_f, xi=0.01, dsr_lambda=0.05,
                    )[0]),
                    x0, method="L-BFGS-B", bounds=signal_bounds,
                    options={"maxiter": 50, "ftol": 1e-10},
                )
                if res.fun < -best_acq + 1e-10:
                    best_acq = -res.fun
                    best_sig_x = res.x

            sig_params = _from_normalized(best_sig_x, signal_sp)
            best_port_params = {
                k: best_params_local.get(k, (sp.low + sp.high) / 2.0)
                for k, sp in portfolio_sp.items()
            }
            params = {**sig_params, **best_port_params}
            # ── 去重：EI 在退化 GP 上会反复提议已评估点，命中即跳过 ──
            if _params_key(params) in _seen:
                _consecutive_skips += 1
                if _consecutive_skips >= 3:
                    logger.info(
                        f"  Bayes-Sig[{i}] 连续 {_consecutive_skips} 次重复提议，转随机探索"
                    )
                    rng_fb = np.random.RandomState(seed + 9000 + i)
                    for _attempt in range(200):
                        sig_params = _from_normalized(rng_fb.uniform(0, 1, n_signal), signal_sp)
                        params = {**sig_params, **best_port_params}
                        if _params_key(params) not in _seen:
                            break
                    _consecutive_skips = 0
                else:
                    logger.info(f"  Bayes-Sig[{i}] 重复提议已评估参数，跳过评估（第 {_consecutive_skips} 次连续）")
                    if progress_cb:
                        progress_cb(2, i, n_iter_signal, _seen[_params_key(params)]["sharpe"])
                    continue
            else:
                _consecutive_skips = 0
            result = _eval_once(params, fidelity=1)
            sharpe = result["sharpe"]
            x_norm = _to_normalized(params, spaces)
            _track(sharpe, params, x_norm, result.get("equity"))

            X_all, Y_arr = _unique_x_agg(np.array(X_hist), np.array(Y_hist))

            if progress_cb:
                progress_cb(2, i, n_iter_signal, sharpe)
            logger.debug(f"  Bayes-Sig[{i}/{n_iter_signal}] sharpe={sharpe:.4f} best={best_sharpe_local:.4f}")

    _p2_end = time.time()
    logger.info(f"[Phase 1+2] 完成: best={best_sharpe_local:.4f}, 耗时={_p2_end-_opt_t0:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: 固定信号最优值，优化组合参数
    # ═══════════════════════════════════════════════════════════
    # 取出最优的信号参数
    if n_signal > 0:
        best_signal_params = {
            k: best_params_local[k] for k in signal_names
        }
    else:
        best_signal_params = {}
    # 预热信号缓存（Level 2 需要）
    if best_signal_params:
        controller.warm_cache(best_signal_params)

    logger.info(f"[Phase 3/4] Bayes portfolio: {n_iter_portfolio} 轮")

    if n_init_portfolio > 0 and len(portfolio_sp) > 0:
        sobol_port = _sobol_samples(n_init_portfolio, len(portfolio_sp), seed=seed + 999)
        _consecutive_skips = 0
        for i in range(n_init_portfolio):
            port_params = _from_normalized(sobol_port[i], portfolio_sp)
            params = {**best_signal_params, **port_params}
            x_norm = _to_normalized(params, spaces)
            result = _eval_once(params, fidelity=0)
            sharpe = result["sharpe"]
            _track(sharpe, params, x_norm, result.get("equity"))
            if progress_cb:
                progress_cb(3, i, n_init_portfolio + n_iter_portfolio, sharpe)

    if len(portfolio_sp) > 0:
        _consecutive_skips = 0
        for i in range(n_iter_portfolio):
            # 构建 GP 只用于组合参数（信号参数已固定）
            X_port, Y_arr = _unique_x_agg(
                np.array([x[n_signal:] for x in X_hist]), np.array(Y_hist)
            )

            if len(X_port) < 3:
                logger.debug(f"  组合参数数据点 < 3，跳过 BO 迭代")
                continue

            gp_port = build_gp(
                X_port, Y_arr,
                previous_state=restore_gp_state(previous_gp_state, len(portfolio_sp)),
                n_restarts=5,
            )
            port_bounds = np.array([[0.0, 1.0]] * len(portfolio_sp))
            best_f = float(max(Y_hist))

            x_cand, _ = optimize_acquisition(
                gp_port, port_bounds, best_f,
                n_restarts=10, xi=0.01, dsr_lambda=0.05,
                random_state=seed + i + 1000,
            )
            port_params = _from_normalized(x_cand, portfolio_sp)
            params = {**best_signal_params, **port_params}
            # ── 去重：组合参数被冻结/退化时 EI 同样会反复提议同一点 ──
            if _params_key(params) in _seen:
                _consecutive_skips += 1
                if _consecutive_skips >= 3:
                    logger.info(
                        f"  Bayes-Port[{i}] 连续 {_consecutive_skips} 次重复提议，转随机探索"
                    )
                    rng_fb = np.random.RandomState(seed + 9500 + i)
                    for _attempt in range(200):
                        port_params = _from_normalized(
                            rng_fb.uniform(0, 1, len(portfolio_sp)), portfolio_sp
                        )
                        params = {**best_signal_params, **port_params}
                        if _params_key(params) not in _seen:
                            break
                    _consecutive_skips = 0
                else:
                    logger.info(f"  Bayes-Port[{i}] 重复提议已评估参数，跳过评估（第 {_consecutive_skips} 次连续）")
                    if progress_cb:
                        progress_cb(3, n_init_portfolio + i, n_init_portfolio + n_iter_portfolio, _seen[_params_key(params)]["sharpe"])
                    continue
            else:
                _consecutive_skips = 0
            result = _eval_once(params, fidelity=0)
            sharpe = result["sharpe"]
            x_norm = _to_normalized(params, spaces)
            _track(sharpe, params, x_norm, result.get("equity"))
            if progress_cb:
                progress_cb(3, n_init_portfolio + i, n_init_portfolio + n_iter_portfolio, sharpe)
            logger.debug(f"  Bayes-Port[{i}/{n_iter_portfolio}] sharpe={sharpe:.4f} best={best_sharpe_local:.4f}")

    _p3_end = time.time()
    logger.info(f"[Phase 3] 完成: best={best_sharpe_local:.4f}, 耗时={_p3_end-_opt_t0:.1f}s")

    # ═══════════════════════════════════════════════════════════
    # Phase 4: 局部精细化（代理模型上爬山，然后真实评估 top-3）
    # ═══════════════════════════════════════════════════════════
    if n_signal > 0 and len(X_hist) >= 3:
        logger.info(f"[Phase 4/4] Local refinement: top-{n_refine_top}")
        X_all_arr, Y_all_arr = _unique_x_agg(
            np.array([x[:n_signal] for x in X_hist]), np.array(Y_hist)
        )

        gp_refine = build_gp(X_all_arr, Y_all_arr, n_restarts=3)
        top_idx = np.argsort(Y_all_arr)[-n_refine_top:]

        # 取当前最优组合参数
        best_full_x = np.array([x for x in X_hist])[Y_hist.index(max(Y_hist))]
        best_port_params = _from_normalized(best_full_x[n_signal:], portfolio_sp)

        for idx in top_idx:
            refined = _local_refine(X_all_arr[idx], gp_refine, np.array([[0.0, 1.0]] * n_signal))
            sig_params = _from_normalized(refined, signal_sp)
            params = {**sig_params, **best_port_params}
            result = _eval_once(params, fidelity=1)
            sharpe = result["sharpe"]
            x_norm = _to_normalized(params, spaces)
            _track(sharpe, params, x_norm, result.get("equity"))
            logger.debug(f"  Refine sharpe={sharpe:.4f}")

    # ── 最终结果 ──
    best_params = best_params_local
    best_sharpe = best_sharpe_local

    # 提取 GP 状态用于 warm-start：信号 / 组合 / 全空间三个子空间分别保存
    # （对应 Phase 2 全空间、Phase 3 组合、Phase 4 信号 GP 的维度），
    # restore_gp_state 按维度取用，避免维度不匹配导致 warm-start 永远失效。
    gp_state = None
    if len(X_hist) > 0:
        try:
            X_all_final, Y_all_final = _unique_x_agg(
                np.array([x for x in X_hist]), np.array(Y_hist)
            )
            sub_states: dict[int, GPState] = {}
            if n_signal > 0:
                sub_states[n_signal] = save_gp_state(
                    build_gp(X_all_final[:, :n_signal], Y_all_final, n_restarts=3),
                    X_all_final[:, :n_signal], Y_all_final,
                )
            if n_portfolio > 0:
                sub_states[n_portfolio] = save_gp_state(
                    build_gp(X_all_final[:, n_signal:], Y_all_final, n_restarts=3),
                    X_all_final[:, n_signal:], Y_all_final,
                )
            if n_total > 0:
                sub_states[n_total] = save_gp_state(
                    build_gp(X_all_final, Y_all_final, n_restarts=3),
                    X_all_final, Y_all_final,
                )
            gp_state = {"sub_states": sub_states, "n_dims": n_signal}
        except Exception as exc:
            logger.warning(f"保存 GP 状态失败: {exc}")

    _opt_elapsed = time.time() - _opt_t0
    logger.info(f"  窗口优化完成: best_sharpe={best_sharpe:.4f}, 耗时={_opt_elapsed:.1f}s, params={best_params}")

    # ── 提取 top-K 参数用于 OOS 验证（PBO 需要多组 OOS 结果） ──
    # 审计修复：按原始 Sharpe 排序，非 DSR。DSR 随 eval_count 递增惩罚，
    # 后评估的高 Sharpe 参数会被压低，导致 Top-K 漏掉真正最优候选。
    n_top = min(n_refine_top, len(Sharpe_hist))
    if n_top >= 2:
        top_indices = np.argsort(Sharpe_hist)[-n_top:]
        top_k_params = [params_hist[i] for i in top_indices]
    else:
        top_k_params = [best_params]

    return best_params, gp_state, top_k_params, best_sharpe, best_equity_local
