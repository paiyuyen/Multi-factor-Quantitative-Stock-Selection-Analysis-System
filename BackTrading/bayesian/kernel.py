from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np
from loguru import logger
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Kernel,
    Matern,
    RBF,
    WhiteKernel,
)


def _default_kernel(n_dims: int, nu: float = 1.5) -> Kernel:
    """默认复合核: ConstantKernel × Matern(ARD) + WhiteKernel.

    Args:
        n_dims: 参数空间维度。
        nu: Matern 平滑度 (0.5=exponential, 1.5=infinitely diff-1, 2.5=infinitely diff-2)。

    Returns:
        可组合的 sklearn Kernel。
    """
    length_scale = np.ones(n_dims)
    # P4-Fix: 白噪音下限从 1e-6 提高到 1e-4，防止 GP 过度拟合观测值
    # 导致代理函数太"尖锐"、L-BFGS-B 找到虚假极值。
    return ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * Matern(
        length_scale=length_scale,
        length_scale_bounds=(1e-3, 1e3),
        nu=nu,
    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e-1))


# ── 序列化状态结构 ──
GPState = dict[str, Any]


def save_gp_state(gp: GaussianProcessRegressor, X: np.ndarray, Y: np.ndarray) -> GPState:
    """将 GPR 模型状态打包为可 pickle 的 dict。"""
    return {
        "kernel_params": gp.kernel_.get_params(),
        "kernel_str": gp.kernel_.__repr__(),  # 用于结构校验
        "X": X,
        "Y": Y,
        "alpha": gp.alpha_ if hasattr(gp, "alpha_") else None,
        "n_dims": X.shape[1],
    }


def restore_gp_state(state: GPState | None, n_dims: int) -> GPState | None:
    """校验并还原 GP 状态（支持多子空间嵌套状态，按维度匹配）。

    兼容两种结构：
      - 嵌套：{"sub_states": {n_signal: {...}, n_total: {...}, ...}}（optimize_window
        同时保存信号/全空间/组合三个子空间的 GP，Phase 2/3 按各自维度取用）
      - 扁平：{"n_dims": d, "kernel_params": ...}（旧格式，仍按 n_dims 精确匹配）
    """
    if state is None:
        return None
    sub = state.get("sub_states")
    if isinstance(sub, dict) and sub:
        inner = sub.get(n_dims)
        if inner is None:
            inner = sub.get(str(n_dims))
        if inner is not None:
            logger.debug(f"GP warm-start: 使用 {n_dims} 维子空间状态")
            return inner
        logger.warning(f"GP 状态无 {n_dims} 维子空间，忽略 warm-start")
        return None
    if state.get("n_dims") != n_dims:
        logger.warning(f"GP 状态维度 {state.get('n_dims')} 不匹配当前 {n_dims}，忽略 warm-start")
        return None
    return state


def build_gp(
    X: np.ndarray,
    Y: np.ndarray,
    previous_state: GPState | None = None,
    n_restarts: int = 5,
    normalize_y: bool = True,
    random_state: int = 42,
) -> GaussianProcessRegressor:
    """构建并拟合 GP 模型，支持 warm-start。

    Args:
        X: (n, d) 归一化参数。
        Y: (n,) 观测值（Sharpe）。
        previous_state: 前一窗口的 GPState（用于 warm-start 核超参）。
        n_restarts: 优化器的随机重启次数。
        normalize_y: 是否对 Y 做零均值标准化。
        random_state: 随机种子。

    Returns:
        拟合好的 GaussianProcessRegressor。
    """
    n_dims = X.shape[1]
    kernel = _default_kernel(n_dims)

    if previous_state is not None:
        try:
            kernel.set_params(**previous_state["kernel_params"])
            logger.debug("GP warm-start: 使用前窗口核超参作为初始值")
        except Exception as exc:
            logger.warning(f"GP warm-start 参数还原失败 ({exc})，使用默认初始化")

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=normalize_y,
        random_state=random_state,
        alpha=0.0,  # 噪声由 WhiteKernel 建模
    )
    gp.fit(X, Y)
    return gp
