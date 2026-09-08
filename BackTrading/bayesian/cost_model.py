from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from BackTrading.engine import (
    EngineConfig,
    _run_single_backtest,
)
from BackTrading.prepare import prepare_backtest_data, _compute_param_hash
from BackTrading.prepare import _data_fingerprint as _prepare_data_fingerprint

# 影响信号计算的参数名（必须与 prepare._compute_param_hash 一致）
_SIGNAL_PARAM_KEYS = frozenset({
    "boll_narrow_ratio",
    "cross_decay_days",
    "golden_cross_bonus",
    "divergence_penalty",
    "conclusion_full_bull",
})

# EngineConfig 中可被优化的字段（必须是 EngineConfig 的 dataclass 字段）
# 注：kelly_fraction / position_a / liq_veto_ratio / risk_none_multiplier
# 为引擎死参数（审计确认仓位恒等权），不纳入寻优。
_TUNABLE_CFG_FIELDS = frozenset({
    "atr_stop_mult", "boll_narrow_ratio", "cross_decay_days",
    "buy_threshold", "max_holdings",
})

# ── 全局信号缓存（跨路径跨窗口共享） ──
# key: (config_hash, data_fingerprint, param_hash) → DataFrame
# 注意：每个条目 ~0.7 GiB（1.8M 行 × 50 列），磁盘缓存已保底，
# 内存保留最近 _GLOBAL_CACHE_MAX 份：Phase1/2/3 交替评估的信号组
# （默认参数、best 信号、refine 候选）可同时驻留，避免来回踢缓存导致
# 同一参数组反复全量重算。4 份 ≈ 2.8GiB，若内存紧张可调回 2。
# #6b 审计修复：存入时 deep copy，防止外部线程修改缓存中的 DataFrame
_GLOBAL_SIGNAL_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
_GLOBAL_CACHE_MAX = 4
_GLOBAL_CACHE_LOCK = threading.Lock()


def _signal_hash(params: dict[str, Any]) -> str:
    """只取信号参数的子集做哈希。

    同时支持扁平 dict 和结构化 dict。
    """
    def _get(key: str, default: Any) -> Any:
        if key in params:
            return params[key]
        if key == "boll_narrow_ratio" and "regime" in params:
            return params["regime"].get(key, default)
        if key == "conclusion_full_bull" and "thresholds" in params:
            return params["thresholds"].get("fully_bull", default)
        if key in ("cross_decay_days", "golden_cross_bonus", "divergence_penalty") and "scoring" in params:
            return params["scoring"].get(key, default)
        return default

    sub = {
        k: _get(k, 0 if k == "boll_narrow_ratio" else (80 if k == "conclusion_full_bull" else 0))
        for k in _SIGNAL_PARAM_KEYS
        if k in params or _get(k, None) is not None
    }
    return hashlib.sha256(json.dumps(sub, sort_keys=True).encode()).hexdigest()[:8]


def _data_fingerprint(df: pd.DataFrame) -> str:
    """与 prepare._data_fingerprint 保持同一实现（含内容哈希），避免两处漂移。"""
    return _prepare_data_fingerprint(df)


def _make_eval_cfg(base: EngineConfig, params: dict[str, Any]) -> EngineConfig:
    """用 params 中的可调字段覆盖 base EngineConfig，生成新副本。"""
    overrides = {k: v for k, v in params.items() if k in _TUNABLE_CFG_FIELDS}
    if not overrides:
        return base
    return dataclasses.replace(base, **overrides)


class FidelityController:
    """两保真度评估控制器。

    Level 1 (昂贵):  compute_signals + 回测
    Level 2 (廉价):  复用缓存信号 + 仅回测

    若输入数据已包含信号列（进场评分/退出评分/风险等级/止损价），
    自动跳过 prepare_backtest_data，直接运行 _run_single_backtest。
    """

    _SIGNAL_COLS = {"进场评分", "退出评分", "风险等级", "止损价"}

    def __init__(
        self,
        kline_df: pd.DataFrame,
        base_engine_cfg: EngineConfig,
        compute_exit_strategy: bool = True,
        vectorized: bool = True,
        eval_start_date: str | None = None,
        st_history: dict | None = None,
        exclude_st: bool = False,  # FIX(P0): 回测默认不排除 ST（复盘单元通过 coordinator.py 显式控制）
        data_version: str | None = None,
        listing_days: dict | None = None,
        db_engine: Any = None,
    ):
        self._kline = kline_df
        self._base_cfg = base_engine_cfg
        self._compute_exit = compute_exit_strategy
        self._vectorized = vectorized
        # 信号预热历史与评估区间分离：传入时 prepare 计算完信号后截断至该日期，
        # 保证引擎只交易 [eval_start_date, 末尾]，预热行不产生交易。
        self._eval_start_date = eval_start_date
        # ST/退市逐日动态剔除（与 runner 最终回测口径一致，注入引擎 params）
        self._st_history = st_history
        self._exclude_st = exclude_st
        # P0-6 ④：上市日期显式注入（与 runner 最终回测口径一致）
        self._listing_days = listing_days
        # P3.1 数据版本（入 ML 冻结缓存 key 与信号缓存指纹，增量同步后整库失效）
        self._data_version = data_version
        # P1-4 行业映射：透传 db_engine 至引擎，启动时刷新 _industry_cache
        self._db_engine = db_engine
        # 实例级缓存（信号参数 hash → DataFrame），上限 1 防 OOM
        #（跨参数组合的命中率本来就低，多窗口下每份 ~0.7GiB 是 OOM 主因）
        self._signal_cache: dict[str, pd.DataFrame] = {}
        self._INSTANCE_CACHE_MAX = 1
        self._last_signal_hash: str | None = None
        # 自动检测数据是否已含信号列
        self._has_signals = self._SIGNAL_COLS.issubset(set(kline_df.columns))
        # 数据指纹（用于全局缓存键）
        self._data_key = _data_fingerprint(kline_df) if not self._has_signals else ""
        if data_version:
            self._data_key = f"{self._data_key or 'pre'}:{data_version}"
        # P4-Fix: 将 eval_start_date 纳入缓存 key，隔离不同 WFO 窗口的信号缓存。
        # 即使 K 线数据相同，不同窗口的 expanding 统计量起点不同（已改为 rolling(252)），
        # 仍应在缓存层面显式隔离，防微杜渐。
        if eval_start_date:
            self._data_key = f"{self._data_key}_eval_start={eval_start_date}"

    def _config_hash(self) -> str:
        """计算当前 BaseConfig 的哈希，用于全局缓存键。"""
        from BackTrading.prepare import _compute_config_hash
        return _compute_config_hash()

    def evaluate(
        self, params: dict[str, Any], fidelity: int = 1
    ) -> dict[str, Any]:
        """评估一组参数。

        Args:
            params: 完整参数 dict（含信号 + 组合参数）。
            fidelity: 1=Level 1(含信号计算), 0=Level 2(仅回测)。

        Returns:
            { "sharpe": float, "total_return": float, "cost": float,
              "elapsed": float, "equity": list, "trades": list }
        """
        t0 = time.perf_counter()

        # ── 信号准备 ──
        need_signal = False
        if self._has_signals:
            data = self._kline
        else:
            need_signal = fidelity == 1 or any(
                k in params for k in _SIGNAL_PARAM_KEYS
            )
            if need_signal:
                sig_hash = _signal_hash(params)
                # 优先查实例级缓存（最快）
                data = self._signal_cache.get(sig_hash)
                if data is None:
                    # 查全局缓存（跨路径共享）
                    global_key = (self._config_hash(), self._data_key, sig_hash)
                    data = _GLOBAL_SIGNAL_CACHE.get(global_key)
                    if data is not None:
                        with _GLOBAL_CACHE_LOCK:
                            _GLOBAL_SIGNAL_CACHE.move_to_end(global_key)
                    if data is None:
                        _t_p = time.perf_counter()
                        data = prepare_backtest_data(
                            self._kline, params=params,
                            compute_exit_strategy=self._compute_exit,
                            vectorized=self._vectorized,
                            backtest_start_date=self._eval_start_date,
                            data_version=self._data_version,
                        )
                        logger.debug(f"  [evaluate] 信号准备耗时 {time.perf_counter()-_t_p:.1f}s（data={len(data)} 行, hash={sig_hash}）")
                        with _GLOBAL_CACHE_LOCK:
                            if len(_GLOBAL_SIGNAL_CACHE) >= _GLOBAL_CACHE_MAX:
                                _GLOBAL_SIGNAL_CACHE.pop(next(iter(_GLOBAL_SIGNAL_CACHE)))
                            # #6b 审计修复：deep copy 防止外部修改缓存
                            _GLOBAL_SIGNAL_CACHE[global_key] = data.copy(deep=True)
                    self._signal_cache[sig_hash] = data
                    while len(self._signal_cache) > self._INSTANCE_CACHE_MAX:
                        self._signal_cache.pop(next(iter(self._signal_cache)))
                self._last_signal_hash = sig_hash
            else:
                if self._signal_cache:
                    data = next(iter(self._signal_cache.values()))
                else:
                    logger.warning("fidelity=0 但信号缓存为空，自动升级到 Level 1")
                    return self.evaluate(params, fidelity=1)

        # ── 回测 ──
        trade_log: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []
        eval_cfg = _make_eval_cfg(self._base_cfg, params)
        # 注入 ST/退市动态剔除数据（在信号哈希计算之后，不污染信号缓存键）
        engine_params = dict(params)
        if self._st_history:
            engine_params["_st_history"] = self._st_history
            engine_params["_exclude_st"] = self._exclude_st
        # P0-6 ④：上市日期显式注入（与 runner 最终回测口径一致）
        if self._listing_days:
            engine_params["_listing_days"] = self._listing_days
        # P1-4 行业映射：注入 db_engine_url 至引擎（FIX：用 URL 字符串替代 Engine 对象，
        # 阻断 Engine 混入 best_params → json.dumps 崩溃）
        if self._db_engine is not None:
            engine_params["_db_engine_url"] = str(self._db_engine.url)
        _t_b = time.perf_counter()
        total_return = _run_single_backtest(
            data, engine_params, eval_cfg, trade_log, equity_curve
        )
        logger.debug(f"  [evaluate] 回测耗时 {time.perf_counter()-_t_b:.1f}s, 交易 {len(trade_log)} 笔")

        # ── 计算 Sharpe ──
        try:
            from LogicAnalyzer.backtest_metrics import compute_risk_metrics
            risk = compute_risk_metrics(equity_curve) or {}
            sharpe = risk.get("sharpe_ratio", -1e10)
            if sharpe is None or (isinstance(sharpe, float) and np.isnan(sharpe)):
                sharpe = -1e10
        except Exception:
            sharpe = -1e10

        elapsed = time.perf_counter() - t0
        return {
            "sharpe": float(sharpe),
            "total_return": float(total_return) if total_return is not None else -1.0,
            "cost": 1.0 if need_signal else 0.1,
            "elapsed": elapsed,
            "equity": equity_curve,
            "trades": trade_log,
        }

    def warm_cache(self, signal_params: dict[str, Any]) -> str:
        """预热信号缓存（用于 Level 2 优化前固定信号参数）。"""
        if self._has_signals:
            return "_presignaled"
        sig_hash = _signal_hash(signal_params)
        # 检查实例级缓存
        if sig_hash not in self._signal_cache:
            # 检查全局缓存
            global_key = (self._config_hash(), self._data_key, sig_hash)
            data = _GLOBAL_SIGNAL_CACHE.get(global_key)
            if data is not None:
                with _GLOBAL_CACHE_LOCK:
                    _GLOBAL_SIGNAL_CACHE.move_to_end(global_key)
            if data is None:
                _t_p = time.perf_counter()
                data = prepare_backtest_data(
                    self._kline, params=signal_params,
                    compute_exit_strategy=self._compute_exit,
                    vectorized=self._vectorized,
                    backtest_start_date=self._eval_start_date,
                    data_version=self._data_version,
                )
                logger.debug(f"  [warm_cache] 信号准备耗时 {time.perf_counter()-_t_p:.1f}s（data={len(data)} 行, hash={sig_hash}）")
                with _GLOBAL_CACHE_LOCK:
                    if len(_GLOBAL_SIGNAL_CACHE) >= _GLOBAL_CACHE_MAX:
                        _GLOBAL_SIGNAL_CACHE.pop(next(iter(_GLOBAL_SIGNAL_CACHE)))
                    # #6b 审计修复：deep copy 防止外部修改缓存
                    _GLOBAL_SIGNAL_CACHE[global_key] = data.copy(deep=True)
            self._signal_cache[sig_hash] = data
            while len(self._signal_cache) > self._INSTANCE_CACHE_MAX:
                self._signal_cache.pop(next(iter(self._signal_cache)))
        self._last_signal_hash = sig_hash
        return sig_hash

    @property
    def cached_signal_hashes(self) -> list[str]:
        return list(self._signal_cache.keys())
