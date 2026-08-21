from __future__ import annotations

import random
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from BackTrading.engine import EngineConfig, _run_single_backtest
from BackTrading.bayesian.kernel import GPState
from BackTrading.bayesian.optimizer import optimize_window
from BackTrading.bayesian.space import ParamSpace, build_spaces, split_by_cost
from BackTrading.domain.models import CostModel
from BackTrading.prepare import prepare_backtest_data
from BackTrading.snapshot import begin_snapshot_session, save_failure_snapshot, set_run_context


def _snap_log(snapshot_id: str | None) -> str:
    return f" | snapshot_id={snapshot_id}" if snapshot_id else ""


def _datetime64_to_string_guard(df: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
        df = df.copy()
        df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")
    elif df["trade_date"].dtype == object:
        # 已经是字符串，确保格式统一
        sample = str(df["trade_date"].iloc[0]) if len(df) > 0 else ""
        if "T" in sample or len(sample) > 10:
            df = df.copy()
            df["trade_date"] = df["trade_date"].apply(
                lambda x: str(x).split("T")[0][:10] if pd.notna(x) else x
            )
    return df


def _to_date_str(d) -> str:
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    d_str = str(d)
    return d_str.split("T")[0][:10] if "T" in d_str else d_str[:10]


# 信号上下文预热天数：MACD/ATR/MA 等指标至少需要 120 个交易日的前文，
# 训练/OOS 切片前额外补足这段历史，保证窗口首日起信号有效。
_SIGNAL_WARMUP_DAYS = 120

# OOS 连续失效提前终止阈值：单窗口优化约 2~3 小时（350 次评估），
# 连续 N 个窗口 OOS 全废（Sharpe≤0/无交易）说明策略在当前区间系统性泛化失效，
# 继续跑完剩余窗口只空耗算力且结果必然废弃，达到阈值立即兜底返回。
_MAX_CONSECUTIVE_OOS_FAILURES = 3


class WFOSystematicFailure(Exception):
    """WFO 系统性优化失败异常。

    当所有路径/窗口的 OOS 校验全部不通过时抛出，携带回退参数供
    runner 记录失败状态并中断流水线，避免静默回退导致反复重跑。

    Attributes:
        fallback_params: 参数空间中间值兜底参数。
        reason: 失败原因描述。
        failed_windows: 连续失败的窗口数。
    """
    def __init__(self, fallback_params: dict[str, float], reason: str, failed_windows: int = 0) -> None:
        self.fallback_params = fallback_params
        self.reason = reason
        self.failed_windows = failed_windows
        super().__init__(f"WFO 系统性失败: {reason}")


def _window_dates(
    unique_dates: list,
    train_period: int,
    test_period: int,
    offset: int = 0,
    purge_days: int = 0,
    embargo_days: int = 0,
    max_idx: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """生成 WFO 窗口切分（CPCV：净化 + 禁运）。

    purge_days:  训练窗口尾部剔除天数（标签视界），防止训练期最后的持仓
                 （标签 = 前向收益）跨入 OOS 窗口起点，造成 IS/OOS 泄露。
    embargo_days: 训练结束与测试开始之间的禁运间隔，切断自相关污染。
    max_idx:     WFO 寻参上界（exclusive）。末段 holdout 对 WFO 全程禁触时传入，
                 使 test_end 与循环终止均不越过此界；None 时回退用全部交易日。

    Returns:
        [(train_start, train_end, test_start, test_end), ...] 索引元组。
    """
    windows = []
    start = offset
    n = len(unique_dates) if max_idx is None else min(max_idx, len(unique_dates))
    step = test_period + embargo_days
    while start + train_period + step <= n:
        train_end = start + train_period - purge_days
        test_start = start + train_period + embargo_days
        test_end = min(test_start + test_period, n)
        if train_end > start:  # purge 后训练期须至少保留 1 天
            windows.append((start, train_end, test_start, test_end))
        start += step
    return windows


def _oos_validate(
    test_data: pd.DataFrame,
    params_list: list[dict[str, float]],
    engine_cfg: EngineConfig,
    top_m: int = 5,
    eval_start_date: str | None = None,
    st_history: dict | None = None,
    exclude_st: bool = True,
    data_version: str | None = None,
    listing_days: dict | None = None,
    db_engine: Any = None,
) -> list[dict[str, Any]]:
    """对 top-M 参数组合做 OOS 验证。

    eval_start_date: 信号预热历史已并入 test_data 时，用该日期截断，
    保证回测引擎只交易 [eval_start_date, 末尾] 的 OOS 区间。
    """
    from LogicAnalyzer.backtest_metrics import compute_risk_metrics, compute_trade_metrics

    _SIGNAL_KEYS = frozenset({
        "boll_narrow_ratio", "cross_decay_days",
        "golden_cross_bonus", "divergence_penalty", "conclusion_full_bull",
    })

    oos_results = []
    for rank_idx, params in enumerate(params_list[:top_m]):
        tl: list[dict[str, Any]] = []
        ec: list[dict[str, Any]] = []
        
        # 使用优化后的组合参数构建引擎配置（而非 base_cfg 默认值）
        from dataclasses import replace as _replace_dc
        _ec = _replace_dc(engine_cfg,
            atr_stop_mult=params.get("atr_stop_mult", engine_cfg.atr_stop_mult),
            buy_threshold=int(params.get("buy_threshold", engine_cfg.buy_threshold)),
            max_holdings=int(params.get("max_holdings", engine_cfg.max_holdings)),
        )

        signal_params = {k: v for k, v in params.items() if k in _SIGNAL_KEYS}
        if signal_params:
            _prepared = prepare_backtest_data(
                test_data, params=signal_params, compute_exit_strategy=True,
                vectorized=True, backtest_start_date=eval_start_date,
                data_version=data_version,
            )
        else:
            _prepared = test_data
            # 无信号参数时也按 eval_start_date 截断，避免预热段产生交易
            if eval_start_date and "trade_date" in _prepared.columns:
                _prepared = _prepared[_prepared["trade_date"] >= eval_start_date]
        engine_params = dict(params)
        if st_history:
            engine_params["_st_history"] = st_history
            engine_params["_exclude_st"] = exclude_st
        # P0-6 ④：上市日期显式注入（与 runner 最终回测口径一致）
        if listing_days:
            engine_params["_listing_days"] = listing_days
        # P1-4 行业映射：透传 db_engine 至引擎
        if db_engine is not None:
            engine_params["_db_engine"] = db_engine
        _run_single_backtest(_prepared, engine_params, _ec, tl, ec)
        risk = compute_risk_metrics(ec) or {}
        trade = compute_trade_metrics(tl) or {}
        sr = risk.get("sharpe_ratio")
        sr = sr if sr is not None and not (isinstance(sr, float) and np.isnan(sr)) else None
        oos_results.append({
            "params": params,
            "is_rank": rank_idx + 1,
            "oos_sharpe": sr,
            "oos_equity": ec,
            "total_return": risk.get("total_return", 0),
            "max_drawdown": risk.get("max_drawdown", 0),
            "annual_return": risk.get("annual_return", 0),
            "annual_vol": risk.get("annual_vol", 0),
            "var_95": risk.get("var_95", 0),
            "cvar_95": risk.get("cvar_95", 0),
            "win_rate": trade.get("win_rate", 0),
            "profit_factor": trade.get("profit_factor", 0),
            "total_trades": trade.get("total_trades", 0),
        })
    return oos_results


def bayesian_walk_forward_multi(
    kline_df: pd.DataFrame,
    param_grid: dict | None = None,    # 保留签名兼容，不再使用
    train_period: int = 120,
    test_period: int = 20,
    num_paths: int = 3,
    initial_cash: float = 1_000_000.0,
    spaces: dict[str, ParamSpace] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """贝叶斯 WFO 主入口 — 多路径 + 多窗口 + 跨窗口迁移学习。

    Args:
        kline_df: K 线数据。
        param_grid: 保留签名兼容，不再使用（由 spaces 替代）。
        train_period: IS 窗口大小（交易日）。
        test_period: OOS 窗口大小。
        num_paths: 多路径数。
        initial_cash: 初始资金。
        spaces: 预构建 ParamSpace（None 时自动从 config 解析）。
        **kwargs: 透传给 EngineConfig 的额外参数（commission, slippage 等），
            另支持: purge_days / embargo_days（CPCV 净化+禁运）、
            time_budget_seconds（总时间预算）、max_no_improve_windows
            （连续无改进早停阈值）。

    Returns:
        DataFrame，每行一个窗口，与旧 walk_forward 格式兼容。
    """
    from UtilsManager.ConfigParser import Config as _Config

    if spaces is None:
        full_cfg = _Config()
        bt_cfg = full_cfg.app_config.backtest
        po_cfg = getattr(full_cfg.app_config, "portfolio_optimizer", None)
        spaces = build_spaces(bt_cfg, portfolio_optimizer_config=po_cfg)

    # ── A2 失败快照：开启会话（重置计数 + 过期清理），注入 run/task 上下文 ──
    begin_snapshot_session()
    set_run_context(run_id=kwargs.get("run_id"), task_id=kwargs.get("task_id"))

    signal_sp, portfolio_sp = split_by_cost(spaces)

    # ── 构建基座 EngineConfig（挂载 CostModel：ADV 动态冲击成本 + 流动性分档） ──
    # 与 runner.py 最终回测路径同源：优先从 BacktestConfig 构建完整 CostModel
    # （含印花税日期分段 / 经手费 / 证管费 / 过户费 / 流动性分档冲击），
    # 保证寻优期间摩擦口径与终验完全一致；否则引擎回退固定费率，大单冲击成本
    # 被系统性低估（1.8 交易摩擦合规 FAIL）。
    # fallback：config 不可用时退回 kwargs 手动构造（单元测试路径）。
    _slip = kwargs.get("slippage", 0.001)
    _app_cfg = None
    try:
        _app_cfg = _Config().app_config
    except Exception:
        pass
    if _app_cfg is not None:
        _wfo_cost = CostModel.from_backtest_config(
            _app_cfg.backtest,
            trading_cost=_app_cfg.trading_cost,
        )
    else:
        _wfo_cost = CostModel(
            commission_rate=kwargs.get("commission", 0.0003),
            stamp_tax_rate=kwargs.get("stamp_tax", 0.0005),
            market_slippage=_slip,
            limit_slippage=_slip * 0.5,
            min_commission_per_trade=kwargs.get("min_commission", 5.0),
        )
    base_cfg = EngineConfig(
        initial_cash=initial_cash,
        commission_rate=kwargs.get("commission", 0.0003),
        stamp_tax_rate=kwargs.get("stamp_tax", 0.0005),
        slippage=_slip,
        max_position_pct=kwargs.get("max_position_pct", 0.1),
        portfolio_method=kwargs.get("portfolio_method", "score_weighted"),
        point_in_time=kwargs.get("point_in_time", True),
        execution_model=kwargs.get("execution_model", "next_open"),
        cost_model=_wfo_cost,
        # P1-5 max_order_pct 分档注入
        max_order_pct=kwargs.get("max_order_pct", 0.30),
        max_order_pct_high=kwargs.get("max_order_pct_high", 0.20),
        max_order_pct_low=kwargs.get("max_order_pct_low", 0.10),
    )

    # 从 config 读取 BO 预算
    try:
        bt_cfg = _Config().app_config.backtest
        n_init_signal = bt_cfg.BAYESIAN_N_INIT_SIGNAL
        n_iter_signal = bt_cfg.BAYESIAN_N_ITER_SIGNAL
        n_init_portfolio = bt_cfg.BAYESIAN_N_INIT_PORTFOLIO
        n_iter_portfolio = bt_cfg.BAYESIAN_N_ITER_PORTFOLIO
    except Exception:
        n_init_signal, n_iter_signal = 15, 35
        n_init_portfolio, n_iter_portfolio = 20, 150

    # ── P2.1 CPCV 净化+禁运（默认从 config 读取，可被 kwargs 覆盖） ──
    try:
        bt_cfg2 = _Config().app_config.backtest
        _def_purge = int(getattr(bt_cfg2, "BAYESIAN_CPCV_PURGE_DAYS", 0))
        _def_embargo = int(getattr(bt_cfg2, "BAYESIAN_CPCV_EMBARGO_DAYS", 0))
        _def_time_budget = float(getattr(bt_cfg2, "BAYESIAN_TIME_BUDGET_SECONDS", 8 * 3600))
        _def_no_improve = int(getattr(bt_cfg2, "BAYESIAN_MAX_NO_IMPROVE_WINDOWS", 0))
    except Exception:
        _def_purge, _def_embargo = 0, 0
        _def_time_budget, _def_no_improve = 8 * 3600, 0
    purge_days = int(kwargs.get("purge_days", _def_purge))
    embargo_days = int(kwargs.get("embargo_days", _def_embargo))
    time_budget_seconds = float(kwargs.get("time_budget_seconds", _def_time_budget))
    max_no_improve_windows = int(kwargs.get("max_no_improve_windows", _def_no_improve))
    holdout_days = int(kwargs.get("holdout_days", 0))
    data_version = kwargs.get("data_version")
    if purge_days or embargo_days:
        logger.info(
            f"  CPCV 净化/禁运: purge={purge_days}天, embargo={embargo_days}天"
            f"（训练尾部剔除标签视界，训练/OOS 间留自相关缓冲）"
        )
    if time_budget_seconds > 0:
        logger.info(f"  时间预算: {time_budget_seconds/3600:.1f}h（超时提前终止，防任务挂死）")
    if max_no_improve_windows > 0:
        logger.info(f"  连续无改进早停阈值: {max_no_improve_windows} 个窗口")

    _t_wfo_start = time.monotonic()
    _time_up = False

    # 窗口切分以"正式回测起点"为坐标轴；起点之前为信号预热历史（不参与交易）
    backtest_start_date = kwargs.get("backtest_start_date")
    all_dates_str = sorted({_to_date_str(d) for d in kline_df["trade_date"].unique()})
    if backtest_start_date:
        _cut = backtest_start_date[:10]
        unique_dates = [d for d in all_dates_str if d >= _cut]
    else:
        unique_dates = all_dates_str
    _trim_offset = len(all_dates_str) - len(unique_dates)
    n_dates = len(unique_dates)
    # 末段独立 holdout：WFO 寻参上界（holdout_days>0 时末段对 WFO 全程禁触，供终验独立使用）
    if holdout_days > 0:
        wfo_max_idx = max(0, n_dates - holdout_days)
        logger.info(
            f"  末段 holdout: {holdout_days} 天（占比 {holdout_days / n_dates:.0%}），"
            f"WFO 寻参上界 wfo_max_idx={wfo_max_idx}（末 {holdout_days} 天禁触）"
        )
    else:
        wfo_max_idx = None
    logger.info(f"  n_dates={n_dates}, train_period={train_period}, test_period={test_period}, required={train_period + test_period}")
    if n_dates < train_period + test_period:
        _msg = f"数据不足: {n_dates} 个交易日（正式回测起点后），需要至少 {train_period + test_period}"
        _sid = save_failure_snapshot(
            ohlcv=kline_df,
            window_name="entry",
            metric_name="window_setup",
            error_code="DATA_INSUFFICIENT",
            error_message=_msg,
        )
        logger.warning(f"WFO 入口失败快照已保存{_snap_log(_sid)}: {_msg}")
        raise ValueError(_msg)

    # ── 窗口预检：股票池级摘要（执行级拦截在 prepare/indicator_cache 内部） ──
    try:
        from BackTrading.precheck import precheck_summary as _pc_summary
        _pc_counts = _pc_summary(kline_df, {"mode": kwargs.get("precheck_mode")})
        if _pc_counts:
            logger.info(f"  Precheck 股票池摘要: {_pc_counts}（mode={kwargs.get('precheck_mode') or '配置'}，"
                        f"SKIP/NEED_FILL 由 prepare 阶段执行拦截或填充）")
    except Exception as _pce:
        logger.warning(f"  Precheck 摘要计算失败（不影响主流程）: {_pce}")
    show_progress = kwargs.get("show_progress", False)
    # ST/退市逐日动态剔除（runner 注入，寻优与最终回测口径一致）
    st_history = kwargs.get("st_history")
    exclude_st = bool(kwargs.get("exclude_st", True))
    # P0-6 ④：上市日期显式注入（与 runner 最终回测口径一致）
    listing_days = kwargs.get("listing_days")
    # P1-4 行业映射：透传 db_engine 至引擎
    db_engine = kwargs.get("db_engine")

    # ── 多路径收集 ──
    all_path_results: dict[int, list[dict[str, Any]]] = {}  # window_id → [path_results]
    _K_FOR_OOS = 10  # P1-6 OOS验证top-K参数数提升至≥10，供PBO计算（原3组合随机1/3违反概率）

    for path_idx in range(num_paths):
        # 确定性偏移：各路径互不重叠，避免 IS/OOS 数据泄露
        offset = path_idx * test_period
        _span = train_period + test_period + embargo_days  # purge 仅缩训练尾部，不占额外天数
        # WFO 寻参上界受 holdout 限制（末段禁触）；无 holdout 时回退 n_dates
        _wfo_cap = wfo_max_idx if wfo_max_idx is not None else n_dates
        max_offset = max(0, _wfo_cap - _span)
        if offset > max_offset:
            logger.warning(f"路径 {path_idx + 1} offset={offset} 超出数据范围 (max={max_offset})，跳过")
            continue

        windows = _window_dates(unique_dates, train_period, test_period, offset,
                                purge_days=purge_days, embargo_days=embargo_days,
                                max_idx=wfo_max_idx)
        logger.info(f"路径 {path_idx + 1}/{num_paths}: offset={offset}, 窗口数={len(windows)}")

        if not windows:
            logger.warning(f"路径 {path_idx + 1} 无有效窗口，检查 n_dates({n_dates}) >= train({train_period})+test({test_period})?")
            _sid = save_failure_snapshot(
                ohlcv=kline_df,
                window_name=f"path-{path_idx + 1}",
                metric_name="window_setup",
                error_code="WINDOW_SLICE_EMPTY",
                error_message=f"路径 {path_idx + 1} 无有效窗口: n_dates={n_dates}, "
                              f"train={train_period}, test={test_period}, offset={offset}",
            )
            logger.warning(f"路径 {path_idx + 1} 窗口切片为空，失败快照已保存{_snap_log(_sid)}")
            continue

        previous_gp_state: GPState | None = None
        _consecutive_oos_failures = 0
        _path_best_sharpe = float("-inf")
        _consecutive_no_improve = 0

        for win_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
            if _time_up:
                logger.warning(f"  [{path_idx + 1}-{win_idx}] 时间预算已耗尽，提前终止路径")
                break
            train_dates = unique_dates[tr_s:tr_e]
            test_dates = unique_dates[te_s:te_e]

            # ── datetime guard ──
            _df = _datetime64_to_string_guard(kline_df.copy())
            # 统一使用字符串格式进行匹配
            train_dates_str = [_to_date_str(d) for d in train_dates]
            test_dates_str = [_to_date_str(d) for d in test_dates]
            # ── 信号上下文预热：窗口起点前补 _SIGNAL_WARMUP_DAYS 天历史 ──
            # 指标（MACD/ATR/MA）与 ML 需要前文；预热行仅用于信号计算，
            # 通过 eval_start_date 在 prepare/引擎侧截断，不参与交易。
            # 关键：训练切片止于训练期末（tr_e），绝不包含 OOS 区间，
            # 否则训练期目标函数混入样本外收益，OOS/PBO/DSR 全部失真。
            ctx_s_full = _trim_offset + max(0, tr_s - _SIGNAL_WARMUP_DAYS)
            train_mask = _df["trade_date"].isin(all_dates_str[ctx_s_full:_trim_offset + tr_e])
            train_data = _df[train_mask]
            ctx_s_test = _trim_offset + max(0, te_s - _SIGNAL_WARMUP_DAYS)
            test_mask = _df["trade_date"].isin(all_dates_str[ctx_s_test:_trim_offset + te_e])
            test_data = _df[test_mask]
            train_eval_start = train_dates_str[0]
            test_eval_start = test_dates_str[0]

            if train_data.empty:
                logger.warning(f"  [{path_idx + 1}-{win_idx}] 训练数据为空(train_dates={train_dates[0]}~{train_dates[-1]}), 跳过")
                _sid = save_failure_snapshot(
                    ohlcv=kline_df,
                    window_name=f"{path_idx + 1}-{win_idx}",
                    window_start=_to_date_str(train_dates[0]),
                    window_end=_to_date_str(train_dates[-1]),
                    window_size=0,
                    metric_name="window_train",
                    error_code="WINDOW_TRAIN_EMPTY",
                    error_message=f"训练数据为空: 窗口 {_to_date_str(train_dates[0])}~{_to_date_str(train_dates[-1])}",
                )
                logger.warning(f"  [{path_idx + 1}-{win_idx}] 训练数据为空，失败快照已保存{_snap_log(_sid)}")
                continue

            if show_progress:
                logger.info(f"  [{path_idx + 1}-{win_idx}] 优化窗口: "
                           f"{_to_date_str(train_dates[0])}~{_to_date_str(train_dates[-1])}")
            logger.info(f"  [{path_idx + 1}-{win_idx}] 训练数据: {len(train_data)}行, {train_data['symbol'].nunique()}只股票")

            # ── 贝叶斯优化 IS ──
            try:
                best_params, gp_state, top_k_params, is_sharpe, is_equity = optimize_window(
                    kline_df=train_data,
                    engine_cfg=base_cfg,
                    spaces=spaces,
                    n_init_signal=n_init_signal,
                    n_iter_signal=n_iter_signal,
                    n_init_portfolio=n_init_portfolio,
                    n_iter_portfolio=n_iter_portfolio,
                    previous_gp_state=previous_gp_state,
                    seed=42 + path_idx * 100 + win_idx,
                    eval_start_date=train_eval_start,
                    st_history=st_history,
                    exclude_st=exclude_st,
                    data_version=data_version,
                    listing_days=listing_days,
                    db_engine=db_engine,
                )
            except Exception as opt_err:
                _sid = save_failure_snapshot(
                    ohlcv=train_data,
                    window_name=f"{path_idx + 1}-{win_idx}",
                    window_start=_to_date_str(train_dates[0]),
                    window_end=_to_date_str(train_dates[-1]),
                    window_size=len(train_data),
                    metric_name="window_optimize",
                    error_code="WINDOW_OPTIMIZE_FAILED",
                    error_message=str(opt_err),
                    traceback_text=traceback.format_exc(),
                )
                logger.opt(exception=True).warning(
                    f"  [{path_idx + 1}-{win_idx}] 窗口优化失败: {opt_err}{_snap_log(_sid)}"
                )
                continue

            # ── 跨窗口迁移 ──
            # 传递嵌套子空间状态（信号/组合/全空间），
            # optimize_window 内部按 Phase 维度（n_signal/n_portfolio/n_total）分别取用，
            # 不可在此解包为单一维度，否则后续窗口维度不匹配、warm-start 永远失效。
            previous_gp_state = gp_state

            # ── OOS 验证（top-K 参数，PBO 需要多组 OOS 结果） ──
            if not test_dates:
                logger.warning(f"  [{path_idx + 1}-{win_idx}] OOS 区间为空, 跳过 OOS 验证")
                continue

            oos_params = top_k_params[:min(_K_FOR_OOS, len(top_k_params))]
            try:
                oos_results = _oos_validate(
                    test_data,
                    oos_params,
                    base_cfg,
                    top_m=len(oos_params),
                    eval_start_date=test_eval_start,
                    st_history=st_history,
                    exclude_st=exclude_st,
                    data_version=data_version,
                    listing_days=listing_days,
                    db_engine=db_engine,
                )
            except Exception as oos_err:
                _sid = save_failure_snapshot(
                    ohlcv=test_data,
                    window_name=f"{path_idx + 1}-{win_idx}",
                    window_start=_to_date_str(test_dates[0]),
                    window_end=_to_date_str(test_dates[-1]),
                    window_size=len(test_data),
                    metric_name="window_oos",
                    error_code="WINDOW_OOS_FAILED",
                    error_message=str(oos_err),
                    traceback_text=traceback.format_exc(),
                )
                logger.opt(exception=True).warning(
                    f"  [{path_idx + 1}-{win_idx}] OOS 验证失败: {oos_err}{_snap_log(_sid)}"
                )
                continue

            # ── 计算 OOS 指标 ──
            oos = oos_results[0] if oos_results else {}
            oos_sharpe = oos.get("oos_sharpe", 0) or 0

            # ── DM 检验（P2.2）：rank-1 参数 vs 基准中位数参数，OOS 显著性 ──
            # 若寻优参数相对基准无显著优势（p ≥ 0.05），说明"最优"只是噪声尖峰，
            # 该窗口不参与最终稳健中位数主路径（由 runner._extract_best_params 过滤）。
            _dm_stat: float | None = None
            _dm_p_value: float | None = None
            _dm_pass: bool = True
            try:
                from BackTrading.overfitting import compute_dm_test as _dm_test

                _mid_params = {n: (sp.low + sp.high) / 2 for n, sp in spaces.items()}
                _base_results = _oos_validate(
                    test_data, [_mid_params], base_cfg, top_m=1,
                    eval_start_date=test_eval_start,
                    st_history=st_history, exclude_st=exclude_st,
                    data_version=data_version,
                    listing_days=listing_days,
                    db_engine=db_engine,
                )
                _rank1_ec = oos_results[0].get("oos_equity", []) if oos_results else []
                _base_ec = _base_results[0].get("oos_equity", []) if _base_results else []
                if len(_rank1_ec) >= 10 and len(_base_ec) >= 10:
                    _ra = np.diff(np.array([e.get("portfolio_value", 0) for e in _rank1_ec], dtype=float)) / \
                          np.maximum(np.array([e.get("portfolio_value", 0) for e in _rank1_ec[:-1]], dtype=float), 1e-9)
                    _rb = np.diff(np.array([e.get("portfolio_value", 0) for e in _base_ec], dtype=float)) / \
                          np.maximum(np.array([e.get("portfolio_value", 0) for e in _base_ec[:-1]], dtype=float), 1e-9)
                    _dm_stat, _dm_p_value = _dm_test(_ra, _rb)
                    _dm_pass = bool(_dm_p_value < 0.05 and _dm_stat > 0)
                    if not _dm_pass:
                        logger.warning(
                            f"  [{path_idx + 1}-{win_idx}] DM检验: rank-1 相对基准中位数无显著优势"
                            f"（stat={_dm_stat:.2f}, p={_dm_p_value:.3f}），窗口不参与稳健中位数主路径"
                        )
            except Exception as _dme:
                logger.warning(f"  [{path_idx + 1}-{win_idx}] DM 检验异常: {_dme}，跳过")

            # ── 时间预算（P2.4）：窗口粒度检查，超时提前终止 ──
            if time_budget_seconds > 0 and (time.monotonic() - _t_wfo_start) > time_budget_seconds:
                logger.critical(
                    f"时间预算 {time_budget_seconds/3600:.1f}h 已耗尽，"
                    f"提前终止 WFO（已收集 {len(all_path_results)} 组窗口结果）"
                )
                _time_up = True

            # ── OOS 衰减 gate（业务规则：IS→OOS 风险调整收益衰减 > 30% 即废弃） ──
            # IS 净值曲线 = 优化器最优候选在训练集上的回测曲线（严格不含 OOS）；
            # OOS 净值曲线 = rank-1 参数在独立测试集上的回测曲线。
            _decay_pass = True
            _decay_report = None
            if is_equity is not None and oos_results:
                _oos_curve = oos_results[0].get("oos_equity")
                if _oos_curve:
                    try:
                        from BackTrading.overfitting import validate_oos_decay as _vd_oos
                        _decay_report = _vd_oos(
                            is_equity, _oos_curve,
                            is_days=len(train_dates), oos_days=len(test_dates),
                        )
                        _decay_pass = _decay_report.passed
                    except Exception as _de:
                        logger.warning(f"  [{path_idx + 1}-{win_idx}] OOS 衰减校验异常: {_de}，窗口照常保留")
            if not _decay_pass:
                _oos_trades = oos.get("total_trades", 0) or 0
                if _oos_trades == 0:
                    logger.warning(
                        f"  [{path_idx + 1}-{win_idx}] OOS 区间 {len(test_dates)} 天无任何交易，"
                        f"信号未触发（非过拟合），窗口结果无效"
                    )
                logger.warning("=" * 64)
                logger.warning(
                    f"  [{path_idx + 1}-{win_idx}] OOS 衰减校验未通过"
                    f"（IS_Sharpe={float(is_sharpe):.2f} → OOS_Sharpe={oos_sharpe:.2f}），"
                    f"疑似超参数过度网格搜索或特征工程隐性泄露，该窗口结果直接废弃"
                )
                logger.warning("=" * 64)
                _consecutive_oos_failures += 1
                if _consecutive_oos_failures >= _MAX_CONSECUTIVE_OOS_FAILURES:
                    logger.critical(
                        f"连续 {_consecutive_oos_failures} 个窗口 OOS 全部失效"
                        f"（单窗口耗时 ~2-3h），策略在当前数据区间系统性泛化失败，"
                        f"提前终止 WFO"
                    )
                    mid = {n: (sp.low + sp.high) / 2 for n, sp in spaces.items()}
                    raise WFOSystematicFailure(
                        fallback_params=mid,
                        reason=f"连续 {_consecutive_oos_failures} 个窗口 OOS 衰减校验全部失败；"
                               f"IS_Sharpe={float(is_sharpe):.2f} → OOS_Sharpe={oos_sharpe:.2f}；"
                               f"策略在当前数据区间无泛化能力",
                        failed_windows=_consecutive_oos_failures,
                    )
                continue

            # ── 收集 IS Sharpe（优化器返回的真实样本内绩效） ──
            train_sharpe = float(is_sharpe)

            entry: dict[str, Any] = {
                "window": win_idx,
                "train_start": _to_date_str(train_dates[0]),
                "train_end": _to_date_str(train_dates[-1]),
                "test_start": _to_date_str(test_dates[0]) if test_dates else "",
                "test_end": _to_date_str(test_dates[-1]) if test_dates else "",
                "params": best_params,
                "train_sharpe": train_sharpe,
                "sharpe_ratio": oos_sharpe,
                "total_return": oos.get("total_return", 0),
                "max_drawdown": oos.get("max_drawdown", 0),
                "annual_return": oos.get("annual_return", 0),
                "annual_vol": oos.get("annual_vol", 0),
                "var_95": oos.get("var_95", 0),
                "cvar_95": oos.get("cvar_95", 0),
                "win_rate": oos.get("win_rate", 0),
                "profit_factor": oos.get("profit_factor", 0),
                "total_trades": oos.get("total_trades", 0),
                "num_combos": n_init_signal + n_iter_signal + n_init_portfolio + n_iter_portfolio + 3,  # +3=P1-6 Phase4 refine_top 默认值，纳入 DSR num_trials
                "oos_combos": oos_results,
            }
            if _dm_stat is not None:
                entry["dm_stat"] = round(float(_dm_stat), 4)
                entry["dm_p_value"] = round(float(_dm_p_value), 4)
                entry["dm_pass"] = bool(_dm_pass)
            if _decay_report is not None:
                entry["sharpe_decay"] = round(_decay_report.sharpe_decay, 4)
                entry["sortino_decay"] = round(_decay_report.sortino_decay, 4)
            all_path_results.setdefault(win_idx, []).append(entry)
            _consecutive_oos_failures = 0

            # ── 连续无改进早停（P2.4） ──
            if max_no_improve_windows > 0:
                if oos_sharpe > _path_best_sharpe:
                    _path_best_sharpe = oos_sharpe
                    _consecutive_no_improve = 0
                else:
                    _consecutive_no_improve += 1
                    if _consecutive_no_improve >= max_no_improve_windows:
                        logger.critical(
                            f"  [{path_idx + 1}] 连续 {_consecutive_no_improve} 个窗口无 OOS 改进"
                            f"（best={_path_best_sharpe:.2f}），提前终止本路径"
                        )
                        break

    # ── 按窗口聚合（多路径取中位数） ──
    if not all_path_results:
        logger.critical("所有路径均无有效窗口通过 OOS 校验，策略系统性泛化失败")
        mid = {n: (sp.low + sp.high) / 2 for n, sp in spaces.items()}
        # 返回兜底行（而非抛异常）：runner.py 仍可通过 runner_result["wfo_rows"] 的空值
        # 判定失败；直接调用方也能拿到非空 DataFrame
        logger.warning(
            f"使用参数空间中间值作为兜底: {mid} — 调用方需检查 Sharpe/交易数决定是否采纳"
        )
        _fallback = dict(mid)
        _fallback["sharpe_ratio"] = 0.0
        _fallback["train_sharpe"] = 0.0
        _fallback["win_rate"] = 0.0
        _fallback["total_trades"] = 0
        _fallback["num_paths"] = 0
        result = pd.DataFrame([_fallback])
        logger.info(f"贝叶斯 WFO 完成: {len(result)} 个窗口, {num_paths} 条路径")
        return result

    agg_rows: list[dict[str, Any]] = []
    for win_id in sorted(all_path_results.keys()):
        entries = all_path_results[win_id]
        # 取第一个作为基础（结构相同）
        base = dict(entries[0])
        # 对 params 取中位数
        all_params = [e["params"] for e in entries if isinstance(e.get("params"), dict)]
        if len(all_params) > 1:
            keys = all_params[0].keys()
            median_params: dict[str, float] = {}
            for k in keys:
                vals = sorted(p[k] for p in all_params)
                median_params[k] = vals[len(vals) // 2]
            base["params"] = median_params
        # sharpe 取中位数
        sharpe_vals = sorted(e["sharpe_ratio"] for e in entries)
        base["sharpe_ratio"] = sharpe_vals[len(sharpe_vals) // 2] if sharpe_vals else 0.0
        # DM 聚合：多路径同窗口任一显著通过即可保留（保守与主路径取并集）
        _dm_flags = [e.get("dm_pass") for e in entries if e.get("dm_pass") is not None]
        if _dm_flags:
            base["dm_pass"] = any(_dm_flags)
            _dm_ps = [e.get("dm_p_value") for e in entries if e.get("dm_p_value") is not None]
            base["dm_p_value"] = float(np.median(_dm_ps))
        base["num_paths"] = len(entries)
        agg_rows.append(base)

    result = pd.DataFrame(agg_rows)
    logger.info(f"贝叶斯 WFO 完成: {len(result)} 个窗口, {num_paths} 条路径")

    # ── 聚合 IS/OOS 衰减 gate（跨窗口均值口径：OOS 均值相对 IS 均值衰减 > 30% 整体废弃） ──
    # 逐窗口 gate 已废弃超限窗口；此处兜底覆盖"单窗口衰减未超限、但整体泛化性崩坏"的情形。
    if len(agg_rows) > 0:
        _is_vals = [r["train_sharpe"] for r in agg_rows
                    if isinstance(r.get("train_sharpe"), (int, float))]
        _oos_vals = [r["sharpe_ratio"] for r in agg_rows
                     if isinstance(r.get("sharpe_ratio"), (int, float))]
        if _is_vals and _oos_vals:
            _agg_is = float(np.mean(_is_vals))
            _agg_oos = float(np.mean(_oos_vals))
            if _agg_is > 0 and _agg_oos > 0:
                _agg_decay = 1.0 - _agg_oos / _agg_is
                if _agg_decay > 0.30:
                    logger.critical(
                        f"[OOS衰减校验] 聚合衰减 {_agg_decay:.1%}（IS均值={_agg_is:.2f} → "
                        f"OOS均值={_agg_oos:.2f}）> 30%，判定超参数过度网格搜索或特征工程隐性泄露，"
                        f"本次寻优结果整体废弃"
                    )
                    mid = {n: (sp.low + sp.high) / 2 for n, sp in spaces.items()}
                    raise WFOSystematicFailure(
                        fallback_params=mid,
                        reason=f"聚合 IS/OOS 衰减 {_agg_decay:.1%}（IS均值={_agg_is:.2f} → "
                               f"OOS均值={_agg_oos:.2f}）超过 30% 阈值；"
                               f"疑似超参数过度网格搜索或特征工程隐性泄露",
                        failed_windows=len(agg_rows),
                    )
    return result
