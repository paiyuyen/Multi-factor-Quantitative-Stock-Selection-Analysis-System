from __future__ import annotations

import math
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sqlalchemy import text

from BackTrading.alert import BacktestAlert
from LogicAnalyzer.backtest_metrics import compute_risk_metrics, compute_trade_metrics
from BackTrading.calibration import (
    CALIB_PARAM_MAP,
    CalibrationResult,
    apply_calibration_to_config,
    load_calibration,
    run_bayesian_walk_forward as run_walk_forward,
    save_calibration,
    write_calibration_to_ini,
)
from BackTrading.calibration_log import ensure_table, get_last_run, record_run, should_rerun
from UtilsManager.IDataProvider import BacktestDataProvider
from BackTrading.prepare import _build_params, merge_best_params_into_structured, prepare_backtest_data
from UtilsManager.ConfigParser import Config
from DataManager.DbEngine import get_engine


_BACKTEST_LOCK_KEY = 987654321
# 会话级 advisory lock 专用连接：回测全程持有，外部增量同步（IncrementalSyncEngine）
# 在写 K 线前探测同一 key，被占用即跳过，防止运行中数据被改写导致缓存内容漂移。
_RUN_LOCK_CONN: Any = None


def _acquire_lock(engine: Any) -> None:
    """获取回测分布式锁（pg_advisory_xact_lock + NOWAIT，失败则 exit）。"""
    from sqlalchemy import text as _t

    with engine.connect() as conn:
        locked = conn.execute(
            _t(f"SELECT pg_try_advisory_xact_lock({_BACKTEST_LOCK_KEY})")
        ).scalar()
        if locked:
            logger.info("  获取回测分布式锁成功")
        else:
            logger.error("回测分布式锁被占用，终止执行（可能有另一个进程正在运行）")
            sys.exit(1)  # P1-13：非零退出码，让调度/CI正确识别失败而非"已完成"

    # 会话级锁：在专用连接上持有整个回测期间（session-level，跨事务存活）。
    # 同一会话重复获取返回 True（回测自身的启动同步不受影响），
    # 外部进程探测同一 key 返回 False → 同步引擎跳过本次执行。
    global _RUN_LOCK_CONN
    _RUN_LOCK_CONN = engine.connect()
    try:
        held = _RUN_LOCK_CONN.execute(
            _t(f"SELECT pg_try_advisory_lock({_BACKTEST_LOCK_KEY})")
        ).scalar()
        if held:
            logger.info("  获取会话级数据隔离锁成功（外部数据同步将让路）")
        else:
            logger.warning("  会话级数据隔离锁被占用，仍继续执行")
    except Exception as exc:
        logger.warning(f"  会话级数据隔离锁获取失败: {exc}")
        _RUN_LOCK_CONN.close()
        _RUN_LOCK_CONN = None


def _release_run_lock() -> None:
    """释放会话级 advisory lock（关闭专用连接即自动释放）。"""
    global _RUN_LOCK_CONN
    if _RUN_LOCK_CONN is not None:
        try:
            _RUN_LOCK_CONN.close()
        except Exception:
            pass
        _RUN_LOCK_CONN = None


def _to_date(v: Any) -> date | None:
    """统一日期归一化：任意时间类型 → datetime.date。

    支持 str / pd.Timestamp / datetime.datetime / numpy.datetime64 / int 等。
    解析失败返回 None（调用方负责处理）。
    """
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return pd.Timestamp(v).date()
    except (ValueError, TypeError, OverflowError):
        return None


def _holdout_equity_slice(
    equity_curve: list[dict[str, Any]] | pd.DataFrame | None,
    final_prepared: pd.DataFrame,
    holdout_days: int,
) -> tuple[list[dict[str, Any]] | pd.DataFrame | None, str | None]:
    """P0-3 / P1.5：按交易日索引切出末段 holdout 净值曲线。

    P1.5 修复：PIT 过滤后个股交易日集不同步（新股上市/停牌复牌 → 末段样本量不足）。
    旧实现用 ``pd.unique(final_prepared["trade_date"])`` 取"并集" —— 任何股票
    出现在的日期都算交易日。新股在 holdout 期中途上市时，该标的 holdout 样本量
    被拉长，导致净值曲线权重失真。

    新逻辑：取有效交易日集合 = 至少有 ``min_coverage_pct × 总标的数`` 行数据的日期
    （默认 50%），排除"稀疏日"。日志输出 holdout 期有效标的分布与缺失天数。

    Returns:
        (切片后的净值曲线, holdout 起始交易日字符串)；数据不足/类型不支持时
        返回 (None, None)。
    """
    if holdout_days <= 0 or equity_curve is None:
        return None, None
    if not isinstance(equity_curve, (list, pd.DataFrame)):
        return None, None

    # P2.4 修复：净值曲线必需列名断言（防上游列名漂移导致下游 KeyError 静默崩溃）
    EQUITY_REQUIRED_COLS = {"time", "portfolio_value"}
    if isinstance(equity_curve, pd.DataFrame):
        actual_cols = set(equity_curve.columns)
        missing = EQUITY_REQUIRED_COLS - actual_cols
        if missing:
            logger.warning(
                f"[P2.4] 净值曲线缺少必需列 {missing}（实际列: {sorted(actual_cols)}），"
                f"holdout 切片跳过"
            )
            return None, None
    elif isinstance(equity_curve, list) and equity_curve:
        # list[dict] 分支检查首个 dict 的 keys
        actual_keys = set(equity_curve[0].keys())
        missing = EQUITY_REQUIRED_COLS - actual_keys
        if missing:
            logger.warning(
                f"[P2.4] 净值曲线 dict 缺少必需键 {missing}（实际键: {sorted(actual_keys)}），"
                f"holdout 切片跳过"
            )
            return None, None

    # ── P1.5 PIT 对齐：取有效交易日交集（≥50% 标的覆盖率） ──
    _all_dates = sorted(pd.unique(final_prepared["trade_date"]))
    if "symbol" in final_prepared.columns:
        _total_symbols = int(final_prepared["symbol"].nunique())
    else:
        _total_symbols = 0

    # P1.11 最小标的数守卫：holdout 窗口参与标的过少时告警
    if _total_symbols < 3:
        logger.warning(
            f"[Holdout] 仅{_total_symbols}只标的参与窗口，覆盖率极低——"
            f"holdout 评估结果可能不可信。"
        )

    _min_rows = max(1, int(_total_symbols * 0.50)) if _total_symbols > 0 else 1

    # 按日期 groupby 计数，取行数 ≥ min_rows 的日期为有效交易日
    _date_counts = None
    if "trade_date" in final_prepared.columns:
        _date_counts = final_prepared.groupby("trade_date").size()
    if _date_counts is not None:
        _dense_dates = sorted(
            str(d)[:10] for d in _date_counts[_date_counts >= _min_rows].index
        )
        if len(_dense_dates) < holdout_days:
            logger.warning(
                f"[Holdout] PIT有效交易日仅{len(_dense_dates)}天<holdout_demand{holdout_days}，"
                f"回退到并集日期（覆盖率容差放宽）"
            )
            _dense_dates = None

    _fp_dates_str = [str(d)[:10] for d in _all_dates if d is not None]
    _fp_dates_obj = [
        d for d in map(_to_date, _fp_dates_str) if d is not None
    ]
    _fp_dates_obj.sort()

    # 优先取 dense_dates（覆盖率达标），否则 fallback 到原始并集
    if _dense_dates and len(_dense_dates) >= holdout_days:
        _dense_obj = sorted([d for d in map(_to_date, _dense_dates) if d is not None])
        if len(_dense_obj) >= holdout_days:
            _fp_dates_obj = _dense_obj

    if len(_fp_dates_obj) < holdout_days:
        return None, None
    _start_date = _fp_dates_obj[-holdout_days]  # datetime.date

    if isinstance(equity_curve, pd.DataFrame):
        if equity_curve.empty:
            return None, None
        if "time" in equity_curve.columns:
            _dates = equity_curve["time"].apply(_to_date)
        else:
            _dates = equity_curve.index.to_series().apply(_to_date)
        mask = _dates >= _start_date
        result = equity_curve[mask]

        # P1.5 日志输出：holdout 期有效标的数量与缺失天数
        _ec_dates = set(_dates[mask].dropna().astype(str).str[:10])
        if _total_symbols > 0:
            _holdout_counts = None
            if "trade_date" in final_prepared.columns and "symbol" in final_prepared.columns:
                _ht_mask = final_prepared["trade_date"].astype(str).str[:10].isin(_ec_dates)
                _holdout_counts = final_prepared.loc[_ht_mask].groupby("trade_date")["symbol"].nunique()
            if _holdout_counts is not None:
                _min_coverage = _holdout_counts.min()
                _avg_coverage = _holdout_counts.mean()
                _missing_days = (_holdout_counts < _total_symbols * 0.5).sum()
                logger.info(
                    f"[Holdout] 有效交易日{len(_ec_dates)}天 | "
                    f"标的覆盖率 min={_min_coverage} / avg={_avg_coverage:.0f} / 总数{_total_symbols} | "
                    f"稀疏日{_missing_days}天"
                )
        return result, _start_date.isoformat()

    # list[dict] 分支
    result = [
        e for e in equity_curve
        if _to_date(e.get("time")) is not None and _to_date(e.get("time")) >= _start_date
    ]
    return result, _start_date.isoformat()


def _acceptance_gate(
    *,
    promote: bool,
    oos_decay_pass: bool,
    overfitting_critical: bool,
    sig_pass: bool,
    robust_pass: bool,
    pbo_gate: bool,
    dsr_gate: bool,
) -> tuple[bool, list[str]]:
    """P0-5：统一参数采纳门控（save_calibration 与 write_calibration_to_ini 共用）。

    修复门控不一致：write_calibration_to_ini 曾未应用 PBO/DSR 门控（save_calibration
    有），PBO 过拟合参数集仍可落盘进生产 config.ini。两处采纳决策必须走同一
    门控——任一关键项不通过，calibration_result.json 与 config.ini 均不写入。

    Returns:
        (是否全部通过, 未通过原因列表)。
    """
    reasons: list[str] = []
    if not promote:
        reasons.append("模拟验证未通过")
    if not oos_decay_pass:
        reasons.append("OOS 衰减校验未通过")
    if overfitting_critical:
        reasons.append("多重测试惩罚 CRITICAL")
    if not sig_pass:
        reasons.append("统计显著性未通过")
    if not robust_pass:
        reasons.append("参数稳健性自检不通过")
    if not pbo_gate:
        reasons.append("PBO > 5% 阈值（过拟合风险）")
    if not dsr_gate:
        reasons.append("DSR < 50% 阈值（缩水 Sharpe 不足）")
    return (len(reasons) == 0), reasons


def run_backtest_pipeline(
    config: Config | None = None,
    force: bool = False,
) -> CalibrationResult | None:
    """月度回测管线入口。

    Args:
        config: Config 实例，为空时自动创建。
        force: 是否强制重新运行（忽略 enabled / 频率检查，跳过交互提示）。

    Returns:
        CalibrationResult 或 None（跳过时）。
    """
    if config is None:
        config = Config()

    cfg = config.app_config
    bt = cfg.backtest
    alert = BacktestAlert(config)

    if not force and not bt.ENABLED:
        logger.info("回测未启用 (BACKTEST.enabled=false)，跳过")
        return None

    engine = get_engine(config)
    ensure_table(engine)

    # ── 分布式锁（pg_advisory_xact_lock + NOWAIT 防止阻塞） ──
    _acquire_lock(engine)

    # ── P3.1/P3.2 四方绑定：数据版本 + 配置哈希，变化即强制重跑 ──
    from BackTrading.prepare import _compute_config_hash as _cfg_hash
    _data_version = _compute_kline_data_version(engine)
    _cur_config_hash = _cfg_hash()

    last = get_last_run(engine)
    should_run, reason = should_rerun(
        last, bt.OPTIMIZE_FREQUENCY,
        data_version=_data_version,
        config_hash=_cur_config_hash,
    )

    if not should_run and not force:
        # P0-11：移除阻塞式 input() 交互（生产调度/每日 02:00 DAG 中无终端会挂起）。
        # 默认跳过并提示；需强制重跑时显式传 force=True（或在调度侧传参）。
        logger.info(f"{reason} → 跳过（如需强制重跑请调用 run_backtest_pipeline(force=True)）")
        return load_calibration()

    logger.info("=" * 50)
    logger.info("开始回测管线 ...")
    logger.info(f"  优化频率: {bt.OPTIMIZE_FREQUENCY}")
    logger.info(f"  数据起始日期: {bt.BACKTEST_START_DATE}")
    logger.info(f"  样本外天数: {bt.OUT_OF_SAMPLE_DAYS}")
    logger.info(f"  初始资金: {bt.INITIAL_CASH:,.0f}")

    # ── A2 失败快照：进程级 run_id/task_id 上下文（随日志/告警/快照透出） ──
    import uuid as _uuid

    _run_id = _uuid.uuid4().hex[:12]
    from BackTrading.snapshot import begin_snapshot_session, save_failure_snapshot, set_run_context
    set_run_context(run_id=_run_id, task_id="backtest_pipeline")
    begin_snapshot_session()

    kline_df: pd.DataFrame | None = None

    _step_times: dict[str, float] = {"start": time.time()}
    def _log_step(name: str) -> None:
        _step_times[name] = time.time()
        _elapsed = _step_times[name] - _step_times.get(list(_step_times.keys())[-2] if len(_step_times) >= 2 else "start", 0)
        _total = _step_times[name] - _step_times["start"]
        logger.info(f"[STEP] {name} ({_elapsed:.1f}s, 累计 {_total:.1f}s)")

    try:
        symbols = _resolve_symbols(engine, config)
        logger.info(f"  股票数量: {len(symbols)}")
        _log_step("resolve_symbols")

        kline_df, _delisted_synced = _fetch_kline(engine, symbols, bt.BACKTEST_START_DATE)
        if kline_df.empty:
            logger.warning("K 线数据为空，跳过回测")
            return None

        # P3-5 审计修复：退市股历史 K 线已在 _fetch_kline 内部完成同步
        # （_sync_delisted_stocks 仅接收 engine 与起始日期，且返回退市股集合，
        # 同步结果已并入本次查询）——此处不再重复调用，避免签名不匹配/覆盖 DataFrame。
        _log_step("sync_delisted_stocks")

        logger.info(f"  K 线行数: {len(kline_df)}")

        # P3.1: 从内存 DataFrame 计算数据版本（消除 fetch→version 竞态窗口）
        _data_version = _compute_kline_data_version(engine, kline_df=kline_df)

        # ── ST/退市历史早加载（供 WFO / 模拟验证 / 最终回测全链路使用） ──
        # P0-5: 查询起点覆盖 K 线预热缓冲（_fetch_kline 用 360 日历日缓冲），
        # 否则缓冲期内 ST 涨跌幅 5% 判定缺失。
        _bt_start_iso = datetime.strptime(bt.BACKTEST_START_DATE, "%Y%m%d").date().isoformat()
        _st_query_start = (
            datetime.strptime(bt.BACKTEST_START_DATE, "%Y%m%d").date() - timedelta(days=360)
        ).isoformat()
        _end_date = kline_df["trade_date"].max()
        if pd.api.types.is_datetime64_any_dtype(kline_df["trade_date"]):
            _end_date = _end_date.strftime("%Y-%m-%d")
        # P0-5: ST/退市 PIT 同步（全历史逐日状态回填；网络失败优雅降级，仅告警不阻断）
        try:
            from DataManager.StPitSync import ensure_st_history_table, sync_st_pit
            ensure_st_history_table(engine)
            sync_st_pit(engine, symbols, start_date=_st_query_start, end_date=_end_date)
        except Exception as e:
            logger.warning(f"  ST PIT 同步失败（使用现有 stock_st_history 数据）: {e}")
        st_history = _load_st_history(engine, symbols, _st_query_start, _end_date)

        # ── P0-6 ④: 上市日期表同步（AkShare stock_info_a_code_name → stock_listing_days） ──
        # 显式注入 IPO 日期，引擎禁止从行情数据推断上市日（数据缺口会误判新股，
        # 错误激活"注册制前 5 日无涨跌幅"豁免）。网络失败优雅降级（仅告警不阻断）。
        try:
            from DataManager.ListingDaysSync import (
                ensure_listing_days_table, sync_listing_days,
            )
            ensure_listing_days_table(engine)
            sync_listing_days(engine, symbols)
        except Exception as e:
            logger.warning(f"  上市日期同步失败（引擎将停用新股豁免逻辑）: {e}")
        listing_days = _load_listing_days(engine, symbols, _st_query_start)
        _log_step("load_listing_days")

        # 生存偏差实测评估：池内退市股的历史 K 线是否真实纳入（其退市前负收益才会计入）。
        # P3-5（审计）：评估改用独立数据源（AkShare 交易所退市列表）交叉验证，与
        # stock_st_history PIT 表解耦——PIT 同步失败不应导致"生存偏差受控"误报。
        # 独立源拉取失败 → 降级到 PIT 退市标记口径并注明降级（行为与旧版一致）。
        _kline_syms = set(kline_df["symbol"].astype(str))
        _survival_source = "AkShare 退市列表（独立数据源）"
        try:
            from DataManager.StPitSync import fetch_delisted_symbols
            _delisted_syms = fetch_delisted_symbols() or set()
        except Exception as e:  # noqa: BLE001
            _delisted_syms = set()
            _survival_source = f"PIT 退市标记（独立源拉取失败降级: {e}）"
        if not _delisted_syms:
            _delisted_syms = {
                s for s, recs in st_history.items()
                if any(_is_del for _is_st, _is_del in recs.values())
            }
            if _survival_source.startswith("AkShare"):
                _survival_source = "PIT 退市标记（独立源为空降级）"
        _missing_delisted = sorted(_delisted_syms - _kline_syms)
        if _delisted_syms:
            logger.info(
                f"  独立退市列表（{_survival_source}）识别 {len(_delisted_syms)} 只退市股，"
                f"其中 {len(_delisted_syms & _kline_syms)} 只已纳入 K 线"
            )
            if _missing_delisted:
                logger.warning(
                    f"生存偏差: {len(_missing_delisted)} 只退市股的历史 K 线缺失"
                    f"（如 {_missing_delisted[:5]}），其退市前负收益未被计入，"
                    f"建议扩展数据同步范围（含已退市股票）"
                )
            else:
                logger.info(
                    "生存偏差受控: 退市股历史 K 线已纳入股票池，"
                    "退市前负收益计入回测；ST/退市日的逐日剔除由引擎按 stock_st_history 执行"
                )
        else:
            logger.warning(
                "生存偏差: 独立退市列表与 stock_st_history 均无退市记录，"
                "股票池可能仅含当前存活股票，已退市/ST 股票的历史负收益未被计入"
            )
        _log_step("load_st_history")

        # ── FIX(P0) Subtask-4：退市股覆盖率门禁 ──
        # 如果股票池中几乎没有退市股（覆盖率<5%），可能存在严重数据偏差：
        # 回测只基于存活股票，忽略已退市公司带来的尾部风险。
        # 覆盖率 < 5% 时中断回测（RuntimeError）；独立源为空时仅告警不断。
        # kline中无退市股（_covered==0）时降级为警告，不中断回测——
        # 这是腾讯API无老退市股历史K线数据的已知限制，不应阻断回测。
        _pool_size = len(symbols)
        _delist_coverage = len(_delisted_syms & _kline_syms) / max(_pool_size, 1)
        if _delist_coverage < 0.05 and _delisted_syms:
            _covered = len(_delisted_syms & _kline_syms)
            if _covered == 0:
                logger.warning(
                    f"退市股覆盖率门禁: 退市股覆盖率仅 {_delist_coverage:.1%} "
                    f"（{_covered} / {_pool_size} < 5%），"
                    f"腾讯API无老退市股历史K线数据（已知限制），"
                    f"门禁降级为警告（回测继续，可能存在生存偏差）"
                )
            else:
                raise RuntimeError(
                    f"退市股覆盖率门禁: 退市股覆盖率仅 {_delist_coverage:.1%} "
                    f"（{_covered} / {_pool_size} < 5%），"
                    f"回测结果可能因生存偏差而系统性高估。"
                    f"请扩展数据同步范围以包含退市标的历史 K 线。"
                )
        elif not _delisted_syms and _pool_size > 0:
            logger.warning(
                f"退市股覆盖率门禁: 独立退市列表与 stock_st_history 均无退市记录，"
                f"数据源可能未覆盖退市标的；回测结果可能存在生存偏差（仅告警，不中断）"
            )

        # 窗口坐标轴以正式回测起点为准（起点前为信号预热历史，不参与 WFO 交易）
        # 统一使用 _to_date 归一化，消除 str[:10] 时区/格式脆弱性
        _bt_start = _to_date(bt.BACKTEST_START_DATE)
        assert _bt_start is not None, f"无效的 BACKTEST_START_DATE: {bt.BACKTEST_START_DATE!r}"
        total_trading_days = sum(
            1 for d in kline_df["trade_date"].unique()
            if _to_date(d) is not None and _to_date(d) >= _bt_start
        )
        _oos = bt.OUT_OF_SAMPLE_DAYS
        # ── 末段独立 holdout：终验只在该段进行，WFO 全程禁触 ──
        # holdout_days = round(total * ratio)；钳制 ≥OOS 且 WFO 寻参域须留足 120+OOS，
        # 否则禁用 holdout 回退旧逻辑（自引用，但保证 WFO 至少可跑）。
        _holdout_ratio = float(getattr(bt, "HOLDOUT_RATIO", 0.0))
        _holdout_days = 0
        _holdout_active = False
        if _holdout_ratio > 0:
            _holdout_days = round(total_trading_days * _holdout_ratio)
            _wfo_total = total_trading_days - _holdout_days
            if _holdout_days >= _oos and _wfo_total >= 120 + _oos:
                _holdout_active = True
            else:
                logger.warning(
                    f"  末段 holdout={_holdout_days} 天但条件不满足"
                    f"（holdout≥OOS? {_holdout_days >= _oos} | WFO域={_wfo_total}≥{120 + _oos}? "
                    f"{_wfo_total >= 120 + _oos}），禁用 holdout 回退旧逻辑"
                )
                _holdout_days = 0
        _wfo_total = total_trading_days - _holdout_days
        # 数据自适应 WFO 配置：路径 p 的 offset = p*OOS，需满足 offset + IS + OOS + embargo <= n，
        # 否则路径 2/3 必然越界跳过（如 IS=805+OOS=60 在 865 天数据上只有 1 条路径有效）。
        # ⚠️ 公式必须扣减 embargo：否则 max_offset 恒 = (num_paths-1)*OOS - embargo < 末路径 offset，
        # 末条路径在任何数据长度下都被跳过（train = n - num_paths*OOS 时 span 恰好越界 embargo 天）。
        _embargo_days = max(0, int(bt.BAYESIAN_CPCV_EMBARGO_DAYS))
        _np_cfg = max(1, int(bt.WFO_NUM_PATHS))
        # P1-16 拦截校验：wfo_num_paths 默认≥5，低于5时阻断回测（而非仅告警）
        if _np_cfg < 5:
            logger.error(
                f"WFO 多路径数 wfo_num_paths={_np_cfg} 低于审计最低要求(≥5)，"
                f"回测终止。请修改 config.ini [BACKTEST] wfo_num_paths ≥ 5"
            )
            sys.exit(2)
        # P1-7 WFO维度采样加固：IS窗口下限从120提升至180，提高训练样本质量
        _is_min = 180  # P1-7 最低IS窗口长度（原120）
        _max_np = max(1, (_wfo_total - _is_min - _embargo_days) // _oos) if _wfo_total > _oos + _is_min + _embargo_days else 1
        _num_paths = min(_np_cfg, _max_np)
        train_period = max(
            _is_min,
            min(
                _wfo_total - _oos - _embargo_days,
                _wfo_total - _oos * _num_paths - _embargo_days,
            ),
        )
        _holdout_label = (
            f" | Holdout: {_holdout_days}天(占比{_holdout_days/total_trading_days:.0%},独立终验)"
            if _holdout_active else " | Holdout: 禁用(自引用回退)"
        )
        logger.info(
            f"  交易日数: {total_trading_days} | IS训练窗口: {train_period}天 | OOS: {_oos}天"
            f" | WFO路径数: {_num_paths}（配置 {_np_cfg}，数据上限 {_max_np}）"
            f"{_holdout_label}"
        )
        _log_step("fetch_kline")
        # ── WFO 系统性失败拦截：捕获 WFOSystematicFailure，记录失败并中断流水线 ──
        from BackTrading.bayesian.meta_optimizer import WFOSystematicFailure

        try:
            wf_result = run_walk_forward(
                kline_df=kline_df,
                num_paths=_num_paths,
                train_period=train_period,
                test_period=bt.OUT_OF_SAMPLE_DAYS,
                initial_cash=bt.INITIAL_CASH,
                commission=bt.COMMISSION_RATE,
                stamp_tax=bt.STAMP_TAX_RATE,
                slippage=bt.SLIPPAGE,
                max_position_pct=bt.MAX_POSITION_PCT,
                portfolio_method=bt.PORTFOLIO_METHOD,
                point_in_time=bt.POINT_IN_TIME,
                show_progress=True,
                backtest_start_date=_bt_start_iso,
                st_history=st_history,
                exclude_st=False,  # FIX(P0): 回测不排除 ST，防止 train/serve skew
                # 复盘单元的 ST 过滤已在 Review/coordinator.py 硬编码执行（不复归 config 控制），
                # 回测时排除 ST 会导致模型在"干净"样本池训练 → OOS 失效。
                listing_days=listing_days,
                # P2.1 CPCV 净化+禁运
                purge_days=int(bt.BAYESIAN_CPCV_PURGE_DAYS),
                embargo_days=int(bt.BAYESIAN_CPCV_EMBARGO_DAYS),
                # P2.4 预算制
                time_budget_seconds=float(bt.BAYESIAN_TIME_BUDGET_SECONDS),
                max_no_improve_windows=int(bt.BAYESIAN_MAX_NO_IMPROVE_WINDOWS),
                # 末段独立 holdout：WFO 寻参上界切除末段，供终验独立使用
                holdout_days=_holdout_days,
                # P3.1 数据版本入缓存 key
                data_version=_data_version,
                # A2 失败快照上下文
                run_id=_run_id,
                task_id="backtest_pipeline",
                # P1-4 行业映射注入：将 db_engine 透传至引擎，启动时刷新行业缓存
                db_engine=engine,
                # P1-5 max_order_pct 分档注入（getattr 回退防止旧版模型缺少字段）
                max_order_pct=float(getattr(bt, "MAX_ORDER_PCT", 0.30)),
                max_order_pct_high=float(getattr(bt, "MAX_ORDER_PCT_HIGH", 0.20)),
                max_order_pct_low=float(getattr(bt, "MAX_ORDER_PCT_LOW", 0.10)),
            )
        except WFOSystematicFailure as wfo_err:
            # WFO 系统性失败：策略在当前数据区间无泛化能力，中断流水线
            logger.critical("=" * 60)
            logger.critical(f"[WFO系统性失败] {wfo_err.reason}")
            logger.critical(
                "策略在当前数据区间不具备泛化能力，建议检查：\n"
                "  1. 特征工程是否存在隐性数据泄露\n"
                "  2. 信号逻辑是否适应当前市场 regime\n"
                "  3. 回看区间是否过短/过长导致过拟合\n"
                "本次回测已标记为失败；回退参数（参数空间中位）仍会写入 config.ini，"
                "下次调度将跳过（需手动 force=True 或数据/配置变化后重跑）"
            )
            logger.critical("=" * 60)

            # 记录失败状态，避免下次调度重复执行
            record_run(
                engine=engine,
                frequency=bt.OPTIMIZE_FREQUENCY,
                backtest_start_date=bt.BACKTEST_START_DATE,
                out_of_sample_days=bt.OUT_OF_SAMPLE_DAYS,
                initial_cash=bt.INITIAL_CASH,
                params=wfo_err.fallback_params,
                sharpe=0,
                total_return=0,
                max_drawdown=0,
                status="failed",
                config_hash=_cur_config_hash,
                data_version=_data_version,
            )

            # 即使 WFO 系统性失败，本次回测实际评估过的参数（回退参数 =
            # 参数空间中位）仍直接写入 config.ini [BACKTEST_CALIBRATED]，
            # 与成功路径写回闭环同口径，保证下次加载沿用本次结果。
            try:
                written = write_calibration_to_ini(wfo_err.fallback_params)
                if written:
                    logger.warning(
                        f"[WFO系统性失败] 回退参数已写入 config.ini "
                        f"[BACKTEST_CALIBRATED]: {written}"
                    )
                else:
                    logger.warning("[WFO系统性失败] 回退参数为空，未写入 config.ini")
            except Exception as _wfo_write_err:
                logger.opt(exception=True).warning(
                    f"[WFO系统性失败] 回退参数写入 config.ini 被拒"
                    f"（config 保持原值，下次运行将重新评估）: {_wfo_write_err}"
                )

            alert.on_failure(wfo_err)
            return None
        _log_step("walk_forward")
        logger.info(f"  Walk-Forward 片段数: {len(wf_result)}")

        if not wf_result.empty and wf_result["sharpe_ratio"].max() > 3.0:
            logger.warning(f"akquant 结果异常: Sharpe={wf_result['sharpe_ratio'].max():.2f}>3.0，可能存在前瞻偏差")

        best_params = _extract_best_params(wf_result, config=config)
        logger.info(f"  最佳参数(Sharpe加权前{min(5, len(wf_result))}): {best_params}")

        # ST/退市逐日数据注入（引擎按 params 消费，WFO 已同口径）
        best_params["_st_history"] = st_history
        best_params["_exclude_st"] = False  # FIX(P0): 回测不排除 ST，防止 train/serve skew
        #   ST 过滤已在复盘单元（Review/coordinator.py）硬编码执行（不复归 config 控制）
        #   回测时排除 ST → 模型在"干净"样本池训练 → 生产含 ST → OOS 失效（过拟合）
        # P0-6 ④：上市日期显式注入（引擎禁止数据推断；空表时豁免逻辑整体停用）
        if listing_days:
            best_params["_listing_days"] = listing_days

        # FIX(P2): 过滤 best_params 中的非 JSON 可序列化字段（如 Engine 对象），
        # 阻断其流入 record_run → json.dumps 导致 TypeError。
        # 注意：_db_engine 在下方全量回测前重新注入（L610），此处过滤不影响回测执行。
        _json_types = (int, float, str, bool, dict, list, tuple, type(None))
        for _k in list(best_params.keys()):
            if not isinstance(best_params[_k], _json_types):
                logger.debug(f"  [P2] 过滤 best_params 非 JSON 字段: {_k} ({type(best_params[_k]).__name__})")
                del best_params[_k]

        from BackTrading.engine import EngineConfig, run_full_backtest
        from BackTrading.domain.models import CostModel

        _sc = config.app_config.scoring_params
        # 组合参数若未被寻优（兜底路径），取校准覆写值（无校准则配置默认，
        # P0-7 ②：与 [BACKTEST_CALIBRATED] 写回闭环一致，替代旧的区间中位口径）
        ecfg = EngineConfig(
            initial_cash=bt.INITIAL_CASH,
            commission_rate=bt.COMMISSION_RATE,
            stamp_tax_rate=bt.STAMP_TAX_RATE,
            slippage=bt.SLIPPAGE,
            max_position_pct=bt.MAX_POSITION_PCT,
            portfolio_method=bt.PORTFOLIO_METHOD,
            point_in_time=bt.POINT_IN_TIME,
            atr_stop_mult=best_params.get("atr_stop_mult", _sc.ATR_STOP_MULT),
            buy_threshold=int(best_params.get("buy_threshold", bt.BUY_THRESHOLD)),
            max_holdings=int(best_params.get("max_holdings", bt.MAX_HOLDINGS)),
            cost_model=CostModel.from_backtest_config(
                bt, trading_cost=config.app_config.trading_cost
            ),
            execution_model=bt.EXECUTION_MODEL,
            simulate_limit_up_down=bool(bt.SIMULATE_LIMIT_UP_DOWN),
            limit_seal_ratio=float(bt.LIMIT_SEAL_RATIO),
            limit_tradable_ratio=float(bt.LIMIT_TRADABLE_RATIO),
            limit_intraday_ratio=float(bt.LIMIT_INTRADAY_RATIO),
            limit_seal_decay=float(bt.LIMIT_SEAL_DECAY),
            # P0-6 ⑥：开盘集合竞价成交率分档
            auction_fill_ratio=float(bt.AUCTION_FILL_RATIO),
            # 技术债修复：经验填充模型（历史日线分位数替代固定比例常量）
            limit_ratio_mode=str(bt.LIMIT_RATIO_MODE),
            limit_calib_min_samples=int(bt.LIMIT_CALIB_MIN_SAMPLES),
            # P0-6 ⑤：市场状态客观变量（指数20日收益 + 波动率分位）
            regime_ret20_full=float(bt.REGIME_RET20_FULL),
            regime_ret20_half=float(bt.REGIME_RET20_HALF),
            regime_vol_pct_max=float(bt.REGIME_VOL_PCT_MAX),
            resume_gap_up=float(bt.RESUME_GAP_UP),
            resume_gap_down=float(bt.RESUME_GAP_DOWN),
            # ── P4 组合优化器配置 ──
            optimizer_method=config.app_config.portfolio_optimizer.METHOD,
            optimizer_risk_aversion=config.app_config.portfolio_optimizer.RISK_AVERSION,
            optimizer_turnover_penalty=config.app_config.portfolio_optimizer.TURNOVER_PENALTY,
            optimizer_max_weight=config.app_config.portfolio_optimizer.MAX_WEIGHT,
            optimizer_cov_lookback=config.app_config.portfolio_optimizer.COV_LOOKBACK,
            optimizer_shrinkage=config.app_config.portfolio_optimizer.SHRINKAGE,
            optimizer_industry_neutral=config.app_config.portfolio_optimizer.INDUSTRY_NEUTRAL,
            optimizer_industry_deviation=config.app_config.portfolio_optimizer.INDUSTRY_DEVIATION,
            optimizer_max_holdings=config.app_config.portfolio_optimizer.MAX_HOLDINGS,
            optimizer_target_cash=config.app_config.portfolio_optimizer.TARGET_CASH_RATIO,
            optimizer_solve_timeout=config.app_config.portfolio_optimizer.SOLVE_TIMEOUT,
            optimizer_verbose=config.app_config.portfolio_optimizer.VERBOSE,
            # ── 市场过滤器（A3：大盘风控开关） ──
            market_filter_enabled=bool(getattr(bt, "MARKET_FILTER_ENABLED", False)),
            market_filter_bull_ratio=float(getattr(bt, "MARKET_FILTER_BULL_RATIO", 0.55)),
            market_filter_min_stocks=int(getattr(bt, "MARKET_FILTER_MIN_STOCKS", 10)),
            # ── ATR 风险驱动仓位控制（A4） ──
            risk_per_trade=float(getattr(bt, "RISK_PER_TRADE", 0.02)),
            # P1-5 max_order_pct 分档注入（getattr 回退防止旧版模型缺少字段）
            max_order_pct=float(getattr(bt, "MAX_ORDER_PCT", 0.30)),
            max_order_pct_high=float(getattr(bt, "MAX_ORDER_PCT_HIGH", 0.20)),
            max_order_pct_low=float(getattr(bt, "MAX_ORDER_PCT_LOW", 0.10)),
        )
        final_params = _build_params(config)
        # 统一信号参数注入：使用 prepare.merge_best_params_into_structured 唯一入口，
        # 杜绝 runner / simulated_trading / prepare 三处白名单各自维护导致的漂移。
        fb_cfg = config.app_config.full_bull_scoring
        merge_best_params_into_structured(
            best_params, final_params, full_bull_default=fb_cfg.CONCLUSION_FULL_BULL
        )

        # ── 模拟交易验证：优先用末段独立 holdout 验证集（WFO 全程禁触），
        #    未激活时回退最近交易日（自引用，validate_params 内告警）──
        from BackTrading.simulated_trading import validate_params as _sim_validate
        _wf_sharpe = float(wf_result["sharpe_ratio"].mean()) if not wf_result.empty else 0.0
        _holdout_dates: set[str] | None = None
        if _holdout_active and _holdout_days > 0:
            _k_dates = sorted(pd.Series(kline_df["trade_date"]).astype(str).unique())
            _holdout_dates = set(_k_dates[-_holdout_days:])
        _sim_verdict = _sim_validate(
            kline_df=kline_df, best_params=best_params,
            oos_sharpe=_wf_sharpe, sim_days=20,
            config=config, engine_cfg=ecfg,
            validation_dates=_holdout_dates,
            oos_returns=_extract_oos_returns(wf_result),
        )
        _promote = _sim_verdict.promote
        if not _promote:
            logger.warning(f"模拟验证不通过，参数不写入 config.ini: {_sim_verdict.reason}")

        _log_step("prepare_final_signals")
        final_prepared = prepare_backtest_data(kline_df, params=final_params, compute_exit_strategy=True, vectorized=True, backtest_start_date=_bt_start_iso, data_version=_data_version)
        _log_step("full_backtest")
        # ST 历史已早加载并注入 best_params（见上方 _st_history 注入）
        # P1-4 行业映射：透传 db_engine 至引擎
        best_params["_db_engine"] = engine
        trade_log, equity_curve = run_full_backtest(final_prepared, best_params, ecfg)
        _log_step("compute_metrics")
        risk = compute_risk_metrics(equity_curve) or {}
        trade = compute_trade_metrics(trade_log) or {}

        logger.info(f"  ── 绩效分析 ──")
        logger.info(f"  Sharpe={risk.get('sharpe_ratio', 0):.2f} | Sortino={risk.get('sortino_ratio', 0):.2f} | Calmar={risk.get('calmar_ratio', 0):.2f}")
        logger.info(f"  VaR(95%)={risk.get('var_95', 0):.2%} | CVaR(95%)={risk.get('cvar_95', 0):.2%} | MaxDD={risk.get('max_drawdown', 0):.2%}")
        logger.info(f"  交易={trade.get('total_trades', 0)} | 胜率={trade.get('win_rate', 0):.1%} | 盈亏比={trade.get('profit_factor', 0):.2f}")
        logger.info(f"  日均换手率={risk.get('avg_turnover', 0):.2%} | 最高单日换手率={risk.get('max_turnover', 0):.2%}")
        _avg_to = risk.get("avg_turnover", 0)
        if _avg_to and _avg_to > 0.30:
            logger.warning(f"日均换手率 {_avg_to:.2%} > 30%，扣费后实际收益可能打 7 折")
        logger.info(f"  最佳参数(Sharpe加权前{min(5, len(wf_result))}): {best_params}")

        # ── P1#6 冲击成本参数校准摘要（报告输出接口） ──
        # 将实际生效的冲击成本模型参数打印到日志，便于审计/复现。
        try:
            _cm = ecfg.cost_model
            if _cm is not None:
                _tier_bases = getattr(_cm, 'tier_impact_base', None)
                _tier_caps = getattr(_cm, 'tier_cap', None)
                _tier_thresh = getattr(_cm, 'tier_threshold', None)
                _tier_edges = getattr(_cm, 'tier_edges', None)
                if _tier_bases is not None:
                    logger.info(f"  ── 冲击成本模型参数 ──")
                    logger.info(f"  流动性分档边界(AMOUNT_MA20): {_tier_edges}")
                    logger.info(f"  各档 impact_base: {_tier_bases}")
                    logger.info(f"  各档 threshold: {_tier_thresh}")
                    logger.info(f"  各档 cap: {_tier_caps}")
                    # 估算综合冲击成本占比（基于 turnover 和 avg impact）
                    _total_cost_sum = trade.get('total_commission', 0) + trade.get('total_stamp_tax', 0) + trade.get('total_impact_cost', 0)
                    _gross_pnl = risk.get('total_return', 0) * bt.INITIAL_CASH
                    if _gross_pnl > 0 and _total_cost_sum > 0:
                        _impact_pct = trade.get('total_impact_cost', 0) / bt.INITIAL_CASH
                        logger.info(f"  冲击成本总额={trade.get('total_impact_cost', 0):,.0f}元 (占总资金{_impact_pct:.2%})")
        except Exception:
            pass  # 不影响主流程

        logger.info(f"  ── 绩效分析 ──")
        # 三项硬 gate：样本量 / 持仓周期健康度 / 牛熊覆盖
        _sig_pass = True
        try:
            from LogicAnalyzer.statistical_significance import run_significance_check as _sig_check
            _sig_summary = _sig_check(trade_log, kline_df)
            if not _sig_summary.passed:
                _sig_pass = False
                logger.warning(
                    f"[统计显著性] 综合判定 FAIL — {_sig_summary.reason}"
                )
        except Exception as e:
            logger.warning(f"[统计显著性] 自检异常: {e}，不阻断（建议人工复核）")

        # ── 持仓打分卡：当期持仓的因子分解 ──
        try:
            _holdings = [t for t in trade_log if t.get("action") == "buy"][-20:]  # 最近 20 笔买入
            if _holdings and not final_prepared.empty:
                _last_date = final_prepared["trade_date"].max()
                if pd.api.types.is_datetime64_any_dtype(final_prepared["trade_date"]):
                    _fp = final_prepared.copy()
                    _fp["trade_date"] = _fp["trade_date"].dt.strftime("%Y-%m-%d")
                    _last_date_str = _last_date.strftime("%Y-%m-%d") if hasattr(_last_date, "strftime") else str(_last_date)
                    _latest = _fp[_fp["trade_date"] == _last_date_str]
                else:
                    _latest = final_prepared[final_prepared["trade_date"] == _last_date]
                _score_cols = ["MACD趋势分", "金叉信号分", "柱状动能分", "DIF斜评分",
                               "背离信号分", "量价配合分", "K线形态分"]
                _held_syms = list({t["symbol"] for t in _holdings if t["symbol"] in _latest["symbol"].values})
                if _held_syms:
                    _card = _latest[_latest["symbol"].isin(_held_syms)][
                        ["symbol", "进场评分", "综合评分", "风险等级"] + _score_cols
                    ].copy()
                    _card.columns = ["股票", "进场分", "综合分", "风险"] + [
                        "MACD趋势", "金叉", "动能", "DIF斜率", "背离", "量价", "K线"
                    ]
                    logger.info(f"  ── 持仓因子分解（{_last_date}）──")
                    for _, r in _card.iterrows():
                        _factors = " | ".join(f"{c}={r[c]:.0f}" for c in ["MACD趋势","金叉","动能","DIF斜率","背离","量价","K线"])
                        logger.info(f"    {r['股票']}: 综合{r['综合分']:.0f}/进场{r['进场分']:.0f}/{r['风险']} | {_factors}")
        except Exception:
            pass

        # ── 因子暴露归因 ──
        try:
            _ec_df = pd.DataFrame(equity_curve).set_index("time")
            _ec_df.index = pd.to_datetime(_ec_df.index)
            _port_rets = _ec_df["portfolio_value"].pct_change().dropna()
            if len(_port_rets) > 20:
                from BackTrading.attribution import factor_exposure as _fe
                # 用市场指数收益率作为因子代理
                _index_map = {"000300.SH": "沪深300", "000905.SH": "中证500", "000852.SH": "中证1000"}
                _factor_data = {}
                for _code, _name in _index_map.items():
                    try:
                        from UtilsManager.IDataProvider import BacktestDataProvider as _Bdp
                        from DataManager.DbEngine import get_engine as _ge
                        _e2 = _ge(config)
                        _p = _Bdp(_e2)
                        _idx = _p.get_index_kline(_code, start=_port_rets.index[0].strftime("%Y-%m-%d"))
                        if _idx is not None and not _idx.empty:
                            _idx = _idx.set_index("trade_date")
                            _idx.index = pd.to_datetime(_idx.index)
                            _factor_data[_name] = _idx["close"].pct_change()
                    except Exception:
                        continue
                if _factor_data:
                    _fdf = pd.DataFrame(_factor_data)
                    _fe_result = _fe(_port_rets, _fdf)
                    _fe_line = " | ".join(
                        f"{k}: β={_fe_result.exposures.get(k, 0):.2f}"
                        f"(p={_fe_result.p_values.get(k, 1):.2f})"
                        for k in _fdf.columns
                    )
                    logger.info(f"  因子暴露[{_fdf.columns.tolist()}]: {_fe_line}")
                    logger.info(f"  回归R²={_fe_result.rsquared:.2%}, adjR²={_fe_result.adj_rsquared:.2%}")
        except Exception:
            pass

        # ── 组合风险暴露（行业 + 风格） ──
        try:
            if pd.api.types.is_datetime64_any_dtype(final_prepared["trade_date"]):
                _fp2 = final_prepared.copy()
                _fp2["trade_date"] = _fp2["trade_date"].dt.strftime("%Y-%m-%d")
                _last_bar = _fp2[_fp2["trade_date"] == _fp2["trade_date"].max()]
            else:
                _last_bar = final_prepared[final_prepared["trade_date"] == final_prepared["trade_date"].max()]
            _risk_holdings = {t["symbol"]: t.get("value", 0) for t in trade_log if t.get("action") == "buy"}
            _total_val = sum(_risk_holdings.values()) or 1
            _pw = pd.Series({k: v / _total_val for k, v in _risk_holdings.items()})
            if len(_pw) > 1 and "行业" in _last_bar.columns:
                from BackTrading.risk_model import compute_industry_exposure, industry_hhi
                _ind_map = _last_bar.set_index("symbol")["行业"].to_dict()
                _ind_exp = compute_industry_exposure(_pw, pd.Series({k: _ind_map.get(k, "未知") for k in _pw.index}))
                _top_ind = sorted(_ind_exp.items(), key=lambda x: -x[1])[:5]
                _hhi = industry_hhi(_ind_exp)
                _ind_line = " | ".join(f"{s}: {w:.1%}" for s, w in _top_ind)
                logger.info(f"  行业暴露 Top5: {_ind_line}")
                if _hhi > 0.3:
                    logger.warning(f"  行业 HHI={_hhi:.2f} > 0.3，集中度偏高")
        except Exception:
            pass

        # ── 因子衰减检查（信号分 vs 前向收益的 Rank IC） ──
        # P1-11 修复：用 close_normal（后复权）计算前向收益，避免除权日不复权close跳空污染Rank-IC
        try:
            _price_col_ic = "close_normal" if "close_normal" in final_prepared.columns else "close"
            _fwd_ret = final_prepared.groupby("symbol")[_price_col_ic].transform(
                lambda s: s.shift(-5) / s - 1
            )
            _ic_cols = ["MACD趋势分", "金叉信号分", "柱状动能分", "DIF斜评分", "背离信号分", "量价配合分", "K线形态分"]
            _ic_factors = {c: "MACD趋势", "金叉信号": "金叉", "柱状动能": "动能",
                           "DIF斜评分": "斜率", "背离信号": "背离", "量价配合": "量价", "K线形态分": "K线"}
            _ics = []
            for _c in _ic_cols:
                if _c not in final_prepared.columns:
                    continue
                _valid = final_prepared[_c].notna() & _fwd_ret.notna()
                if _valid.sum() < 20:
                    continue
                _rho, _ = spearmanr(final_prepared.loc[_valid, _c], _fwd_ret[_valid])
                if not np.isnan(_rho):
                    _ics.append((_ic_factors.get(_c, _c), _rho))
            if _ics:
                _ic_line = " | ".join(f"{n}: IC={r:.3f}" for n, r in _ics)
                logger.info(f"  信号Rank IC（5日前向收益）: {_ic_line}")
        except Exception:
            pass

        top = wf_result.dropna(subset=["sharpe_ratio"]).sort_values("sharpe_ratio", ascending=False).head(5)
        sharpe_avg = float(top["sharpe_ratio"].mean()) if not top.empty else 0.0
        # 兜底结果帧（无有效窗口时）可能缺少绩效列，逐列防御
        total_return_avg = float(top["total_return"].mean()) if "total_return" in top.columns and not top.empty else 0.0
        max_dd_avg = float(top["max_drawdown"].mean()) if "max_drawdown" in top.columns and not top.empty else 0.0

        # ── Holdout 终验 Sharpe（修正"选优报优"乐观偏差）──
        # holdout 激活时，业绩报告使用 holdout 终验 sharpe（末段 20% 独立回测），
        # WFO Top 5 均值仅用于参数选择，不对外报告。
        holdout_sharpe = None
        if _holdout_active and _holdout_days > 0:
            try:
                holdout_equity, _holdout_start_date = _holdout_equity_slice(
                    equity_curve, final_prepared, _holdout_days
                )
                if holdout_equity is not None and len(holdout_equity) >= 20:
                    holdout_risk = compute_risk_metrics(holdout_equity) or {}
                    holdout_sharpe = holdout_risk.get("sharpe_ratio")
                    logger.info(
                        f"  [Holdout终验] {_holdout_start_date}起末段{_holdout_ratio:.0%}"
                        f"共{len(holdout_equity)}条, Sharpe={holdout_sharpe:.4f}"
                        f"（WFO Top5 均值={sharpe_avg:.4f}）"
                    )
                elif holdout_equity is not None:
                    logger.warning(f"  [Holdout终验] 数据仅{len(holdout_equity)}条<20，回退 WFO 均值")
                else:
                    logger.warning(f"  [Holdout终验] 净值曲线为空或交易日不足，回退 WFO 均值")
            except Exception as e:
                logger.warning(f"  [Holdout终验] 计算异常: {e}，回退 WFO 均值")

        # 业绩报告 sharpe：优先 holdout 终验，其次 WFO Top 5
        report_sharpe = holdout_sharpe if holdout_sharpe is not None else sharpe_avg

        # P3 审计修复：报告层注明区间口径——Sharpe 来自 holdout 末段/WFO Top5
        # （选择期），total_return/max_drawdown 来自全周期最终回测（评估期），
        # 选择期≠评估期，跨期对比指标时必须区分区间，避免口径混淆
        if holdout_sharpe is not None:
            _sharpe_scope = f"末段独立 holdout（{_holdout_days} 个交易日）"
        else:
            _sharpe_scope = "WFO Top5 窗口均值"
        logger.info(
            f"[指标口径] 报告 Sharpe={report_sharpe:.4f} 区间={_sharpe_scope}（选择期）；"
            f"total_return={risk.get('total_return', 0):.2%} / "
            f"max_drawdown={risk.get('max_drawdown', 0):.2%} 为全周期最终回测口径"
            f"（评估期）——两者区间不一致属预期设计，跨期对比请注意"
        )

        from BackTrading.calibration import _get_git_commit
        from BackTrading.prepare import _compute_config_hash

        from BackTrading.overfitting import compute_pbo, compute_dsr_from_equity_curve

        wf_results_list = wf_result.to_dict("records") if not wf_result.empty else []
        pbo = compute_pbo(wf_results_list)
        num_combos = int(wf_result["num_combos"].iloc[0]) if not wf_result.empty and "num_combos" in wf_result.columns else 1
        num_trials = num_combos * len(wf_result)
        dsr = compute_dsr_from_equity_curve(equity_curve, num_trials)

        logger.info(f"  Deflated Sharpe Ratio(DSR)={dsr:.2%} | PBO={pbo:.2%} | 试验次数={num_trials}")
        if pbo > 0.5:
            logger.warning(f"PBO={pbo:.2%}>50%，过拟合风险较高，建议缩减参数网格或增加数据")
        # P1-6 DSR阈值动态化：试验次数越多，随机发现"好"结果概率越高，阈值应相应收紧
        _dsr_threshold = min(0.5, max(0.3, 0.5 * math.sqrt(100 / max(num_trials, 100))))
        if dsr < _dsr_threshold:
            logger.warning(f"DSR={dsr:.2%} < 动态阈值{_dsr_threshold:.2%}（试验{num_trials}次），统计显著性不足")

        # ══ 多重测试惩罚（Multiple Testing Deception）══
        # 统计同区间调参次数，超限则对 Sharpe/Sortino 施加统计学硬扣减
        from BackTrading.calibration_log import (
            count_tuning_attempts as _count_attempts,
            apply_multiple_testing_penalty as _apply_penalty,
            MAX_TUNING_ATTEMPTS as _max_attempts,
            MULTIPLE_TESTING_PENALTY as _penalty_rate,
        )
        _raw_sharpe = risk.get("sharpe_ratio", sharpe_avg)
        _raw_sortino = risk.get("sortino_ratio", 0)
        _attempt_count = _count_attempts(engine, bt.BACKTEST_START_DATE, bt.OUT_OF_SAMPLE_DAYS) + 1  # +1 包含本次
        _pun_sharpe, _pun_sortino, _warning_level = _apply_penalty(
            _raw_sharpe, _raw_sortino, _attempt_count,
            bt.BACKTEST_START_DATE, bt.OUT_OF_SAMPLE_DAYS,
        )
        # 用惩罚后的值替代原始值
        if _warning_level != "INFO":
            risk["sharpe_ratio"] = _pun_sharpe
            risk["sortino_ratio"] = _pun_sortino
            logger.warning(
                f"[多重测试惩罚] 原始 Sharpe={_raw_sharpe:.4f} → 惩罚后={_pun_sharpe:.4f} | "
                f"原始 Sortino={_raw_sortino:.4f} → 惩罚后={_pun_sortino:.4f}"
            )
        logger.info(
            f"[多重测试惩罚] 同区间累计调参 {_attempt_count} 次，阈值 {_max_attempts}，"
            f"惩罚率 {_penalty_rate:.0%}，级别={_warning_level}"
        )
        # 高危级别额外阻断
        _overfitting_critical = _warning_level == "CRITICAL"

        # ══ 邻近参数抖动自检（Parameter Robustness Check）══
        # Sharpe > 2.0 时自动触发 ±10% 参数扰动测试
        _robust_pass = True
        try:
            from BackTrading.parameter_robustness import run_robustness_check as _robust_check
            _robust_report = _robust_check(
                kline_df, best_params, _pun_sharpe, config, ecfg,
            )
            if _robust_report.triggered and not _robust_report.overall_robust:
                _robust_pass = False
                if _robust_report.warning_level == "CRITICAL":
                    logger.critical(
                        f"[参数稳健性] 🔴 CRITICAL: {len(_robust_report.failed_params)} 个参数扰动后 "
                        f"Sharpe 断崖式下跌，策略不具备统计稳健性: {_robust_report.failed_params}"
                    )
                else:
                    logger.warning(
                        f"[参数稳健性] ⚠️ {len(_robust_report.failed_params)} 个参数扰动后 "
                        f"Sharpe 显著下跌，建议谨慎: {_robust_report.failed_params}"
                    )
        except Exception as e:
            logger.warning(f"[参数稳健性] 自检异常: {e}，不阻断（建议人工复核）")

        # ── 样本外衰减校验（审计 gate：IS vs OOS 夏普/索提诺衰减 ≤ 30%） ──
        # P1-3 修复：OOS 段必须使用 WFO 全程禁触的独立 holdout 数据。
        # 自引用回退（从全周期净值尾部切段）被 WFO 评估过——门控失效。
        # 策略：holdout 未激活时，OOS 衰减门直接 FAIL 并拦截参数写入。
        from BackTrading.overfitting import validate_oos_decay as _validate_oos_decay

        _oos_decay_pass = True
        try:
            if _holdout_active and _holdout_days > 0:
                _oos_n = _holdout_days
                _decay_tag = "独立Holdout"
            else:
                # P0.1 修复：自引用 OOS 不能证明泛化能力，直接 FAIL
                _oos_decay_pass = False
                _decay_tag = "自引用回退(已禁用)"
                logger.warning(
                    f"[OOS衰减校验] Holdout 未激活 → REJECT (reject-by-default)。"
                    f"自引用 OOS 与 WFO 评估段重叠，无法证明泛化能力，参数将不予采纳。"
                )
            _eq = pd.DataFrame(equity_curve) if isinstance(equity_curve, list) else equity_curve
            if not _eq.empty and "time" in _eq.columns:
                # 确保日期为字符串统一比较
                if pd.api.types.is_datetime64_any_dtype(_eq["time"]):
                    _eq = _eq.copy()
                    _eq["time"] = _eq["time"].dt.strftime("%Y-%m-%d")

                # P1-3：holdout 激活时，OOS 门控使用与 WFO 禁触边界同一交易日口径（_holdout_dates）
                # 避免从 equity_curve 自行切段与 WFO 的 validation_dates 错位
                if _holdout_active and _holdout_dates is not None:
                    _oos_dates = _holdout_dates
                    _all_dates = sorted(_eq["time"].unique())
                    _is_dates = set(_all_dates) - _oos_dates
                    _is_curve = _eq[_eq["time"].isin(_is_dates)]
                    _oos_curve = _eq[_eq["time"].isin(_oos_dates)]
                else:
                    _all_dates = sorted(_eq["time"].unique())
                    _total_td = len(_all_dates)
                    _is_end = max(_total_td - _oos_n, 1)
                    _is_dates = set(_all_dates[:_is_end])
                    _oos_dates = set(_all_dates[_is_end:])
                    _is_curve = _eq[_eq["time"].isin(_is_dates)]
                    _oos_curve = _eq[_eq["time"].isin(_oos_dates)]

                if len(_oos_dates) >= 2:

                    _report = _validate_oos_decay(
                        _is_curve, _oos_curve,
                        decay_threshold=0.25,  # P1-7 门控收紧：从30%降至25%
                        is_days=len(_is_dates),
                        oos_days=len(_oos_dates),
                    )
                    if not _report.passed:
                        _oos_decay_pass = False
                        logger.warning(
                            f"[OOS衰减校验][{_decay_tag}] FAIL | IS_Sharpe={_report.is_sharpe:.2f} → "
                            f"OOS_Sharpe={_report.oos_sharpe:.2f} (衰减 {_report.sharpe_decay:.1%}) | "
                            f"IS_Sortino={_report.is_sortino:.2f} → OOS_Sortino={_report.oos_sortino:.2f} "
                            f"(衰减 {_report.sortino_decay:.1%})"
                        )
                        logger.warning(f"[OOS衰减校验][{_decay_tag}] {_report.reason}")
                    else:
                        logger.info(
                            f"[OOS衰减校验][{_decay_tag}] PASS | IS_Sharpe={_report.is_sharpe:.2f} → "
                            f"OOS_Sharpe={_report.oos_sharpe:.2f} (衰减 {_report.sharpe_decay:.1%}) | "
                            f"IS_Sortino={_report.is_sortino:.2f} → OOS_Sortino={_report.oos_sortino:.2f} "
                            f"(衰减 {_report.sortino_decay:.1%})"
                        )
                else:
                    logger.info(f"[OOS衰减校验] OOS 交易日仅 {len(_oos_dates)} 天 < 2 天，跳过")
            else:
                logger.info("[OOS衰减校验] 净值曲线为空，跳过")
        except Exception as e:
            logger.warning(f"[OOS衰减校验] 执行异常: {e}，不阻断（建议人工复核）")

        if not _oos_decay_pass:
            logger.warning("=" * 50)
            logger.warning("[OOS衰减校验] 未通过 —— 参数组不予写入 config.ini，结果已废弃")
            logger.warning("=" * 50)

        # ── 序列化当前成本模型快照（方案C：持久化回测验证假设）──
        from dataclasses import asdict as _dc_asdict
        _full_cost = _dc_asdict(ecfg.cost_model)
        _cost_snapshot = {
            k: _full_cost[k] for k in [
                "commission_rate", "stamp_tax_rate", "market_slippage", "limit_slippage",
                "impact_threshold", "impact_base", "impact_cap",
                "min_commission_per_trade", "transfer_fee_rate",
                "handling_fee_rate", "csrc_fee_rate",
                "commission_includes_fees",
            ] if k in _full_cost
        }

        cal_result = CalibrationResult(
            params=best_params,
            score=report_sharpe,
            sharpe=report_sharpe,
            sortino=risk.get("sortino_ratio", 0),
            calmar=risk.get("calmar_ratio", 0),
            max_drawdown=risk.get("max_drawdown", max_dd_avg),
            max_drawdown_duration=int(risk.get("max_drawdown_duration", 0)),
            total_return=risk.get("total_return", total_return_avg),
            annual_return=risk.get("annual_return", 0),
            annual_vol=risk.get("annual_vol", 0),
            var_95=risk.get("var_95", 0),
            cvar_95=risk.get("cvar_95", 0),
            win_rate=trade.get("win_rate", 0),
            profit_factor=trade.get("profit_factor", 0),
            total_trades=trade.get("total_trades", 0),
            timestamp=datetime.now().isoformat(),
            git_commit=_get_git_commit(),
            config_hash=_compute_config_hash(),
            pbo=round(pbo, 4),
            dsr=round(dsr, 4),
            num_trials=num_trials,
            cost_model_snapshot=_cost_snapshot,
        )

        # ── P2.3 回测结果持久化写入 DB（DB 不可用时静默跳过）──
        try:
            from BackTrading.backtest_persistor import BacktestPersistor
            _run_id = f"BT_{bt.BACKTEST_START_DATE}_{cal_result.config_hash[:12]}"
            _persistor = BacktestPersistor(engine)
            _metrics_db = {
                "total_return": cal_result.total_return,
                "sharpe_ratio": cal_result.sharpe,
                "max_drawdown": cal_result.max_drawdown,
                "win_rate": cal_result.win_rate,
            }
            _persistor.persist_run(
                run_id=_run_id,
                params=best_params,
                trade_log=trade_log,
                equity_curve=equity_curve,
                metrics=_metrics_db,
                strategy_name=getattr(bt, "STRATEGY_NAME", "default"),
            )
        except Exception as _e:
            logger.warning(f"[P2.3] 回测结果持久化失败（不影响主流程）: {_e}")

        # ── 统一参数采纳门控（P0-5 审计修复） ──
        # save_calibration（calibration_result.json）与 write_calibration_to_ini
        # （生产 config.ini）共用同一门控：统计显著性 + OOS 衰减 + PBO/DSR 硬性
        # 拒绝 + 多重测试惩罚 + 稳健性 + 模拟验证，杜绝门控不一致导致 PBO 过拟合
        # 参数集仍落盘进生产。
        _pbo_gate = pbo <= 0.05
        _dsr_gate = dsr >= 0.5
        if not _pbo_gate:
            logger.warning(
                f"[过拟合防护] PBO={pbo:.4f} > 0.05 阈值，参数组统计显著性不足，拒绝采纳"
            )
        if not _dsr_gate:
            logger.warning(
                f"[过拟合防护] DSR={dsr:.4f} < 0.5 阈值，缩水 Sharpe 比过低，拒绝采纳"
            )
        _gate_pass, _gate_reasons = _acceptance_gate(
            promote=_promote,
            oos_decay_pass=_oos_decay_pass,
            overfitting_critical=_overfitting_critical,
            sig_pass=_sig_pass,
            robust_pass=_robust_pass,
            pbo_gate=_pbo_gate,
            dsr_gate=_dsr_gate,
        )
        if _gate_pass:
            save_calibration(cal_result)
        else:
            logger.warning("=" * 50)
            logger.warning(
                "[采纳门控] 参数组未通过统一采纳门控，calibration_result.json 不予保存: "
                + "；".join(_gate_reasons)
            )
            logger.warning("=" * 50)

        # ── 多策略组合回测 ──
        _enable_ms = getattr(bt, "MULTI_STRATEGY_ENABLED", False)
        if _enable_ms:
            try:
                from BackTrading.multi_strategy import run_multi_strategy_backtest as _rms
                _ms_result = _rms(kline_df, ecfg, best_params, trade_log, equity_curve)
                logger.info(f"  多策略组合完成: {len(_ms_result)} 个子策略")
            except Exception as e:
                logger.warning(f"  多策略组合回测异常: {e}")

        # ── 压力测试 ──
        try:
            from BackTrading.stress_test import run_stress_tests as _rst
            _stress_results = _rst(kline_df, ecfg, best_params)
            _worst_dd = min((r.get("max_drawdown", 0) for r in _stress_results.values()), default=0)
            if _worst_dd < -0.3:
                logger.warning(f"  压力测试: 历史极端场景最大回撤 {_worst_dd:.2%} > 30%，建议评估风险")
        except Exception as e:
            logger.warning(f"  压力测试异常: {e}")

        # ── 涨跌停专项压力测试（技术债修复：一字涨停/竞价触板/炸板 worst-case 成本） ──
        try:
            if bool(getattr(bt, "LIMIT_STRESS_ENABLED", True)):
                from BackTrading.limit_stress import run_limit_stress as _rls
                _limit_stress = _rls(kline_df, ecfg, best_params)
                _wc = _limit_stress.get("worst_case", {})
                logger.info(
                    f"  涨跌停压力: 买入成交率最低={_wc.get('min_buy_fill_rate', 1):.2%} "
                    f"卖出成交率最低={_wc.get('min_sell_fill_rate', 1):.2%} "
                    f"买入未成交敞口最大={_wc.get('max_unfilled_buy_value', 0):.0f}元 "
                    f"卖出未成交敞口最大={_wc.get('max_unfilled_sell_value', 0):.0f}元"
                )
        except Exception as e:
            logger.warning(f"  涨跌停压力测试异常: {e}")

        if _gate_pass:
            # ── VAEO：波动率自适应退出参数学习 ──
            try:
                from BackTrading.replay_optimizer import optimize_vol_exits
                learned_t1, learned_t2 = optimize_vol_exits(
                    trade_log, final_prepared, 
                    sl_mult=best_params.get("atr_stop_mult", 1.5)
                )
                best_params["learned_t1_mult"] = round(learned_t1, 2)
                best_params["learned_t2_mult"] = round(learned_t2, 2)
                logger.info(f"[VAEO] 学习完成: T1={learned_t1:.2f}, T2={learned_t2:.2f}，将写入配置闭环")
            except Exception as e:
                logger.warning(f"[VAEO] 退出策略优化失败: {e}，回测结果将沿用默认止盈参数")
                
            write_calibration_to_ini(best_params)
            apply_calibration_to_config(config)
            logger.info("模拟验证通过，参数已写入 config.ini 并生效")
        else:
            # P0-5：同一统一门控——任一关键项未通过，config.ini 保持不变
            logger.warning("=" * 50)
            logger.warning(
                "[采纳门控] config.ini 参数保持不变（结果可作回测报告参考，已记录数据库）: "
                + "；".join(_gate_reasons)
            )
            logger.warning("=" * 50)
            # 仍将结果写入数据库用于历史追踪

        # P2 防御性过滤：从记录参数中剔除运行时依赖对象（Engine、DataFrame），
        # 防止不可序列化类型进入 json.dumps。_pyval 兜底已能处理，但源头清理更优。
        _runtime_keys = {"_db_engine", "_st_history", "_listing_days"}
        clean_params = {k: v for k, v in best_params.items() if k not in _runtime_keys}

        record_run(
            engine=engine,
            frequency=bt.OPTIMIZE_FREQUENCY,
            backtest_start_date=bt.BACKTEST_START_DATE,
            out_of_sample_days=bt.OUT_OF_SAMPLE_DAYS,
            initial_cash=bt.INITIAL_CASH,
            params=clean_params,
            sharpe=cal_result.sharpe,
            total_return=cal_result.total_return,
            max_drawdown=cal_result.max_drawdown,
            extra_metrics=risk | trade | {"pbo": cal_result.pbo, "dsr": cal_result.dsr, "num_trials": cal_result.num_trials},
            git_commit=cal_result.git_commit,
            config_hash=cal_result.config_hash,
            data_version=_data_version,
        )

        updated_sections = set()
        for k in best_params:
            if k in CALIB_PARAM_MAP:
                updated_sections.add(CALIB_PARAM_MAP[k][0])
        if _gate_pass:
            logger.info(f"  寻优结果已采纳并写入 calibration_result.json + config.ini [{', '.join(sorted(updated_sections))}]")
        else:
            logger.info("  寻优结果未通过统一采纳门控，calibration_result.json / config.ini 未写入")
        alert.on_success(cal_result)
        return cal_result

    except Exception as exc:
        logger.opt(exception=True).error(f"回测管线失败: {exc}")
        # A2：管线级兜底快照（窗口级快照由 meta_optimizer 内部落盘）
        import traceback as _tb

        _snap_id = save_failure_snapshot(
            ohlcv=kline_df if kline_df is not None and not kline_df.empty else None,
            metric_name="pipeline",
            error_code="PIPELINE_FAILED",
            error_message=str(exc),
            traceback_text=_tb.format_exc(),
        )
        if _snap_id:
            logger.error(f"回测管线失败快照已保存 | snapshot_id={_snap_id} | run_id={_run_id}")
        try:
            record_run(
                engine=engine,
                frequency=bt.OPTIMIZE_FREQUENCY,
                backtest_start_date=bt.BACKTEST_START_DATE,
                out_of_sample_days=bt.OUT_OF_SAMPLE_DAYS,
                initial_cash=bt.INITIAL_CASH,
                params={},
                sharpe=0,
                total_return=0,
                max_drawdown=0,
                status="failed",
                data_version=_data_version,
            )
        except Exception as log_err:
            logger.warning(f"回测失败记录写入异常: {log_err}")
        alert.on_failure(exc, snapshot_id=_snap_id)
        return None
    finally:
        _release_run_lock()


def _compute_kline_data_version(engine: Any, kline_df: pd.DataFrame | None = None) -> str:
    """数据版本标识：用于信号缓存隔离与 calibration_log 调度。

    优先从内存 DataFrame 计算（runner 主路径），消除 fetch→version 之间的
    竞态窗口（IncrementalSyncEngine 不回听 advisory lock，可能并发写入）；
    无 DataFrame 时回退到数据库查询（should_rerun 调度场景）。

    版本仅使用 max(trade_date) 作为粗粒度失效信号：
    - COUNT 已移除：新增股票全量历史数据会剧烈变动 COUNT，但存量 OHLC
      内容不变，导致不必要的整库重跑；细粒度内容变更由
      _data_fingerprint 的 OHLCV 采样 hash + 行数覆盖。
    - 仅 max(trade_date) 变动才失效：新交易日到达即触发重跑，符合业务直觉。
    """
    # 路径 ①：从内存 DataFrame 计算（与消费数据严格一致，无竞态）
    if kline_df is not None and not kline_df.empty and "trade_date" in kline_df.columns:
        try:
            _max_date = kline_df["trade_date"].max()
            # 安全归一化为 ISO 日期字符串
            _ts = pd.Timestamp(_max_date)
            return _ts.strftime("%Y-%m-%d")
        except Exception as exc:
            logger.warning(f"从 DataFrame 计算数据版本失败: {exc}")

    # 路径 ②：从数据库查询（should_rerun 调度场景，无 DataFrame）
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT COALESCE(MAX(trade_date)::text, '') FROM stock_daily_kline"
            )).fetchone()
        if row is None or not row[0]:
            return ""
        return str(row[0])
    except Exception as exc:
        logger.warning(f"计算 kline 数据版本失败: {exc}")
        return ""


def _fallback_universe(engine: Any) -> set[str]:
    """从零启动时解析股票池的兜底全市场名单（返回 6 位纯数字代码集合）。

    背景：股票池自引用自 stock_daily_kline ∪ stock_st_history。从零启动
    （清库重拉 / 全新环境）时 K 线表为空，池子塌缩至 st_history 残余，
    导致无法全量下载数据。此兜底链恢复完整股票池：

      ① stock_basic_info_sw：日常复盘管线维护的全市场名单（离线、稳定）
      ② AkShare stock_info_a_code_name：在线沪深 A 股全名单（网络失败优雅降级）

    已退市股票不在上述来源中，仍依赖 st_history 残余记录并入池。
    主板过滤与市场前缀由 _resolve_symbols 统一处理。
    """
    from UtilsManager.CodeNormalizer import CodeNormalizer

    syms: set[str] = set()
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT stock_code FROM stock_basic_info_sw")
            ).fetchall()
        syms = {CodeNormalizer.normalize(r[0]) for r in rows}
        syms.discard("")
    except Exception as e:
        logger.warning(f"  兜底源① stock_basic_info_sw 不可用: {e}")
    if syms:
        logger.info(f"  兜底源① stock_basic_info_sw 提供 {len(syms)} 只全市场代码")
        return syms

    try:
        import akshare as ak

        df = ak.stock_info_a_code_name()
        col = "代码" if "代码" in df.columns else df.columns[0]
        syms = {CodeNormalizer.normalize(c) for c in df[col]}
        syms.discard("")
        logger.info(f"  兜底源② AkShare stock_info_a_code_name 提供 {len(syms)} 只全市场代码")
    except Exception as e:
        logger.warning(f"  兜底源② AkShare 全市场名单拉取失败（股票池可能不完整）: {e}")
    return syms


def _resolve_symbols(engine: Any, config: Config | None = None) -> list[str]:
    """解析股票列表，仅保留沪深主板（60x/00x 开头）。

    为消除生存者偏差，股票池包含所有曾有过交易记录的股票（含已退市）。
    ST/*ST/退市的逐日动态剔除由引擎配合 stock_st_history 完成，此处不做静态剔除。
    系统仅覆盖沪深主板，创业板/科创板/北交所已从业务中剔除。

    从零启动兜底：stock_daily_kline 为空时股票池自引用失效（仅剩 st_history
    残余），自动回落 _fallback_universe 全市场名单，保证首跑即全量下载。
    """
    from UtilsManager.CodeNormalizer import CodeNormalizer

    with engine.connect() as conn:
        kline_syms = {
            str(r[0]) for r in
            conn.execute(text("SELECT DISTINCT symbol FROM stock_daily_kline")).fetchall()
        }
        try:
            st_syms = {
                str(r[0]) for r in
                conn.execute(text("SELECT DISTINCT symbol FROM stock_st_history")).fetchall()
            }
        except Exception:
            # 全新数据库：stock_st_history 尚未创建（ensure_st_history_table 在管线后段执行）
            st_syms = set()

    raw = sorted(kline_syms | st_syms)
    if not kline_syms:
        # 从零启动：K 线表为空 → 池子自引用失效，回落全市场名单
        fallback = _fallback_universe(engine)
        added = len(fallback - set(raw))
        if added:
            logger.info(
                f"  stock_daily_kline 为空，启用全市场兜底名单 +{added} 只"
                f"（{len(raw)} → {len(raw) + added}）"
            )
            raw = sorted(set(raw) | fallback)
    # 硬编码主板过滤：仅保留 60x / 00x 开头代码
    before = len(raw)
    raw = [s for s in raw if s.replace("sh", "").replace("sz", "").startswith(("60", "00"))]
    if len(raw) < before:
        logger.info(f"主板过滤后剩余: {len(raw)} / {before} 只")

    # FIX(P0): 主板池子不足时自动兜底补全全市场名单
    # 旧逻辑仅在 kline_syms 为空时走 _fallback_universe；但表中有少量残留数据
    #   时兜底链被跳过 → 用局部数据（如 105 只）跑全部回测，结果无意义。
    # 新逻辑：主板过滤后低于 MIN_MAIN_BOARD 阈值，自动合并全市场兜底名单补齐。
    _MIN_MAIN_BOARD = 200
    if len(raw) < _MIN_MAIN_BOARD:
        logger.warning(
            f"  主板股票池仅 {len(raw)} 只（低于门槛 {_MIN_MAIN_BOARD}），"
            "自动启用全市场兜底名单补齐..."
        )
        fallback = _fallback_universe(engine)
        # 兜底名单也做主板过滤
        fallback_main = [
            s for s in fallback
            if s.replace("sh", "").replace("sz", "").startswith(("60", "00"))
        ]
        missing = set(fallback_main) - set(raw)
        if missing:
            raw = sorted(set(raw) | missing)
            logger.info(
                f"  全市场兜底补齐主板 +{len(missing)} 只 → 总计 {len(raw)} 只"
            )
        else:
            logger.warning(
                f"  兜底名单经主板过滤后未带来新增（可能全市场数据本身不完整）"
            )
    # P1 防御性断言：即使补齐后仍低于阈值，给出严重警告但不阻断（避免在线环境断网死锁）
    if len(raw) < _MIN_MAIN_BOARD:
        logger.warning(
            f"  [数据完整性告警] 主板股票池仅 {len(raw)} 只，回测结果可能不可靠。"
            "请检查 stock_basic_info_sw 表或 AkShare 在线接口是否正常。"
        )

    # 注意：不再做静态 ST 剔除，退市整理期/摘牌日由引擎无条件强平
    # ST 过滤已在复盘单元（Review/coordinator.py）硬编码执行，不复归 config 控制
    if not raw:
        logger.warning("回测股票池为空，请检查数据库 stock_daily_kline 表")
    # 二次校验：确保输出池不含非主板代码（300/688/8xx/4xx）
    _non_main = [
        s for s in raw
        if not s.replace("sh", "").replace("sz", "").startswith(("60", "00"))
    ]
    if _non_main:
        logger.warning(
            f"  检测到 {len(_non_main)} 只非主板代码被误纳入股票池 "
            f"（如 {_non_main[:5]}），自动过滤"
        )
        raw = [s for s in raw if s not in set(_non_main)]

    return sorted({CodeNormalizer.add_market_prefix(s) if not s.startswith(("sh", "sz")) else s for s in raw})


def _load_st_history(engine: Any, symbols: list[str], start_date: str, end_date: str) -> dict[str, dict[str, tuple[bool, bool]]]:
    """
    加载股票在日期范围内的 ST/退市状态历史（PIT 逐日序列）。

    P0-5 审计修复：
      - SQL 注入：旧实现字符串插值拼接 symbol 进 SQL（sym_placeholders），
        已改为 DataManager.StPitSync.load_st_pit 的参数化 = ANY(:syms)。
      - 非 PIT：旧表常为最近快照，历史 ST 期缺失导致 5% 涨跌幅被错按 10%、
        ST 禁买/强平失效；数据由 sync_st_pit 回填的全历史 PIT 序列提供。

    Returns:
        dict: {symbol: {trade_date: (is_st, is_delisting)}}
    """
    from DataManager.StPitSync import load_st_pit

    return load_st_pit(engine, symbols, start_date, end_date)


def _load_listing_days(engine: Any, symbols: list[str], start_date: str) -> dict[str, str]:
    """
    加载股票上市日期（显式注入 IPO 日期，P0-6 ④）。

    P0-6 审计修复：引擎不再从行情数据推断上市日期（数据缺口会误判新股，
    错误激活"注册制前 5 日无涨跌幅"豁免）；上市日期由 stock_listing_days 表
    （AkShare stock_info_a_code_name 上市日期列）提供，缺失时豁免整体停用。
    """
    from DataManager.ListingDaysSync import load_listing_days

    return load_listing_days(engine, symbols, start_date)


def _fetch_kline(
    engine: Any,
    symbols: list[str],
    backtest_start_date: str,
) -> pd.DataFrame:
    from DataManager.sync import ensure_table
    from DataManager.IncrementalSyncEngine import IncrementalSyncEngine

    # 将配置日期对齐到首个交易日，与 IncrementalSyncEngine 内部逻辑一致
    aligned_start = IncrementalSyncEngine.align_to_trading_day(backtest_start_date)

    ensure_table(engine)

    # 补齐缺失股票的历史 K 线
    _sync_missing_stocks(engine, symbols, aligned_start)

    # P3-5 审计修复(P0): 同步退市股历史K线（消除生存偏差）
    # 获取已同步的退市股symbols集，后续合并到K线查询列表
    _delisted_syms, _delisted_synced = _sync_delisted_stocks(engine, backtest_start_date)

    end = date.today()
    start = datetime.strptime(aligned_start, "%Y%m%d").date()

    # 前拉缓冲期确保技术指标充分预热（MACD/ATR/MA等需至少 120 个交易日）
    _buffer_trading_days = 180
    _buffer_calendar_days = _buffer_trading_days * 2
    buffer_start = (start - timedelta(days=_buffer_calendar_days)).isoformat()

    # 合并回测symbols与退市股symbols，确保退市股K线被一并读取
    _query_symbols = sorted(set(symbols) | {_s for _s in _delisted_syms if isinstance(_s, str)})

    provider = BacktestDataProvider(engine)
    df: pd.DataFrame = provider.get_kline(_query_symbols, start_date=buffer_start, end_date=end.isoformat())
    if df.empty:
        return df, _delisted_synced
    df = df.sort_values(["symbol", "trade_date"])
    return df, _delisted_synced


def _sync_missing_stocks(engine: Any, symbols: list[str], backtest_start_date: str) -> None:
    """补齐 + 刷新 stock_daily_kline 数据。检查每只股票数据是否齐全，检测除权除息并重拉。

    同时执行一次性"指标预热回填"：已有数据但最早交易日晚于预热起点的股票，
    强制从预热起点回填历史 K 线（MACD/ATR/MA 等指标至少需要 120 个交易日前文）。
    """
    from DataManager.IncrementalSyncEngine import IncrementalSyncEngine

    start = datetime.strptime(backtest_start_date, "%Y%m%d").date()
    _buffer_calendar_days = 360
    buffer_start_iso = (start - timedelta(days=_buffer_calendar_days)).isoformat()

    syncer = IncrementalSyncEngine(engine, default_start=backtest_start_date)

    # 检查哪些股票完全缺失
    with engine.connect() as conn:
        existing = {
            r[0] for r in
            conn.execute(text("SELECT DISTINCT symbol FROM stock_daily_kline")).fetchall()
        }
    missing = [s for s in symbols if s not in existing]
    if missing:
        logger.info(f"  stock_daily_kline 缺少 {len(missing)} 只股票，开始补齐...")
        n = syncer.sync_all(missing, force_start_iso=buffer_start_iso)
        logger.info(f"  补齐完成，新增 {n} 行")

    # 对已有数据的股票执行增量刷新：检查最新日期、除权除息检测
    existing_symbols = [s for s in symbols if s not in missing]
    if existing_symbols:
        logger.info(f"  检查 {len(existing_symbols)} 只股票数据完整性...")
        total = syncer.sync_all(existing_symbols)
        logger.info(f"  刷新完成，新增 {total} 行")

    # 一次性指标预热回填：数据起点晚于预热起点的股票（缺早期历史，指标前文不足）
    if existing_symbols:
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT symbol, MIN(trade_date) AS first_d FROM stock_daily_kline "
                    "GROUP BY symbol"
                )).fetchall()
            first_by_symbol = {r[0]: r[1] for r in rows}
            # 容差 10 个日历日：预热起点恰逢非交易日/停牌时最早数据略晚于起点属正常，
            # 避免回填完成后因 1-2 天边界差反复触发全市场强制回填
            _warmup_tolerance = 10
            _warmup_cutoff = (
                pd.Timestamp(buffer_start_iso) + timedelta(days=_warmup_tolerance)
            ).strftime("%Y-%m-%d")
            need_warmup = [
                s for s in existing_symbols
                if s in first_by_symbol and first_by_symbol[s] is not None
                and pd.Timestamp(first_by_symbol[s]).strftime("%Y-%m-%d") > _warmup_cutoff
            ]
            if need_warmup:
                logger.info(
                    f"  {len(need_warmup)} 只股票历史不足 {buffer_start_iso}（指标预热），"
                    f"强制回填中（示例: {need_warmup[:5]}）..."
                )
                w_total = syncer.sync_all(need_warmup, force_start_iso=buffer_start_iso)
                logger.info(f"  预热回填完成，新增 {w_total} 行")
        except Exception as e:
            logger.warning(f"  预热回填失败（回测继续，指标前文可能不足）: {e}")


def _sync_delisted_stocks(engine: Any, backtest_start_date: str) -> tuple[set[str], int]:
    """P3-5 审计修复(P0)：同步退市股历史K线，消除生存偏差。

    从 AkShare 沪深交易所终止上市列表获取退市股代码，
    过滤主板（60x/00x），对缺失K线的退市股强制回填历史数据。

    网络失败/无可利用退市列表时优雅降级（仅告警不阻断）。

    Returns:
        (退市股符号集合, 实际同步写入行数) — 供生存偏差报告与覆盖率门禁使用。
    """
    try:
        from DataManager.IncrementalSyncEngine import IncrementalSyncEngine
        from DataManager.sync import ensure_table
        from UtilsManager.CodeNormalizer import CodeNormalizer
    except ImportError as e:
        logger.warning(f"  退市股同步模块导入失败: {e}")
        return set(), 0

    # 获取退市股代码列表
    delisted_set = _fetch_extended_delisted()
    if not delisted_set:
        logger.info("  退市股列表为空或拉取失败，跳过生存偏差修复")
        return set(), 0

    # 仅保留主板退市股（60x/00x），与 _resolve_symbols 主板过滤保持一致
    main_board_delisted = {
        s for s in delisted_set
        if s.startswith(("60", "00"))
    }
    if not main_board_delisted:
        logger.info("  无主板退市股，跳过生存偏差修复")
        return set(), 0

    # 检查哪些退市股已存在 K 线
    with engine.connect() as conn:
        existing = {
            r[0] for r in
            conn.execute(text("SELECT DISTINCT symbol FROM stock_daily_kline")).fetchall()
        }

    missing_delisted = sorted(main_board_delisted - existing)
    if not missing_delisted:
        logger.info(f"  池内退市股 K 线已齐全 ({len(main_board_delisted)} 只)，无需补充同步")
        return main_board_delisted, 0

    logger.info(f"  P0生存偏差修复: {len(missing_delisted)} 只主板退市股缺失 K 线，开始回填历史数据...")
    try:
        ensure_table(engine)
        start = datetime.strptime(backtest_start_date, "%Y%m%d").date()
        buffer_start_iso = (start - timedelta(days=360)).isoformat()

        syncer = IncrementalSyncEngine(engine, default_start=backtest_start_date)
        synced_count = syncer.sync_all(missing_delisted, force_start_iso=buffer_start_iso)
        logger.info(
            f"  退市股 K 线回填完成: {len(missing_delisted)} 只退市股, 写入 {synced_count} 行"
        )
    except Exception as e:
        logger.warning(f"  退市股 K 线同步失败（回测继续，可能存在生存偏差）: {e}")
        return set(), 0

    return main_board_delisted, synced_count


def _fetch_extended_delisted() -> set[str] | None:
    """从 AkShare 拉取深/沪终止上市股票代码（带市场前缀符号集）。

    独立于 stock_st_history PIT 表——PIT 同步失败不应导致退市股检测失灵。
    任一交易所源成功即返回，全部失败返回 None。
    """
    from UtilsManager.CodeNormalizer import CodeNormalizer

    out: set[str] = set()
    ok = False

    # 深交所终止上市公司
    try:
        import akshare as ak
        df_sz = ak.stock_info_sz_delist("终止上市公司")
        if df_sz is not None and not df_sz.empty:
            code_col_sz = "证券代码" if "证券代码" in df_sz.columns else "公司代码"
            if code_col_sz in df_sz.columns:
                ok = True
                out |= {CodeNormalizer.normalize(str(v)) for v in df_sz[code_col_sz].tolist()}
    except Exception as e:
        logger.warning(f"[生存偏差] stock_info_sz_delist 拉取失败: {e}")

    # 上交所终止上市公司
    try:
        import akshare as ak
        df_sh = ak.stock_info_sh_delist("全部")
        if df_sh is not None and not df_sh.empty:
            code_col_sh = "证券代码" if "证券代码" in df_sh.columns else "公司代码"
            if code_col_sh in df_sh.columns:
                ok = True
                out |= {CodeNormalizer.normalize(str(v)) for v in df_sh[code_col_sh].tolist()}
    except Exception as e:
        logger.warning(f"[生存偏差] stock_info_sh_delist 拉取失败: {e}")

    return out if ok else None


def _extract_oos_returns(wf_result: pd.DataFrame) -> np.ndarray | None:
    """从 WFO 结果提取 rank-1 组合的 OOS 日收益序列（逐窗口内拼接）。

    每窗口的 oos_equity 独立以 initial_cash 起步，窗口边界净值有重置跳变，
    因此按窗口内计算日收益、跨窗口拼接收益（丢弃窗口边界收益）。
    无 oos_combos 列或数据不足时返回 None（validate_params 回退 iid 近似并告警）。
    """
    if wf_result is None or wf_result.empty or "oos_combos" not in wf_result.columns:
        return None
    rets: list[float] = []
    for row in wf_result["oos_combos"]:
        if not row:
            continue
        rank1 = next((c for c in row if c.get("is_rank") == 1), row[0])
        ec = rank1.get("oos_equity") if isinstance(rank1, dict) else None
        if not ec:
            continue
        vals = np.asarray(
            [float(e["portfolio_value"]) for e in ec
             if e.get("portfolio_value") is not None and np.isfinite(float(e["portfolio_value"]))],
            dtype=float,
        )
        if len(vals) < 2 or vals[0] <= 0:
            continue
        r = vals[1:] / vals[:-1] - 1.0
        rets.extend(r[np.isfinite(r)].tolist())
    if len(rets) < 10:
        return None
    return np.asarray(rets, dtype=float)


def _extract_best_params(wf_result: pd.DataFrame, top_n: int = 5, config: Config | None = None) -> dict[str, float]:
    """
    从 Walk-Forward 结果中提取最佳参数。

    主路径（P2.3 稳健中位数）：优先取 DM 检验显著通过（p<0.05）且
    OOS Sharpe>0 的窗口参数中位数——单窗口 Sharpe 尖峰多为噪声，
    中位数对离群窗口稳健；DM 显著过滤保证"寻优确实优于基准"。
    兜底路径：DM 数据缺失时退化为"OOS 为正窗口的中位数"；
    窗口不足时退回原 Sharpe 加权 Top-N 均值；仍失败则用配置中位数。

    如果提取失败（数据不足、Sharpe 全为 NaN/负值、params 列缺失等），
    返回配置中的默认参数中位数作为兜底，并记录警告。
    """
    # 默认兜底参数（从配置区间取中位数）
    def _fallback_params(cfg: Config | None) -> dict[str, float]:
        if cfg is None:
            return {
                "atr_stop_mult": 2.5,
                "boll_narrow_ratio": 0.9,
                "cross_decay_days": 37,
                "conclusion_full_bull": 80,
                "golden_cross_bonus": 10,
                "divergence_penalty": 20,
                "buy_threshold": 12,
                "max_holdings": 5,
            }
        bt = cfg.app_config.backtest
        return {
            "atr_stop_mult": sum(bt.parse_range("ATR_STOP_MULT_RANGE")[:2]) / 2,
            "boll_narrow_ratio": sum(bt.parse_range("BOLL_NARROW_RATIO_RANGE")[:2]) / 2,
            "cross_decay_days": sum(bt.parse_range("CROSS_DECAY_DAYS_RANGE")[:2]) / 2,
            "conclusion_full_bull": sum(bt.parse_range("CONCLUSION_FULL_BULL_RANGE")[:2]) / 2,
            "golden_cross_bonus": sum(bt.parse_range("GOLDEN_CROSS_BONUS_RANGE")[:2]) / 2,
            "divergence_penalty": sum(bt.parse_range("DIVERGENCE_PENALTY_RANGE")[:2]) / 2,
            # P0-7 ②：组合参数兜底优先取校准覆写值（与日频路径 EngineConfig 一致）
            "buy_threshold": bt.BUY_THRESHOLD,
            "max_holdings": bt.MAX_HOLDINGS,
        }

    if wf_result.empty or "params" not in wf_result.columns:
        logger.warning("Walk-Forward 结果为空或缺少 params 列，使用配置中位数作为兜底参数")
        return _fallback_params(config)

    rows = wf_result.dropna(subset=["sharpe_ratio"])
    if rows.empty:
        logger.warning("Walk-Forward 所有组合 Sharpe 均为 NaN，使用配置中位数作为兜底参数")
        return _fallback_params(config)

    def _median_params(rows_: pd.DataFrame) -> dict[str, float]:
        """对行内 params 取逐参数中位数（稳健主路径核心）。"""
        all_params = [r["params"] for _, r in rows_.iterrows() if isinstance(r["params"], dict)]
        if not all_params:
            return {}
        keys = all_params[0].keys()
        median_params: dict[str, float] = {}
        for k in keys:
            vals = sorted(p[k] for p in all_params)
            median_params[k] = vals[len(vals) // 2]
        return median_params

    # ── 主路径：DM 显著通过（p<0.05）且 OOS>0 的窗口 → 参数中位数 ──
    if "dm_p_value" in rows.columns:
        dm_rows = rows[
            (rows["dm_p_value"] < 0.05) & (rows["sharpe_ratio"] > 0)
        ]
        if len(dm_rows) >= 2:
            med = _median_params(dm_rows)
            if med:
                logger.info(
                    f"[稳健中位数主路径] DM 显著窗口 {len(dm_rows)} 个 "
                    f"(p<0.05 且 OOS>0)，取参数中位数: {med}"
                )
                return med
        logger.warning(
            f"DM 显著窗口仅 {len(dm_rows)} 个(<2)，退化到 OOS 正收益窗口中位数"
        )

    # ── 次级路径：OOS Sharpe>0 窗口的参数中位数（无 DM 列时直接走这里） ──
    pos_rows = rows[rows["sharpe_ratio"] > 0]
    if len(pos_rows) >= 2:
        med = _median_params(pos_rows)
        if med:
            logger.info(f"[稳健中位数] OOS 正收益窗口 {len(pos_rows)} 个，取参数中位数: {med}")
            return med

    # ── 兜底：原 Sharpe 加权 Top-N 均值 ──
    rows = rows.sort_values("sharpe_ratio", ascending=False).head(top_n)
    weights = rows["sharpe_ratio"].values
    total_weight = weights.sum()
    if total_weight <= 0:
        logger.warning("Walk-Forward Top-N 组合 Sharpe 权重和 <= 0，使用配置中位数作为兜底参数")
        return _fallback_params(config)

    all_params: list[dict[str, float]] = []
    for _, r in rows.iterrows():
        if isinstance(r["params"], dict):
            all_params.append({k: float(v) for k, v in r["params"].items()})

    if not all_params:
        logger.warning("Walk-Forward params 列无有效 dict，使用配置中位数作为兜底参数")
        return _fallback_params(config)

    keys = all_params[0].keys()
    weighted: dict[str, float] = {}
    for k in keys:
        vals = [p[k] for p in all_params]
        weighted[k] = sum(v * w for v, w in zip(vals, weights)) / total_weight
    return weighted


def start_scheduler(config: Config | None = None) -> None:
    """启动定时调度（每日检查，按配置频率执行回测）。"""
    import time

    import schedule as _schedule

    if config is None:
        config = Config()

    bt = config.app_config.backtest
    if not bt.ENABLED:
        logger.info("回测未启用，调度器不启动")
        return

    engine = get_engine(config)
    ensure_table(engine)

    logger.info(f"启动回测调度器 (频率={bt.OPTIMIZE_FREQUENCY})")

    def job() -> None:
        logger.info("调度触发：检查回测条件 ...")
        tmp_engine = get_engine(config)
        last = get_last_run(tmp_engine)
        should_run, reason = should_rerun(last, bt.OPTIMIZE_FREQUENCY)
        if should_run:
            run_backtest_pipeline(config, force=True)
        else:
            logger.info(f"调度跳过: {reason}")

    _schedule.every().day.at("02:00").do(job)
    logger.info("  每日 02:00 检查回测条件")

    if bt.OPTIMIZE_FREQUENCY == "initial":
        logger.info("  optimize_frequency=initial，立即执行首次回测")
        run_backtest_pipeline(config, force=True)

    while True:
        _schedule.run_pending()
        time.sleep(3600)


def main() -> None:
    """CLI 入口。

    Usage:
        python -m BackTrading.runner            # 执行回测（交互式判断是否已过期）
        python -m BackTrading.runner --force     # 强制重新回测
        python -m BackTrading.runner --schedule  # 启动常驻调度器
    """
    args = sys.argv[1:]
    config = Config()

    if "--schedule" in args:
        start_scheduler(config)
        return

    force = "--force" in args
    result = run_backtest_pipeline(config, force=force)
    if result is None:
        sys.exit(0)
    logger.info(f"回测完成: Sharpe={result.sharpe:.2f}, Return={result.total_return:.2%}")


if __name__ == "__main__":
    main()
