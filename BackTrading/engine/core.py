from __future__ import annotations

from collections import deque
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from loguru import logger

from BackTrading.engine import EngineConfig
from BackTrading.calendar_align import get_official_calendar as _cal_get
from BackTrading.limit_pricing import (
    DELISTING_PERIOD_LIMIT_RATIO,
    MAIN_BOARD_FIRST_DAY_DOWN,
    MAIN_BOARD_FIRST_DAY_UP,
    MAIN_BOARD_LIMIT_RATIO,
    MAIN_BOARD_REFORM_DATE,
    ST_LIMIT_RATIO,
    auction_fill_ratio_for,
    listing_exempt_days,
    lot_size_for,
)
from BackTrading.domain.models import CostModel
# ── P4 组合优化器（数学规划驱动，替代 Top-K 等权） ──
from BackTrading.portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizerConfig,
    CovarianceEstimator,
)

ParamsDict: TypeAlias = dict[str, Any]
TradeLog: TypeAlias = list[dict[str, Any]]
EquityCurve: TypeAlias = list[dict[str, Any]]


# 20 日均量（行业口径：不含当日的滚动均值，避免用当日成交量前视）
_ADV_WINDOW = 20
# P0.3 修复：涨跌停可成交量fallback保守默认值（股）。
# 当日量开盘时不可知（前视）；前日无数据时（停牌复牌/新股首日）
# 使用 500手×100 = 50,000 股作为保守成交代理。
_DEFAULT_VOLUME_FALLBACK: float = 50_000.0


# 固定滑点强制下限（1.8 交易摩擦合规：A股隐性成本不低于单边 0.05%，
# 配置/回退路径任何低于此值的基础滑点一律抬升，防止 Alpha 虚高）
_MIN_SLIPPAGE_FLOOR = 0.0005
# ── 0.1 两档成交模型（P0-6 ②：A股订单时效与停牌废单语义） ──
# 档位 1「次日集合竞价委托」：信号日收盘挂单 → 次日 9:15-9:25 集合竞价撮合，
#   成交价 = 次日开盘价（开=集合竞价价）。订单仅在信号次日的集合竞价有效：
#   - 次日未成交（一字涨停/跌停封死）→ 撤销（A股订单当日有效，隔夜作废）
#   - 次日停牌（无行情/无成交量）→ 废单撤销（A股停牌日委托无效）
#   - 例外：强平单（ST/退市）遇一字跌停/停牌 → 逐日重挂直至成交（终态必须离场）
# 档位 2「盘中市价委托」：不在日频模型内主动建模盘中成交；触板日的成交率由
#   次日开盘集合竞价档近似（auction_fill_ratio_for，仅用 9:25 已知的 open/限价，
#   P1-2 前视修复）。盘中档 fill_ratio（依赖当日 close/high/low，属未来信息）
#   为死计算已删除（P2 审计）。
# 挂单过期天数（P0-6 ②：3 交易日 → 次日过期）：普通挂单信号次日未成交即撤销。
# P3-3（审计）：从硬编码提升为 EngineConfig.order_expiry_days（默认 1，保持原语义）。
import threading
from contextlib import contextmanager

# P3（审计修复）：全局告警标志使用 threading.local() 隔离会话
# 避免多窗口 WFO 或并行回测时标志跨会话污染。
_ENGINE_SESSION = threading.local()


def _round_half_up(x: float, ndigits: int = 2) -> float:
    """ROUND_HALF_UP 四舍五入（A 股 0.01 元最小报价单位口径）。

    Python round / np.round 为银行家舍入（round-half-even）：第三位小数恰为 5
    时舍向偶数。交易所（沪深交易规则）为 ROUND_HALF_UP——第三位恰为 5 一律进位，
    与 limit_pricing 涨跌停价口径一致（P2 审计修复：成交价/权益/日志金额统一）。
    Decimal(str(x)) 规避二进制浮点表示误差。非有限值（NaN/Inf）原样返回，
    由调用方过滤。
    """
    if not np.isfinite(x):
        return float(x)
    q = Decimal("0.01") if ndigits == 2 else Decimal(1).scaleb(-ndigits)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))
# P0-6 ⑤ 市场状态客观变量：市场收益窗口（日）与波动率分位窗口（交易日）
_REGIME_RET_WINDOW = 20
_REGIME_VOL_WINDOW = 250
# P0-11：市场波动率分位所需最小样本数（不足时返回中性分位，避免早期极端 regime）
_REGIME_VOL_MIN = 60


def _vol_percentile(hist, cur: float) -> float:
    """市场波动率在过去 _REGIME_VOL_WINDOW 个交易日内的分位（0..1）。

    P0-11：样本不足 _REGIME_VOL_MIN 时返回中性分位 0.5——回测早期（数据起点
    附近）分位由极少样本主导，易触发极端 regime 误判；中性分位使高波动判定
    让位于 ret20 趋势，待样本累积后恢复精确分位。
    """
    if len(hist) < _REGIME_VOL_MIN:
        return 0.5
    return sum(1.0 for v in hist if v <= cur) / len(hist)


def _valid_px_mask(px_arr: np.ndarray) -> np.ndarray:
    """成交价有效性掩码（P1-2 审计：统一卖出侧 NaN/非正价守卫）。

    买入侧已有守卫（`not isfinite(px) or px <= 0 → 重新排队`）；卖出侧
    三处路径（挂单撮合/复牌兑现/摘牌强平）必须对称——无效价格进入
    `cash += proc` 会把 NaN 传播到整条权益曲线。True=价格有效可成交。
    """
    return np.isfinite(px_arr) & (px_arr > 0)


def _regime_multiplier_for(mkt_ret20: float, vol_pct: float, engine_cfg: EngineConfig) -> float:
    """P0-6 ⑤ 市场状态仓位倍率（客观状态变量，替代"前日全市场评分中位数"）。

    Args:
        mkt_ret20: 指数 20 日收益代理（全市场后复权收盘 ret_20d 中位数）。
        vol_pct:   市场波动率分位（横截面日收益 std 在过去 250 交易日的分位）。

    规则：
        - mkt_ret20 ≥ regime_ret20_full → 全仓倍率
        - mkt_ret20 ≥ regime_ret20_half 且非高波动（vol_pct ≤ regime_vol_pct_max）
          → 半仓倍率
        - 其余（下跌趋势或高波动）→ 最低倍率
    """
    high_vol = vol_pct > engine_cfg.regime_vol_pct_max
    if mkt_ret20 >= engine_cfg.regime_ret20_full:
        return engine_cfg.regime_full_multiplier
    if mkt_ret20 >= engine_cfg.regime_ret20_half and not high_vol:
        return engine_cfg.regime_half_multiplier
    return engine_cfg.regime_min_multiplier


# ── P4 组合优化器辅助函数（数学规划驱动） ─────────────────────

# 行业映射缓存 (模块级，避免每次调仓重复查库)
_industry_cache: dict[str, str] = {}


def _build_returns_history(
    date_groups: list[tuple[Any, pd.DataFrame]],
    candidate_symbols: list[str],
    lookback: int = 60,
    end_index: int | None = None,
) -> pd.DataFrame:
    """从截至当日 T 的历史构建候选股票收益率矩阵 (T, n)。

    date_groups: [(date, 当日分组df), ...]，与引擎主循环同源（按日期有序）；
    end_index: 当前调仓日在 date_groups 中的索引（含当日）。
               P0-2 修复：只使用 date_groups[:end_index+1] —— 调仓日 T 的
               协方差只能基于 T 及之前的数据，杜绝"全样本尾部窗口"的未来
               数据泄露。None 时使用全部（仅用于外部诊断）。
    收益率优先用后复权 close_normal（跨除权日无机械跳变，与信号空间一致），
    缺失时回退不复权 close（不再用原始价污染协方差）。
    """
    if end_index is not None:
        usable = list(date_groups[: end_index + 1])
    else:
        usable = list(date_groups)
    usable.sort(key=lambda item: str(item[0]))
    recent = usable[-(lookback + 1):]

    close_col = None
    frames: list[pd.DataFrame] = []
    for dt_key, df_day in recent:
        if df_day is None or df_day.empty or "symbol" not in df_day.columns:
            continue
        if close_col is None:
            close_col = "close_normal" if "close_normal" in df_day.columns else "close"
        if close_col not in df_day.columns:
            continue
        mask = df_day["symbol"].isin(candidate_symbols)
        sub = df_day.loc[mask, ["symbol", close_col]].copy()
        if sub.empty:
            continue
        sub["trade_date"] = str(dt_key)[:10]
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    concat_df = pd.concat(frames, ignore_index=True)
    pivot = concat_df.pivot_table(
        index="trade_date", columns="symbol", values=close_col
    )
    if pivot.empty:
        return pd.DataFrame()

    returns = pivot.pct_change().dropna(how="all").tail(lookback)
    available_cols = [c for c in candidate_symbols if c in returns.columns]
    if not available_cols:
        return pd.DataFrame()
    return returns[available_cols]


def _load_industry_map_from_cache(
    candidate_symbols: list[str],
) -> dict[str, str]:
    """从模块级缓存返回行业映射 {symbol: industry}。

    缓存由 _refresh_industry_cache 在回测启动时填充。
    若缓存为空，返回空 dict（优化器会静默跳过行业约束）。
    """
    return {s: _industry_cache[s] for s in candidate_symbols if s in _industry_cache}


def _refresh_industry_cache(
    db_engine: Any,
    symbols: list[str] | None = None,
) -> None:
    """刷新行业映射缓存。

    从 stock_basic_info_sw_l1 表加载申万一级行业分类。
    失败时记录日志（不阻断回测）。
    """
    global _industry_cache
    if db_engine is None:
        return

    try:
        from sqlalchemy import text as sql_text

        with db_engine.connect() as conn:
            if symbols:
                # 仅加载需要的股票
                result = conn.execute(
                    sql_text(
                        "SELECT stock_code, l1_name FROM stock_basic_info_sw_l1 "
                        "WHERE stock_code IN :syms"
                    ),
                    {"syms": symbols},
                )
            else:
                result = conn.execute(
                    sql_text("SELECT stock_code, l1_name FROM stock_basic_info_sw_l1")
                )
            _industry_cache = dict(result.fetchall())
            logger.debug(f"[行业缓存] 加载 {len(_industry_cache)} 条映射")
    except Exception as e:
        logger.debug(f"[行业缓存] 加载失败: {e}（行业约束将跳过）")


# ── 回测引擎核心函数 ─────────────────────────────────────────


def run_full_backtest(
    data: pd.DataFrame,
    params: dict[str, Any],
    engine_cfg: EngineConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if engine_cfg is None:
        # 审计（成本外部化）：默认仍显式携带 CostModel（含逐笔最低佣金 + 分段表），
        # 但显式传入的 EngineConfig 若缺 cost_model 会在引擎内 fail-fast。
        from BackTrading.domain.models import CostModel

        engine_cfg = EngineConfig(cost_model=CostModel())
    tl: list[dict[str, Any]] = []
    ec: list[dict[str, Any]] = []
    _run_single_backtest(data, params, engine_cfg, tl, ec)
    return tl, ec


def _build_day_limit_model(
    syms_str: np.ndarray,
    close_raw: np.ndarray,
    high_arr: np.ndarray | None,
    low_arr: np.ndarray | None,
    prev_bar: dict[str, tuple[float, float]],
    st_syms: set[str],
    day_str: str,
    day_idx: dict[str, int],
    listing_map: dict[str, str] | None,
    streak: dict[str, int],
    sim_limits: bool,
    delist_first_syms: set[str] | None = None,
    delist_period_syms: set[str] | None = None,
    adj_factor_today: np.ndarray | None = None,
    prev_af: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """逐日涨跌停价建模（撮合约束，Task 涨跌停）。

    涨跌停价来自 BackTrading.limit_pricing（主板/创业板/科创板/北交所 + ST 5% +
    上市初期豁免 + 退市整理期）；无前收（数据首日）的标的按无限制处理（维持原行为）。
    P0-6 ① 退市整理期涨跌幅优先级（高于 ST 5%）：
        - 整理期首日（delist_first_syms）：无涨跌幅限制（±100% 近似）
        - 整理期其余日（delist_period_syms）：±10%
    P0-8（审计修复，2026-09-04）：涨跌停基准价统一使用交易所前日收盘价（不复权），
        不再使用 adj_factor 还原"除权除息参考价"。A股交易所涨跌停板价 = 前收 × (1±ratio)
        四舍五入到分，除权日不额外调整。原 exright_reference_price 修正导致分红日
        涨跌停价系统性偏差 = 每股分红/前收，一字板判定失真。
        修正后：涨跌停基准价 = prev_close_raw（不复权原始价），与交易所一致。

    Returns:
        (limit_up, limit_down, at_limit_up, at_limit_down, not_touched_up,
         not_touched_down, touched_up, touched_down, vol_mult, limit_tag)
        - at_limit_up/at_limit_down: 收盘封板口径（用于连板跟踪与一字板判定）
        - touched_up/touched_down:   盘中触板口径（0.3，high/low/open 触及限价即计，
            用于买卖限制；可成交量折算由撮合层 auction_fill_ratio_for 完成——
            P2 审计修复：盘中档 fill_ratio 依赖当日 close/high/low 属未来信息，
            且从未被消费（死计算），已删除不再计算）
        - vol_mult: 单日振幅>5% 剧烈波动日基础滑点翻倍倍率（1.9 流动性拟真）
        - limit_tag:  当日触板方向 "" / "up" / "down"
        - streak:     原地更新连续涨停(+) / 连续跌停(-) 板数
    """
    from BackTrading.limit_pricing import (
        calc_limit_prices_batch,
        limit_prices_for,
    )

    n = len(syms_str)
    limit_up = np.full(n, np.inf)
    limit_down = np.full(n, -np.inf)
    prev_close_arr = np.full(n, np.nan)

    # P2 审计修复：向量化涨跌停价批量计算
    # 第一步：收集有前收的 stock 并批量计算
    _valid_idx = []
    _pc_list = []
    _ratio_up_list = []
    _ratio_down_list = []
    _exempt_list = []

    for j, s in enumerate(syms_str):
        pc = prev_bar.get(s)
        if pc is None:
            continue  # 数据首日：无前收 → 豁免
        _valid_idx.append(j)
        _pc_raw = pc[0]  # A股交易所涨跌停基准价 = 前日收盘价（不复权），不使用 adj_factor 修正
        _pc_list.append(_pc_raw)

        # 确定涨跌幅比例（向量化：pre-compute ratio arrays）
        _ldays = None
        if listing_map:
            _fs = listing_map.get(s)
            if _fs is not None and _fs in day_idx:
                _ldays = max(1, day_idx[day_str] - day_idx[_fs] + 1)

        _exempt = False
        if _ldays is not None:
            _exempt = _ldays <= listing_exempt_days(day_str)

        # P0-6 ① 退市整理期涨跌幅（优先级最高：首日无限制 / 期间 ±10%，独立于 ST 5%）
        if delist_first_syms is not None and s in delist_first_syms:
            _ratio_up_list.append(1.0)
            _ratio_down_list.append(1.0)
        elif delist_period_syms is not None and s in delist_period_syms:
            _ratio_up_list.append(DELISTING_PERIOD_LIMIT_RATIO)
            _ratio_down_list.append(DELISTING_PERIOD_LIMIT_RATIO)
        elif _exempt:
            _ratio_up_list.append(1.0)
            _ratio_down_list.append(1.0)
        elif _ldays == 1 and (day_str is None or str(day_str)[:10] < MAIN_BOARD_REFORM_DATE):
            _ratio_up_list.append(MAIN_BOARD_FIRST_DAY_UP)
            _ratio_down_list.append(MAIN_BOARD_FIRST_DAY_DOWN)
        elif s in st_syms:
            _ratio_up_list.append(ST_LIMIT_RATIO)
            _ratio_down_list.append(ST_LIMIT_RATIO)
        else:
            _ratio_up_list.append(MAIN_BOARD_LIMIT_RATIO)
            _ratio_down_list.append(MAIN_BOARD_LIMIT_RATIO)

    if _valid_idx:
        _pc_arr = np.array(_pc_list, dtype=np.float64)
        _ru_arr = np.array(_ratio_up_list, dtype=np.float64)
        _rd_arr = np.array(_ratio_down_list, dtype=np.float64)
        _lu, _ld = calc_limit_prices_batch(_pc_arr, _ru_arr, _rd_arr)
        for k, idx in enumerate(_valid_idx):
            prev_close_arr[idx] = _pc_arr[k]
            limit_up[idx] = _lu[k]
            limit_down[idx] = _ld[k]

    # 收盘封板口径（连板跟踪 / 一字板判定）
    at_limit_up = close_raw >= limit_up - 1e-9
    at_limit_down = close_raw <= limit_down + 1e-9
    # 盘中触板口径（0.3）：high/open 触及涨停价 / low/open 触及跌停价即视为触板
    touched_up = at_limit_up.copy()
    touched_down = at_limit_down.copy()
    if high_arr is not None:
        touched_up = touched_up | (high_arr >= limit_up - 1e-9)
    if low_arr is not None:
        touched_down = touched_down | (low_arr <= limit_down + 1e-9)
    not_touched_up = ~touched_up
    not_touched_down = ~touched_down

    # 1.9 流动性拟真：单日振幅 = (high-low)/前收，>5% 剧烈波动日基础滑点翻倍
    # （无前收的首日回退当日收盘，与原行为一致；全无前收 → 不翻倍）
    vol_mult = np.ones(n, dtype=np.float64)
    if np.isfinite(prev_close_arr).any() and high_arr is not None and low_arr is not None:
        _pc_fb = np.where(np.isnan(prev_close_arr), close_raw, prev_close_arr)
        _amp = (high_arr - low_arr) / np.maximum(_pc_fb, 1e-9)
        vol_mult = np.where(_amp > 0.05, 2.0, 1.0)

    limit_tag = [""] * n
    if sim_limits:
        for j in range(n):
            if touched_up[j] or touched_down[j]:
                # P2 审计修复：盘中档 fill_ratio（fill_ratio_for，依赖当日
                # close/high/low）为死计算——撮合层仅消费次日开盘竞价档
                # auction_fill_ratio_for（P1-2 前视修复），且 _limit_fill 从未
                # 被 _flush_pending 使用，全市场逐股计算纯 CPU 浪费，已删除。
                limit_tag[j] = "up" if touched_up[j] else "down"

    # 连板跟踪（收盘后状态，供次日竞价档 auction_fill_ratio_for 连板衰减用）
    for j, s in enumerate(syms_str):
        cur = streak.get(s, 0)
        if at_limit_up[j]:
            streak[s] = cur + 1 if cur >= 0 else 1
        elif at_limit_down[j]:
            streak[s] = cur - 1 if cur <= 0 else -1
        else:
            streak[s] = 0

    return (
        limit_up, limit_down, at_limit_up, at_limit_down,
        not_touched_up, not_touched_down, touched_up, touched_down,
        vol_mult, limit_tag,
    )


def _run_single_backtest(
    data: pd.DataFrame,
    params: ParamsDict,
    engine_cfg: EngineConfig,
    trade_log: TradeLog,
    equity_curve: EquityCurve,
    stats_sink: dict[str, Any] | None = None,
) -> float:
    """单次回测。

    Args:
        stats_sink: 可选撮合统计收集器（涨跌停专项压力测试消费）。非 None 时
            引擎累计涨跌停撮合事件（竞价触板成交/部分/未成交、一字板撤销、
            未成交金额、worst-case 单日未成交敞口），键见 LIMIT_SINK_KEYS；
            None 时零开销（默认路径无行为变化）。

    P1-18：listing_days/st_history 通过 params["_listing_days"] /
    params["_st_history"] 注入（runner.py 在调用前注入），引擎内部
    从 params dict 读取。调用方（如 simulated_trading/limit_stress）
    需确保 params dict 含这些键。
    """
    # P3 审计：告警标志使用 _ENGINE_SESSION 隔离会话（见模块级 _ENGINE_SESSION 定义）
    if pd.api.types.is_datetime64_any_dtype(data["trade_date"]):
        data = data.copy()
        data["trade_date"] = data["trade_date"].dt.strftime("%Y-%m-%d")
    # 专业做法：回测日轴 = 交易所日历。数据缺失的官方交易日（全市场无数据）也出现在
    # 日轴与权益曲线上（按上一日市值结转、零换手）；数据中的非日历日视为异常剔除。
    # 日历不可用 / 无标志列（CALENDAR_ALIGN_MODE=off 老版合并）→ 回退按数据日轴迭代。
    _cal_axis = False
    if "is_trading" in data.columns:
        try:
            _cal_dates = _cal_get()
        except Exception:
            _cal_dates = set()
        if _cal_dates:
            _d_min = str(data["trade_date"].min())
            _d_max = str(data["trade_date"].max())
            _axis = sorted(d for d in _cal_dates if _d_min <= d <= _d_max)
            if _axis:
                _cal_axis = True
                # P3-1（审计）：不再物化全量 _grp_map（全市场日×股 groupby 预建
                # dict，回测内存峰值随日轴×股票数增长）。改为 groupby 惰性迭代器
                # 单遍流式扫描：数据日命中即取当日分组，缺失日历日（全市场无数据）
                # → 空表占位（按上一日市值结转、零换手）。sort=True 保证组有序，
                # 单遍扫描与 _axis 严格对齐，语义与原 _grp_map 查询一致。
                _n_data_days = int(data["trade_date"].nunique())
                _grp_iter = iter(data.groupby("trade_date", sort=True))
                _cur_grp = next(_grp_iter, None)
                date_groups = []
                for d in _axis:
                    if _cur_grp is not None and str(_cur_grp[0]) == d:
                        date_groups.append((d, _cur_grp[1]))
                        _cur_grp = next(_grp_iter, None)
                    else:
                        date_groups.append((d, pd.DataFrame()))
                if len(date_groups) != _n_data_days:
                    logger.info(
                        f"[CALENDAR] 引擎日轴对齐官方日历: {len(date_groups)} 日 "
                        f"（数据 {_n_data_days} 日，补全 {max(0, len(date_groups) - _n_data_days)} 日）"
                    )
            else:
                date_groups = list(data.groupby("trade_date", sort=True))
        else:
            date_groups = list(data.groupby("trade_date", sort=True))
    else:
        date_groups = list(data.groupby("trade_date", sort=True))

    symbols = sorted(data["symbol"].unique().tolist())
    # P1-4 修复：引擎启动时刷新行业映射缓存（从 params 注入 db_engine）
    _db_engine = params.get("_db_engine") if isinstance(params, dict) else None
    if engine_cfg.optimizer_industry_neutral:
        _refresh_industry_cache(_db_engine, symbols)

    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    n_syms = len(symbols)
    pos_value = np.zeros(n_syms, dtype=np.float64)
    pos_shares = np.zeros(n_syms, dtype=np.int32)
    # #5 审计修复：标记碎股状态（持仓不足一手，无法独立交易）
    # True 表示该标的有碎股（< lot 股），买入时应跳过（已有持仓），
    # 卖出碎股时随全仓卖出一起清理
    pos_has_fractional = np.zeros(n_syms, dtype=bool)

    # ── ST/退市逐日动态状态机（stock_st_history 由 runner 注入 params） ──
    # P0-6 ① 退市整理期业务规则（P1-4 修复：整理期 15/30 交易日，退市新规分段）：
    #   - 整理期（is_delisting=True 的交易日区间，自摘牌日前 N 个交易日
    #     至摘牌日；N=15（2020-12-31 退市新规及以后摘牌）/ 30（此前摘牌），
    #     由 DataManager/StPitSync 按终止上市日期回填，引擎只消费区间不推算）：
    #       * 首日无涨跌幅限制（±100% 近似，独立于 ST 5%）
    #       * 次日起 ±10%（进入整理期后不再适用 ST 5%）
    #       * 期间正常交易：可买可卖，不做强平（整理期股票仍可交易）
    #   - 摘牌日（整理期区间最后一个交易日）：当日禁止买入 + 当日收盘价强平
    #     （次日无行情可卖，强平为终态事件，以当日收盘价成交是保守近似）
    #   - 摘牌日之后（K线滞后延伸等兜底）：永久禁买 + 强平（维持原终态语义，
    #     避免"强平→次日复购→再强平"的循环刷交易）
    #   - ST/*ST 日（is_st=True 且非退市整理期）：仅当 _exclude_st=True 时
    #     禁止买入并强平（A股 ST 涨跌幅 5%）；exclude_st=False 时 ST 股全程正常交易
    # 预构建 {交易日: symbol 集合}，逐日 O(1) 查询；
    # 掩码用 np.isin 按当日行生成，长度与股票池/PIT 过滤后行数无关
    _st_hist = params.get("_st_history") if isinstance(params, dict) else None
    _exclude_st = bool(params.get("_exclude_st", False)) if isinstance(params, dict) else False  # FIX(P0): 默认不排除 ST
    # 退市整理期日集合（含首日/摘牌日）→ 整理期可交易 + 涨跌幅规则
    _delist_period_syms_by_day: dict[str, set[str]] = {}
    # 整理期首日集合 → 首日无涨跌幅豁免
    _delist_first_by_day: dict[str, set[str]] = {}
    # 摘牌日（整理期最后交易日）集合 → 摘牌日强平 + 当日禁买
    _delist_last_by_day: dict[str, set[str]] = {}
    # 摘牌日之后（理论无行情）→ 永久禁买 + 强平兜底
    _post_delist_block_by_day: dict[str, set[str]] = {}
    # ST/*ST 日（非整理期）→ 禁买 + 强平（仅 _exclude_st=True）
    _st_trade_block_by_day: dict[str, set[str]] = {}
    if _st_hist:
        _all_days = [str(dt) for dt, _g in date_groups]
        _day_set = set(_all_days)
        for _s, _recs in _st_hist.items():
            if not _recs:
                continue
            _del_days = sorted(
                d for d, (_st_f, _dl) in _recs.items()
                if _dl and str(d) in _day_set
            )
            if _del_days:
                _first_del, _last_del = _del_days[0], _del_days[-1]
                for _d_str in _all_days:
                    if _first_del <= _d_str <= _last_del:
                        _delist_period_syms_by_day.setdefault(_d_str, set()).add(_s)
                _delist_first_by_day.setdefault(_first_del, set()).add(_s)
                _delist_last_by_day.setdefault(_last_del, set()).add(_s)
                for _d_str in _all_days:
                    if _d_str > _last_del:
                        _post_delist_block_by_day.setdefault(_d_str, set()).add(_s)
            if _exclude_st:
                for _d_str, (_st_f, _dl) in _recs.items():
                    if _st_f and not _dl:
                        _st_trade_block_by_day.setdefault(str(_d_str), set()).add(_s)

    # ── PIT 过滤：上市日 + 退市日双向拦截 ──
    # point_in_time 模式下，每只股票只在 [上市日, 退市日] 期间可见。
    # 上市日前不可交易（防止前视），退市日后不可交易（防止幻觉交易）。
    # FIX(P0) Subtask-4：原版仅检查上市日（sym_first <= dt），遗漏退市日拦截。
    if engine_cfg.point_in_time:
        pit_first = data.groupby("symbol", sort=False)["trade_date"].min().to_dict()
        # 退市日 = K线数据中该股票的最后交易日（PIT 口径，与 StPitSync 独立源互补）
        pit_last = data.groupby("symbol", sort=False)["trade_date"].max().to_dict()
    else:
        pit_first = None
        pit_last = None
    pit = pit_first  # 保持兼容：pit 别名仍指向上市日字典

    # ── 撮合约束（simulate_limit_up_down）：涨跌停可成交量规则 ──
    # 开启：触板日按 可成交量比例(一字/盘中 × 连板衰减) 部分成交或未成交（日志可追溯）
    # 关闭：回退简化撮合（触板日一律禁止买入/卖出，等价原行为）
    _sim_limits = bool(getattr(engine_cfg, "simulate_limit_up_down", True))
    # P1-2 修复：涨跌停方向流动性不对称——拆分为涨跌停独立成交率
    _tradable_up_ratio = float(getattr(engine_cfg, "limit_tradable_up_ratio", 0.30))
    _tradable_down_ratio = float(getattr(engine_cfg, "limit_tradable_down_ratio", 0.30))
    # P1-2 修复：一字板封死也拆分买卖方向——买入排队极难（逆势），卖出相对容易（提供流动性）
    _seal_sell_ratio = float(getattr(engine_cfg, "limit_seal_sell_ratio", 0.05))
    _seal_buy_ratio = float(getattr(engine_cfg, "limit_seal_buy_ratio", 0.02))
    _seal_decay = float(getattr(engine_cfg, "limit_seal_decay", 0.5))
    # P0-6 ⑥：开盘集合竞价成交率分档（封单量/可成交量代理）——开盘价触板日，
    # 集合竞价可成交量 = 当日成交量 × min(触板档比例, auction_fill_ratio)
    _auction_ratio = float(getattr(engine_cfg, "auction_fill_ratio", 0.12))
    # ── 经验填充模型（技术债修复）：limit_ratio_mode=empirical_median/p10 时，
    # 用历史日线 V_t/V_prev 分位数替代固定比例常量（limit_calibration.py）。
    # 校准表全样本静态统计（等价静态参数选择，应用无前视）；构建失败告警回退 fixed。
    _limit_mode = str(getattr(engine_cfg, "limit_ratio_mode", "fixed")).lower()
    _calib_min_samples = int(getattr(engine_cfg, "limit_calib_min_samples", 20))
    _emp_calib = None
    if _sim_limits and _limit_mode in ("empirical_median", "empirical_p10"):
        try:
            from BackTrading.limit_calibration import build_empirical_calibration as _bec

            _calib_pct = 0.1 if _limit_mode == "empirical_p10" else 0.5
            _emp_calib = _bec(
                data, percentile=_calib_pct, min_samples=_calib_min_samples
            )
            logger.info(
                f"[经验填充] limit_ratio_mode={_limit_mode} 校准表构建完成: "
                f"分位={_calib_pct:.0%} 竞价触板单元格={len(_emp_calib.auction_table)}"
                f" 全天口径单元格={len(_emp_calib.day_type_table)}"
            )
        except Exception as _ce:
            logger.warning(f"[经验填充] 校准表构建失败，回退 fixed 档: {_ce}")
            _emp_calib = None
    # ── 0.6 复牌跳空：停牌后复牌日开盘大幅跳空（补涨兑现卖出 / 补跌日志标记 / 追高禁买） ──
    # 阈值 0 = 关闭该决策（仅识别复牌不动作）
    _resume_gap_up = float(getattr(engine_cfg, "resume_gap_up", 0.05))
    _resume_gap_down = float(getattr(engine_cfg, "resume_gap_down", 0.05))
    # P1-6 修复：复牌跳空卖出流动性冲击放大系数
    _resume_impact_multiplier = float(getattr(engine_cfg, "resume_impact_multiplier", 2.0))
    # ── P2-1 停牌盯市：停牌天数保守衰减折扣（无行业指数数据时的务实替代方案） ──
    _susp_decay_start = int(getattr(engine_cfg, "susp_decay_start_days", 10))
    _susp_daily_decay = float(getattr(engine_cfg, "susp_daily_decay_rate", 0.002))
    _susp_max_discount = float(getattr(engine_cfg, "susp_max_discount", 0.30))
    # ── 0.1 成交时点模型 ──
    # next_open=信号次日开盘成交（默认，A股T+1）/ vwap=信号次日VWAP。
    # next_open/vwap 将当日信号挂单，次日开盘按成交模型撮合，
    # 并与 simulate_limit_up_down 联动：次日一字涨停不可买入、一字跌停不可卖出。
    # close 模式已移除：信号由当日收盘数据计算，以同日收盘价成交=先知交易（前视偏差），
    # 本系统架构下不具备"收盘前下单"的物理可行性。若传入 close 强制回退 next_open。
    _exec_model = str(getattr(engine_cfg, "execution_model", "next_open")).lower()
    if _exec_model == "close":
        logger.error(
            "[执行模型] execution_model=close 已移除（固有前视偏差：信号依赖当日收盘数据计算，"
            "以同日收盘价成交等价于先知交易）。强制回退 next_open（信号次日开盘成交，符合A股T+1）。"
        )
        _exec_model = "next_open"
    if _exec_model not in ("next_open", "vwap"):
        logger.warning(f"[执行模型] 未知 execution_model={_exec_model!r}，回退 next_open")
        _exec_model = "next_open"
    _defer = True  # next_open/vwap 均为 deferred（信号日挂单→次日成交）
    _limit_streak: dict[str, int] = {}  # 连续涨停(+) / 连续跌停(-) 板数
    # P1-2（审计）：竞价路径用前日 streak 快照——_build_day_limit_model 在当日撮合
    # 前已用当日收盘封板状态更新 _limit_streak，竞价若直接读取即为前视。
    _limit_streak_prev: dict[str, int] = {}
    # 上市日映射（仅显式注入 {symbol: "YYYY-MM-DD"}，由 runner 从 stock_listing_days
    # 表加载，来源 AkShare stock_info_a_code_name 的上市日期）
    # P0-6 ④：禁止从行情数据推断上市日期——数据缺口/中途加入的股票会被误判为新股，
    # 错误激活"注册制前 5 日无涨跌幅"豁免（放大收益）。无注入 → 新股豁免逻辑整体停用
    # （仅"数据首日无前收"天然无涨跌停），并告警一次。
    _listing_days_map = params.get("_listing_days") if isinstance(params, dict) else None
    _day_idx = {str(dt): i for i, (dt, _g) in enumerate(date_groups)}
    if _listing_days_map is None:
        if engine_cfg.strict_listing_days:
            # P3-4（审计）：严格模式 fail-fast——上市日表缺失即中止回测，
            # 与数据质量门禁联动；杜绝"表缺失 → 静默停用新股豁免 → 结果口径漂移"。
            raise RuntimeError(
                "[上市日] strict_listing_days=True 但 params._listing_days 未注入"
                "（来源 stock_listing_days / AkShare stock_info_a_code_name）——"
                "新股涨跌停豁免口径不可用，中止回测"
            )
        # P3-5（修复）：threading.local 在从未赋值的线程直接读属性抛 AttributeError
        # （walk-forward 多窗口并发线程首次进入即崩）。getattr 默认 False 保证
        # 每线程首次访问安全，赋值后同线程内仍只告警一次。
        if not getattr(_ENGINE_SESSION, "listing_days_warned", False):
            _ENGINE_SESSION.listing_days_warned = True
            logger.warning(
                "[上市日] 未注入 IPO 日期表（params._listing_days，来源 stock_listing_days / "
                "AkShare stock_info_a_code_name），新股涨跌停豁免逻辑停用；"
                "禁止从行情数据推断上市日（数据缺口会误判新股，P0-6 ④）"
            )
    # 逐日 ST/*ST 集合（涨跌幅 5% 用）
    _st_syms_by_day: dict[str, set[str]] = {}
    if _st_hist:
        for _s, _recs in _st_hist.items():
            if not _recs:
                continue
            for _d_str, (_st_f, _dl) in _recs.items():
                if _st_f:
                    _st_syms_by_day.setdefault(str(_d_str), set()).add(_s)

    # 成本单一来源：CostModel（佣金含最低5元 / 印花税日期分段表 / 过户费日期分段表 / 经手费+证管费 /
    # 滑点 / 流动性分档冲击）。审计（成本外部化）：禁止回退"统一成本"口径——统一成本缺失
    # 逐笔最低佣金下限、历史费率日期分段表与流动性分档，静默回落会低估小额交易成本、忽略
    # 历史费率变更，导致回测结果口径分裂。EngineConfig.cost_model 必须显式传入（主流程
    # runner/meta_optimizer 已通过 CostModel.from_backtest_config 构建）；缺失时 fail-fast。
    cm = engine_cfg.cost_model
    if cm is None:
        raise ValueError(
            "[成本模型] EngineConfig.cost_model 未显式传入：交易成本必须由 CostModel 显式构建"
            "（逐笔最低佣金 min_commission_per_trade + 印花税/过户费/经手费/证管费日期分段表"
            " + 流动性分档冲击）。请通过 CostModel.from_backtest_config(bt, trading_cost=...) "
            "构建后传入，禁止回退统一成本口径。"
        )
    # 成本拆解累计（佣金/印花税/过户费/经手费/证管费/滑点/冲击 + 买卖成交额），
    # 回测结束输出 [成本拆解] 报告（各项占总成本百分比）
    _cost_accum: dict[str, float] = {
        "commission": 0.0, "stamp": 0.0, "transfer": 0.0,
        "handling": 0.0, "csrc": 0.0, "slippage": 0.0, "impact": 0.0,
        "buy_value": 0.0, "sell_value": 0.0,
    }
    # 滚动 20 日成交量（不含当日，前视合规）：value = (deque, run_sum)
    _adv_state: dict[str, tuple[Any, float]] = {}
    # P0-6 ⑤：每只滚动 _REGIME_RET_WINDOW+1 日复权收盘（市场收益/波动率客观口径）
    _close_hist: dict[str, deque] = {}
    _regime_vol_hist: deque = deque(maxlen=_REGIME_VOL_WINDOW)
    _prev_bar: dict[str, tuple[float, float]] = {}
    # P2-1：复权价 prev_bar 仅供 ATR 止损使用（跨除权日连续，无机械跳降）
    _prev_bar_adj: dict[str, tuple[float, float]] = {}
    # P1-X：每只标的上一交易日 adj_factor（除权日涨跌停基准校正）
    _prev_af: dict[str, float] = {}
    # ── 审计修复（P0-1）：_prev_af 防御性初始化守卫 ——
    # 防止旧版本残留路径/异常回滚状态将 _prev_af 置为 None 导致 L1774 崩溃
    _prev_af_guard = [True]  # 单元素 list 用于内部函数 nonlocal 共享
    # P0-11：每只标的上一交易日成交量（真实价体系下涨跌停开盘可成交量的前视合规代理）
    _prev_volume: dict[str, float] = {}
    # P0-11：每只持仓标的最近一次结算日复权因子（真实价体系除权股数调整基准）
    _pos_adjf: dict[str, float] = {}
    # P0-1：上一交易日止损线（后复权空间），今日 close_adj 跌破 → 次日开盘卖出
    _prev_stop: dict[str, float] = {}
    # 0.6 复牌跳空：每只标的最近一次有行情（bar）的交易日，用于识别停牌复牌
    _prev_bar_date: dict[str, str] = {}
    # P1-1 修复：持仓跟踪止损（Trailing Stop）
    # 入场以来的最高收盘价，用于动态上移止损价
    _max_close_since_entry: dict[str, float] = {}
    # P2 修复：跟踪止盈 — 入场收盘价，用于计算浮盈比例
    _entry_close: dict[str, float] = {}
    _buy_date: dict[str, str] = {}
    # P2-2：建仓时 buy_score 快照，退出时 exit_gt 与此值比较（而非当日 buy_score）
    _entry_buy_score: dict[str, float] = {}
    # FIX(P1) Subtask-9：持有期限计数（按标的逐日递增，>max_hold_days 重新评估）
    _held_days: dict[str, int] = {}

    max_pos_pct = engine_cfg.max_position_pct
    _max_holdings = engine_cfg.max_holdings
    _buy_threshold = engine_cfg.buy_threshold
    init_cash = engine_cfg.initial_cash

    _atr_stop = engine_cfg.atr_stop_mult
    # P2 修复：跟踪止盈参数
    _take_profit_pct = float(getattr(engine_cfg, "take_profit_pct", 15.0)) / 100.0
    _trail_profit_ratio = float(getattr(engine_cfg, "trail_profit_ratio", 50.0)) / 100.0
    # P1-5 max_order_pct 分档：按 ADV 成交额选择流动性分档上限
    _max_order_pct_default = engine_cfg.max_order_pct
    _max_order_pct_high = engine_cfg.max_order_pct_high
    _max_order_pct_low = engine_cfg.max_order_pct_low
    _adv_threshold_high = engine_cfg.adv_amount_threshold_high  # 高流动性阈值（元）
    _adv_threshold_low = engine_cfg.adv_amount_threshold_low    # 低流动性阈值（元）
    _top_k = engine_cfg.top_k

    cash = float(init_cash)
    _last_total_value = cash  # Task F 日历轴补全日：权益按上一日市值结转

    # ═══════════════════════════════════════════════════════════
    # P2.8 TODO: 嵌套函数拆分边界标记（后续重构为独立类）
    # ═══════════════════════════════════════════════════════════
    # ── ADV 管理 ── _update_adv / _current_adv
    #   → 提取至 engine/adv_manager.py (AdvManager)
    # ── 持仓盯市 ── _calc_market_value / _susp_position_value
    #   → 提取至 engine/position_manager.py (已有，扩展盯市职责)
    # ── 交易成本 ── _sell_proceeds_and_cost / _buy_cost / _process_sell
    #   → 提取至 engine/cost_calculator.py (已有 CostCalcuator — 对齐方法签名)
    # ── 成交撮合 ── _exec_price_for / _flush_pending / _auction_fill_for
    #   → 提取至 engine/execution_engine.py (已有 ExecutionEngine)
    # ── 涨跌停判定 ── _is_seal_up / _is_seal_down
    #   → 保留或移至 limit_pricing.py (纯计算，无引擎状态依赖)
    # ── 统计收集 ── _sink_inc / _sink_val / _sink_worst
    #   → 提取至 engine/stats_sink.py (轻量 dict 封装)
    # 拆分原则：各嵌套函数仅通过闭包引用引擎状态变量；
    # 重构后改为显式注入状态对象，消除隐式耦合。
    # ═══════════════════════════════════════════════════════════

    def _update_adv(sym: str, vol: float) -> float:
        """滚动 _ADV_WINDOW 日均量（不含当日）。当日 bar 结束后入账，供次日使用。

        P2.7 修复：vol <= 0 时不推进滑动窗口（停牌/零量日 forward fill 上次有效 ADV），
        避免停牌期 0 值拉低 ADV → 复牌后冲击成本/分档上限失真。
        """
        dq, run = _adv_state.get(sym, (None, 0.0))
        # P2.7：停牌/零量日不推进窗口，保持历史 ADV 不变
        if vol <= 0:
            if dq:
                return run / len(dq)
            return 0.0
        if dq is None:
            dq = deque(maxlen=_ADV_WINDOW)
        if len(dq) == dq.maxlen:
            run -= dq[0]
        dq.append(vol)
        run += vol
        _adv_state[sym] = (dq, run)
        return run / len(dq)

    def _current_adv(sym: str) -> float:
        """当前可用 ADV（前一日及之前 _ADV_WINDOW 日滚动均值）；无数据返回 0。"""
        dq, run = _adv_state.get(sym, (None, 0.0))
        return (run / len(dq)) if dq else 0.0

    def _calc_market_value() -> float:
        held = np.where(pos_shares > 0)[0]
        mtm = 0.0
        for si in held:
            s = symbols[si]
            px = close_lookup.get(s)
            if px is None or not np.isfinite(px):
                # 0.4 停牌盯市：当日无 bar（停牌/无行情）或 close_adj 缺失的持仓按
                # "停牌前最后收盘价"估值，不再冻结在买入成本价（否则上涨遇停牌净值被低估、
                # 下跌被高估，且停牌期净值"无波动"会虚高 Sharpe）
                px = _last_close.get(s)
                # P2-1 修复：停牌天数保守衰减折扣（无行业指数数据时的务实替代方案）
                if px is not None and np.isfinite(px):
                    susp_days = _susp_days.get(s, 0)
                    if susp_days > _susp_decay_start:
                        excess_days = susp_days - _susp_decay_start
                        # P1.1 修复：停牌衰减率按个股波动率自适应
                        # 高波动标的停牌不确定性更大 → 加大衰减；低波动 → 缩小
                        vol_scale = max(0.5, min(_last_amp_mult.get(s, 1.0), 3.0))
                        effective_decay = _susp_daily_decay * vol_scale
                        # 指数衰减：(1 - decay_rate)^excess_days 为保留比例
                        discount = 1.0 - (1.0 - effective_decay) ** excess_days
                        discount = min(discount, _susp_max_discount)
                        px = px * (1.0 - discount)
            if px is not None and np.isfinite(px):
                mtm += pos_shares[si] * px
            else:
                mtm += pos_value[si]  # 无任何历史收盘（理论不应发生）→ 退回买入成本
        return mtm

    close_lookup: dict[str, float] = {}
    # 0.4 停牌盯市：每只有行情交易日收盘后更新为当日复权收盘价，供停牌日估值回退
    _last_close: dict[str, float] = {}
    # P2-1：停牌天数追踪（用于保守衰减折扣，无行业指数数据时的务实替代方案）
    _susp_days: dict[str, int] = {}
    # P3 审计修复：记录每标的最新冲击参数（波动倍率/AMOUNT_MA20），供主循环结束后
    # 摘牌末段强平统一传入（与常规卖出口径一致，不再默认 1.0/None）
    _last_amp_mult: dict[str, float] = {}
    _last_amount: dict[str, float] = {}

    def _susp_position_value() -> float:
        """停牌持仓市值（当日无 bar 的持仓按停牌前最后收盘价盯市）。"""
        held = np.where(pos_shares > 0)[0]
        v = 0.0
        for si in held:
            s = symbols[si]
            px = close_lookup.get(s)
            if px is not None and np.isfinite(px):
                continue  # 当日有行情 → 非停牌
            px = _last_close.get(s, pos_value[si] / max(pos_shares[si], 1))
            v += pos_shares[si] * px
        return v

    def _sell_proceeds_and_cost(
        sym: str,
        value: float,
        volume: float,
        amount_ma20: float | None = None,
        dt: str | None = None,
        volatility_multiplier: float = 1.0,
    ) -> tuple[float, float]:
        # P1.2 断言：value 必须为不复权原始价 × 股数，禁止传入复权价格
        assert np.isfinite(value) and value >= 0, (
            f"[P1.2] _sell_proceeds_and_cost 收到非法 value={value}，检查是否传入复权价"
        )
        parts = cm.sell_cost_breakdown(
            value,
            volume,
            _current_adv(sym),
            amount_ma20=amount_ma20,
            dt=dt,
            volatility_multiplier=volatility_multiplier,
            symbol=sym,
        )
        _cost_accum["sell_value"] += value
        for _k in ("commission", "stamp", "transfer", "handling", "csrc", "slippage", "impact"):
            _cost_accum[_k] += parts[_k]
        return value - parts["total"], parts["total"]

    def _buy_cost(
        sym: str, value: float, volume: float, amount_ma20: float | None = None,
        dt: str | None = None, volatility_multiplier: float = 1.0,
    ) -> float:
        # P1.2 断言：value 必须为不复权原始价 × 股数
        assert np.isfinite(value) and value >= 0, (
            f"[P1.2] _buy_cost 收到非法 value={value}，检查是否传入复权价"
        )
        parts = cm.buy_cost_breakdown(
            value,
            volume,
            _current_adv(sym),
            amount_ma20=amount_ma20,
            dt=dt,
            volatility_multiplier=volatility_multiplier,
            symbol=sym,
        )
        _cost_accum["buy_value"] += value
        for _k in ("commission", "stamp", "transfer", "handling", "csrc", "slippage", "impact"):
            _cost_accum[_k] += parts[_k]
        return parts["total"]

    def _process_sell(
        dt,
        s_syms,
        s_idx,
        s_close,
        s_vol,
        partial: bool = False,
        s_amount: np.ndarray | None = None,
        s_amp_mult: np.ndarray | None = None,
        s_fill_ratio: np.ndarray | None = None,
        s_limit_tag: np.ndarray | None = None,
        s_sig_close: np.ndarray | None = None,
        s_force: bool = False,
    ):
        total_sold = 0.0
        for j in range(len(s_syms)):
            si = s_idx[j]
            sh = int(pos_shares[si])
            if sh <= 0:
                continue
            # P1-2（审计）：兜底守卫——成交价无效（NaN/<=0）跳过该笔，
            # 防止 cash += proc 把 NaN 传播到整条权益曲线（买入侧有对称守卫）。
            # 调用方（挂单撮合/复牌兑现/摘牌强平）已用 _valid_px_mask 前置过滤，
            # 此处防御未来新增调用点。
            if not (np.isfinite(float(s_close[j])) and float(s_close[j]) > 0):
                logger.warning(
                    f"[执行模型] {dt} {s_syms[j]} 卖出成交价无效(NaN/<=0) → 跳过该笔"
                )
                continue
            # 申报数量单位：按板块（科创 200 股/手，其余 100 股/手），一处定义全链路复用
            lot = lot_size_for(s_syms[j])
            if partial:
                # 半仓减仓：取最近整手数（四舍五入而非向下取整，避免 300 股只卖 100 股）
                # P1-14 审计修复：使用 round() 替代 int(x + 0.5) — float 边界 149.999→150 更可靠
                _half_lots = round(sh / 2 / lot)
                sell_shares = max(lot, int(_half_lots) * lot)
                if sell_shares >= sh:
                    sell_shares = sh
                # #5 审计修复：半仓卖出后剩余不足一手 → 改为全卖，避免碎股
                _remaining = sh - sell_shares
                if 0 < _remaining < lot:
                    sell_shares = sh
            else:
                sell_shares = sh
            # 撮合约束：跌停/涨停日按可成交量比例部分成交或未成交（日志可追溯）
            _limit_note = None
            if s_fill_ratio is not None:
                _sink_inc("sell_limit_orders", 1)
            if s_fill_ratio is not None and s_fill_ratio[j] < 1.0:
                _req = sell_shares
                _avail = int(float(s_vol[j]) * float(s_fill_ratio[j])) // lot * lot
                _updown = "涨停" if (s_limit_tag is not None and s_limit_tag[j] == "up") else "跌停"
                if _avail < lot:
                    _sink_inc("sell_limit_rejected", 1)
                    _sink_val("sell_limit_unfilled_value", float(_req) * float(s_close[j]))
                    _u_sell += float(_req) * float(s_close[j])
                    _day_agg["reject_sell"] += 1
                    logger.debug(
                        f"[撮合约束] {dt} {s_syms[j]} {_updown} 可成交量不足 → 未成交（卖出） 请求={_req}股 可成交={_avail}股"
                    )
                    continue
                if _avail < sell_shares:
                    _sink_inc("sell_limit_partial", 1)
                    _sink_val("sell_limit_unfilled_value", float(_req - _avail) * float(s_close[j]))
                    _u_sell += float(_req - _avail) * float(s_close[j])
                    sell_shares = _avail
                    _limit_note = s_limit_tag[j] if s_limit_tag is not None else "down"
                    _day_agg["partial_sell"] += 1
                    logger.debug(
                        f"[撮合约束] {dt} {s_syms[j]} {_updown} 部分成交（卖出） 请求={_req}股 成交={sell_shares}股 fill_ratio={float(s_fill_ratio[j]):.3f}"
                    )
            mv = sell_shares * float(s_close[j])
            if s_fill_ratio is not None:
                _sink_val("sell_limit_fill_value", mv)
            pos_shares[si] -= sell_shares
            if pos_shares[si] <= 0:
                pos_value[si] = 0.0
                pos_shares[si] = 0
                pos_has_fractional[si] = False  # #5 清仓清理碎股标记
                # P2-2：清仓时清理建仓 buy_score 快照
                _entry_buy_score.pop(s_syms[j], None)
                # FIX(P1) Subtask-9：清仓时清理持有期限计数器，避免 _held_days 残留
                _held_days.pop(s_syms[j], None)
                # FIX(P1) Subtask-9：清仓时清理持有期限计数器
                _held_days.pop(s_syms[j], None)
                # P0 修复（跟踪止损）：清仓时清理 max_close_since_entry
                _max_close_since_entry.pop(s_syms[j], None)
                # P2 修复（跟踪止盈）：清仓时清理 entry_close
                _entry_close.pop(s_syms[j], None)
                # P0-1 审计修复：清仓时保留 _buy_date 历史记录，不清除。
                # 原实现清仓时 pop(_buy_date) 存在风险：若执行顺序调整（如 next_open 模型
                # 下买入挂单成交后又有同标的卖出挂单），会导致 T+1 守卫被绕过。
                # 保留历史值不影响逻辑——T+1 守卫使用 _buy_date.get(s, "")，
                # 新买入会覆盖旧值；清仓后不持仓则 T+1 守卫恒通过。
            else:
                # P0-9 ①：半仓/部分卖出后 pos_value 按剩余股数比例递减。
                # pos_value 是持仓成本市值（买入成交额，core.py 983 行），清仓分支已归零，
                # 但减仓分支此前保持不变 → 停牌无价回退估值时单位成本 = pos_value/剩余股数
                # 被高估（原成本市值 ÷ 减半后股数 ≈ 2 倍成本价），虚增净值。
                if pos_shares[si] > 0:
                    pos_value[si] *= pos_shares[si] / sh
                # #5 审计修复：标记碎股状态（持仓 < 1 手）
                if 0 < pos_shares[si] < lot:
                    pos_has_fractional[si] = True
            proc, cst = _sell_proceeds_and_cost(
                s_syms[j],
                mv,
                float(sell_shares),
                amount_ma20=float(s_amount[j]) if s_amount is not None else None,
                dt=str(dt),
                volatility_multiplier=float(s_amp_mult[j]) if s_amp_mult is not None else 1.0,
            )
            nonlocal cash
            cash += proc
            total_sold += mv
            _extra = (
                {"limit": _limit_note, "fill_ratio": round(float(s_fill_ratio[j]), 3)}
                if _limit_note is not None
                else {}
            )
            _exec_anchor = float(s_sig_close[j]) if s_sig_close is not None else float(s_close[j])
            trade_log.append(
                {
                    "time": dt,
                    "symbol": s_syms[j],
                    "action": "sell" if sell_shares >= sh else "sell_partial",
                    "price": float(s_close[j]),
                    # P0-11：value 统一为成交毛额（与买入 tv 毛额同口径，成本单列 cost）
                    "value": _round_half_up(mv),
                    "cost": _round_half_up(cst),
                    # 1.9 流动性拟真字段：实际成交数量（A股最小交易单位整数倍）
                    "qty": int(sell_shares),
                    # 1.7 执行滞后自检字段：close 模型下为成交日真实收盘（=price）；
                    # next_open/vwap 下为信号日复权收盘价锚点（成交锚点在 exec_open）
                    "close_adj": (
                        float(s_close[j])
                        if _exec_model == "close"
                        else _exec_anchor
                    ),
                    # #5 审计修复：碎股状态标记（持仓不足一手）
                    "has_fractional": bool(0 < pos_shares[si] < lot),
                    # 0.1 成交参考价（成交日开盘/VWAP/收盘）——成交时序自检锚点
                    "exec_open": float(s_close[j]),
                    **_extra,
                    **({"force_exit": True} if s_force else {}),
                }
            )
        return total_sold

    # ── 0.1 成交时点模型：挂单队列（next_open/vwap） ──
    # 信号日收盘下单 → 次日开盘按成交模型撮合。先卖后买（卖出回笼资金再买入）。
    _pending_sells: list[dict[str, Any]] = []
    _pending_buys: list[dict[str, Any]] = []
    # P2-1（审计）：VWAP 越界弃用计数器——空间一致性断言。
    # P0-1 后 low/high 与 amount/volume 同为不复权原始口径，真实 VWAP=amount/volume
    # 必落在当日 [low, high]；大量越界说明 K 线口径不一致（如库内为 hfq），
    # 主循环结束统一告警（不逐笔刷屏）。
    _vwap_reject_count = 0
    # P3-2（审计）：撮合约束/现金不足逐笔日志聚合器——全市场回测逐笔 logger.info
    # 日志量巨大，改为每日一次汇总（INFO）+ 逐笔降级 DEBUG（保留可追溯性）。
    _day_agg = {"reject_sell": 0, "partial_sell": 0, "partial_buy": 0, "cash_cancel": 0}

    # ── 涨跌停撮合统计（stats_sink，涨跌停专项压力测试消费） ──
    # 统计语义：
    #   buy/sell_limit_*：开盘触板（竞价路径）单子，成交额以决策价×成交股数计
    #   seal_*_rejected：一字板撤销（买入=踏空敞口，卖出=套牢敞口）
    #   *_unfilled_value：请求价值 - 实际成交价值（可成交量不足的未成交敞口）
    #   *_worst_day：[日期, 当日未成交金额]（worst-case 单日敞口）
    _sink = stats_sink

    def _sink_inc(key: str, n: int = 1) -> None:
        if _sink is not None:
            _sink[key] = _sink.get(key, 0) + n

    def _sink_val(key: str, v: float) -> None:
        if _sink is not None and v > 0:
            _sink[key] = _sink.get(key, 0.0) + v

    def _sink_worst(key: str, dt_day, v: float) -> None:
        if _sink is not None and v > 0:
            _prev = _sink.get(key)
            if _prev is None or v > _prev[1]:
                _sink[key] = [str(dt_day), round(v, 2)]

    def _exec_price_for(day_data_ld: pd.DataFrame, j: int, close_local) -> float:
        """成交参考价（0.1 执行时点模型）— 真实价格体系（不复权原始价）。

        P0-11 审计修复（复权货币体系失真）：成交/现金/市值/费用全部以不复权
        原始价结算；复权口径仅保留用于止损线/信号比较与 regime 状态变量。
        除权日持仓股数按 adj_factor 比率调整（见主循环除权调整块），保证真实价
        估值跨除权日无伪盈亏，净值/收益/仓位比例恢复真实值。

        close=当日收盘 / next_open=开盘 / vwap=日成交量加权均价。
        next_open：开盘价缺失/非法 → 返回 NaN（无效成交价，调用方拒单顺延），
        绝不以当日收盘价成交——T+1 收盘在集合竞价时不可知（P1-1 审计修复）。
        vwap：真实日频 VWAP = 成交额 / 成交量（amount/volume 均为不复权原始值）；
        真实 VWAP 必落在当日 [low, high] 内，越界（成交额单位异常）或成交额
        缺失时回退典型价 (O+H+L+C)/4。

        P2 审计修复：成交价统一 quantize 到 0.01 元（ROUND_HALF_UP）——A 股
        最小报价单位为 0.01 元，VWAP=amount/volume 与典型价回退天然非 0.01
        整数倍，直接作成交价违反最小报价单位；与涨跌停价 limit_pricing
        （ROUND_HALF_UP）口径分裂已消除。非有限值原样返回（调用方过滤）。
        """
        nonlocal _vwap_reject_count
        if _exec_model == "next_open":
            # P1-1（审计）修复：开盘价缺失/非法时返回 NaN（无效成交价），
            # 由调用方拒单顺延至下一可交易日开盘。旧实现回退当日收盘价——
            # T+1 收盘价在 9:15-9:25 集合竞价时不可知，进入成交价即前视。
            v = float("nan")
            if "open" in day_data_ld.columns:
                _o = float(day_data_ld["open"].values[j])
                if np.isfinite(_o) and _o > 0:
                    v = _o
            if np.isfinite(v) and v > 0:
                return _round_half_up(v)
            return v
        if _exec_model == "vwap":
            _c = float(close_local[j])  # 真实收盘
            if "open" in day_data_ld.columns:
                _o = float(day_data_ld["open"].values[j])
            else:
                _o = _c
            if "high" in day_data_ld.columns:
                _h = float(day_data_ld["high"].values[j])
            else:
                _h = _c
            if "low" in day_data_ld.columns:
                _l = float(day_data_ld["low"].values[j])
            else:
                _l = _c
            # ── 真实 VWAP = 成交额 / 成交量（不复权原始口径） ──
            _vwap = None
            if "amount" in day_data_ld.columns and "volume" in day_data_ld.columns:
                _amt = float(day_data_ld["amount"].values[j])
                _vol = float(day_data_ld["volume"].values[j])
                if np.isfinite(_amt) and np.isfinite(_vol) and _vol > 0 and _amt > 0:
                    _vwap = _amt / _vol  # 不复权 VWAP（元/股）
                    # 真实 VWAP 必落在当日 [low, high] 内；越界 → 成交额单位异常，弃用
                    # P2-1（审计）：越界计数 → 主循环结束空间一致性告警
                    # （P0-1 后 low/high 与 amount/volume 同为不复权原始口径）
                    if not (np.isfinite(_vwap) and _vwap > 0 and _l - 1e-9 <= _vwap <= _h + 1e-9):
                        _vwap = None
                        _vwap_reject_count += 1
            if _vwap is not None:
                return _round_half_up(_vwap)
            # FIX(P1) Subtask-6：VWAP前视偏差修复——回退链仅允许开盘价。
            # 旧实现回退典型价(O+H+L+C)/4 和收盘价，这两者在集合竞价时不可知（前视偏差）。
            # VWAP 无法计算时：试开盘价 → NaN（让调用方挂单顺延）。
            if np.isfinite(_o) and _o > 0:
                return _round_half_up(_o)
            # VWAP 无法计算且开盘价不可用 → 返回 NaN，挂单顺延至下一可交易日
            return float("nan")
        v = float(close_local[j])
        if not (np.isfinite(v) and v > 0):
            return v
        return _round_half_up(v)

    def _flush_pending(
        dt,
        day_data_ld,
        syms_str_ld,
        idx_ld,
        close_adj_ld,
        close_raw_ld,
        open_arr_ld,
        volume_ld,
        at_limit_up_ld,
        at_limit_down_ld,
        limit_up_ld,
        limit_down_ld,
        adj_ok_ld,
        has_volume_ld,
        amount_ma20_ld,
        _vol_mult_ld,
        _limit_tag_ld,
        resume_gap_up_ld=None,
    ) -> tuple[float, float]:
        """次日开盘撮合（0.1）：先卖后买，一字板联动限制（一字涨停不可买/一字跌停不可卖）。

    P0-6 ② 两档成交模型：
        档位 1「次日集合竞价委托」：本函数即该档（信号日收盘挂单 → 次日 9:15-9:25
        集合竞价撮合，成交价 = 开盘价）。订单仅在信号次日有效：
        停牌/无行情 → 废单撤销；一字封死 → 撤销；强平单（ST/退市）逐日重挂。
        档位 2「盘中市价委托」：不单独建模盘中成交；触板日的成交率由次日开盘
        集合竞价档近似（auction_fill_ratio_for，仅用 9:25 已知的 open/限价判定，
        P1-2 前视修复）。盘中档 fill_ratio（一字/炸板/盘中冲板 × 连板衰减，
        依赖当日 close/high/low，属未来信息且从未被消费）已删除（P2 审计修复）。

    Returns:
        (buy_value, sell_value) 当日入账的买卖金额（用于 turnover 统计）。
    """
        nonlocal cash
        if not _pending_sells and not _pending_buys:
            return 0.0, 0.0
        buy_val, sell_val = 0.0, 0.0
        _u_buy, _u_sell = 0.0, 0.0  # 当日未成交金额（stats_sink worst-case 用）
        sym_row = {s: j for j, s in enumerate(syms_str_ld)}

        def _auction_fill_for(sym: str, j: int, side: str = "sell") -> float:
            """开盘触板日集合竞价可成交量比例（PIT：仅用 9:25 已知 open/限价）。

            P1-2 修复：一字板场景拆分买卖方向成交率——
            涨停板买入极难（排队深，逆势），跌停板卖出极难（恐慌抛压）；
            涨停板卖出相对容易（提供流动性），跌停板买入相对容易（逆势抄底少但有流动性）。

            fixed 档：min(auction_fill_ratio_for, auction_fill_ratio)（原行为）；
            经验档（limit_ratio_mode=empirical_median/p10）：校准表分位数，
            单元格样本不足回退 fixed 档（limit_calibration.py）。
            """
            _o = float(open_arr_ld[j]) if open_arr_ld is not None else None
            # P1-2 修复：一字板封死场景按方向拆分成交率（内联 seal 判定避免前向引用）
            _is_seal = False
            if side == "sell" and at_limit_down_ld[j] and open_arr_ld is not None and abs(open_arr_ld[j] - close_raw_ld[j]) <= 1e-9:
                _is_seal = True
            elif side == "buy" and at_limit_up_ld[j] and open_arr_ld is not None and abs(open_arr_ld[j] - close_raw_ld[j]) <= 1e-9:
                _is_seal = True

            if _is_seal:
                # 一字板封死：使用 seal 档比例（买卖方向独立）
                _seal_r = _seal_sell_ratio if side == "sell" else _seal_buy_ratio
                _streak = abs(_limit_streak_prev.get(sym, 0)) + 1
                return _seal_r * (_seal_decay ** max(0, _streak - 1))

            _fixed = min(
                auction_fill_ratio_for(
                    _o,
                    float(limit_up_ld[j]), float(limit_down_ld[j]),
                    side=side,
                    tradable_up_ratio=_tradable_up_ratio,
                    tradable_down_ratio=_tradable_down_ratio,
                    board_streak=abs(_limit_streak_prev.get(sym, 0)) + 1,
                    seal_decay=_seal_decay,
                ),
                _auction_ratio,
            )
            if _emp_calib is None or _o is None:
                return _fixed
            _t_up = _o >= float(limit_up_ld[j]) - 1e-9
            _t_down = _o <= float(limit_down_ld[j]) + 1e-9
            if not (_t_up or _t_down):
                return _fixed
            return _emp_calib.auction_fill_ratio(
                open_at_limit_up=_t_up,
                open_at_limit_down=_t_down,
                streak=abs(_limit_streak_prev.get(sym, 0)) + 1,
                fallback=_fixed,
            )

        # 一字板判定：开=收=限价（sealed board）
        def _is_seal_up(j: int) -> bool:
            return at_limit_up_ld[j] and open_arr_ld is not None and abs(open_arr_ld[j] - close_raw_ld[j]) <= 1e-9

        def _is_seal_down(j: int) -> bool:
            return at_limit_down_ld[j] and open_arr_ld is not None and abs(open_arr_ld[j] - close_raw_ld[j]) <= 1e-9

        # ── 卖出挂单（先回笼资金） ──
        remaining_sells: list[dict[str, Any]] = []
        for p in _pending_sells:
            # FIX(P1) Subtask-6：前视守卫——执行日必须严格晚于信号日（T+1.deferred语义）
            assert str(dt) != p.get("sig_dt"), (
                f"[前视偏差] 卖出挂单执行日 {dt} == 信号日 {p.get('sig_dt')} "
                f"（{p['sym']}）— 违反T+1延迟成交契约"
            )
            # P0-6 ②：强平单（ST/退市）逐日重挂直至成交（终态必须离场），不设过期；
            # 普通卖出单仅当日有效（停牌/一字跌停即撤销、不重挂）。
            # P1-1（审计）：开盘价缺失顺延的普通卖出单受 order_expiry_days 过期约束，
            # 避免数据异常期间挂单无限滞留。
            p["_age"] = p.get("_age", 0) + 1
            if not p.get("force") and p["_age"] > engine_cfg.order_expiry_days:
                logger.info(
                    f"[执行模型] {dt} {p['sym']} 卖出挂单过期"
                    f"（信号日 {p['sig_dt']}，已顺延 {p['_age'] - 1} 个交易日）→ 撤销"
                )
                continue
            jj = sym_row.get(p["sym"])
            if jj is None or not adj_ok_ld[jj] or not has_volume_ld[jj]:
                # P0-6 ②：停牌/当日无行情 → 废单撤销（A股停牌日委托无效）。
                # 例外：强平单（ST/退市）逐日重挂直至成交（终态必须离场）
                if p.get("force"):
                    remaining_sells.append(p)
                else:
                    logger.info(f"[执行模型] {dt} {p['sym']} 卖出挂单废单撤销（停牌/无行情）")
                continue
            px = _exec_price_for(day_data_ld, jj, close_raw_ld)
            if not (np.isfinite(px) and px > 0):
                # P1-1（审计）：开盘价缺失/非法 → 不得以当日收盘价成交（前视），
                # 顺延至下一可交易日开盘重试（强平单不受过期约束）
                logger.info(f"[执行模型] {dt} {p['sym']} 开盘价缺失 → 卖出挂单顺延")
                remaining_sells.append(p)
                continue
            if _is_seal_down(jj):
                # P1-2 修复：一字跌停不再一律拒绝，改为按 seal_sell_ratio 部分成交
                # （恐慌抛压时仍有少量成交，卖出相对容易因提供流动性）
                _auction_ratio_sell = _auction_fill_for(p["sym"], jj, side="sell")
                _sink_inc("seal_sell_partial", 1)
                # P0.3 修复：前日无数据时不再回退当日量（前视），改用保守默认值。
                _pv_sell = _prev_volume.get(p["sym"], 0.0)
                _vol_ref_sell = _pv_sell if _pv_sell > 0 else _DEFAULT_VOLUME_FALLBACK
                sell_val += _process_sell(
                    dt,
                    np.array([p["sym"]], dtype=object),
                    np.array([idx_ld[jj]], dtype=np.int32),
                    np.array([px]),
                    np.array([_vol_ref_sell]),
                    partial=True,
                    s_amount=np.array([float(amount_ma20_ld[jj])]) if amount_ma20_ld is not None else None,
                    s_amp_mult=np.array([float(_vol_mult_ld[jj])]),
                    s_fill_ratio=np.array([_auction_ratio_sell]),
                    s_limit_tag=["down"] if _limit_tag_ld is not None else None,
                    s_sig_close=None,
                )
                if p.get("force"):
                    pass  # 强平单一字跌停已按部分成交处理
                else:
                    logger.info(
                        f"[执行模型] {dt} {p['sym']} 一字跌停 → 卖出按 seal_sell_ratio 部分成交 "
                        f"(fill_ratio={_auction_ratio_sell:.3f})"
                    )
                continue
            # 0.3 盘中触板：仅在跌停开盘（open ≤ 跌停价）时按日级可成交量折算，
            # 正常开盘不因盘中触板而限制成交（避免误伤正常开盘的单子）
            _open_at_limit_down = (
                open_arr_ld is not None and open_arr_ld[jj] <= limit_down_ld[jj] + 1e-9
            )
            # P0-6 ⑥：开盘集合竞价成交率分档（封单量/可成交量代理）——开盘价触板
            # 时，集合竞价可成交量 = 可参考成交量 × min(触板档比例, auction_fill_ratio)。
            # P0-11 修复：可参考成交量用"前日成交量"（前视合规——当日全天成交量在
            # 开盘竞价时不可知）；前日无数据（数据起点/长期停牌复牌首日）回退当日量。
            # P1-2 修复：触板档比例改用 auction_fill_ratio_for（仅用 9:25 已知的
            # open/限价判定，不复用依赖当日 close/high/low 的盘中档位——前视消除）。
            # 技术债修复：经验档（limit_ratio_mode=empirical_*）改查校准表分位数，
            # 固定比例常量（0.30/0.12）仅作经验单元格样本不足时的回退。
            # 假设文档化：成交价 = 开盘价（集合竞价价）；开盘后向限价收敛的盘中成交
            # 不单独建模（对卖出保守：跌停开盘成交价=跌停价，且成交率受竞价档上限约束）
            _auction_ratio_sell = (
                _auction_fill_for(p["sym"], jj, side="sell")
                if (_sim_limits and _open_at_limit_down)
                else None
            )
            # P0.3 修复：前日无数据时不再回退当日量（前视），改用保守默认值。
            _pv_sell = _prev_volume.get(p["sym"], 0.0)
            _vol_ref_sell = _pv_sell if _pv_sell > 0 else _DEFAULT_VOLUME_FALLBACK
            sell_val += _process_sell(
                dt,
                np.array([p["sym"]], dtype=object),
                np.array([idx_ld[jj]], dtype=np.int32),
                np.array([px]),
                np.array([_vol_ref_sell]),
                partial=bool(p["partial"]),
                s_amount=np.array([float(amount_ma20_ld[jj])]) if amount_ma20_ld is not None else None,
                s_amp_mult=np.array([float(_vol_mult_ld[jj])]),
                s_fill_ratio=np.array([_auction_ratio_sell]) if _auction_ratio_sell is not None else None,
                s_limit_tag=[_limit_tag_ld[jj]] if _auction_ratio_sell is not None else None,
                s_sig_close=np.array([float(p["sig_close"])]),
                s_force=bool(p.get("force", False)),
            )
        _pending_sells[:] = remaining_sells
        # ── 买入挂单（按信号日优先级顺序；一字涨停不可买） ──
        filled = 0
        if _max_holdings > 0:
            _slots = max(0, _max_holdings - int((pos_shares > 0).sum()))
        else:
            _slots = _top_k
        remaining_buys: list[dict[str, Any]] = []
        for p in _pending_buys:
            # FIX(P1) Subtask-6：前视守卫——执行日必须严格晚于信号日（T+1.deferred语义）
            assert str(dt) != p.get("sig_dt"), (
                f"[前视偏差] 买入挂单执行日 {dt} == 信号日 {p.get('sig_dt')} "
                f"（{p['sym']}）— 违反T+1延迟成交契约"
            )
            # ── 挂单过期检查（P0-6 ②：A股订单当日有效，次日未成交即撤销） ──
            p["_age"] = p.get("_age", 0) + 1
            if p["_age"] > engine_cfg.order_expiry_days:
                logger.info(
                    f"[执行模型] {dt} {p['sym']} 买入挂单过期"
                    f"（信号日 {p['sig_dt']}，已等待 {p['_age'] - 1} 个交易日）→ 撤销"
                )
                continue
            if filled >= _slots:
                continue  # 无空仓额度 → 撤销
            jj = sym_row.get(p["sym"])
            if jj is None or not adj_ok_ld[jj] or not has_volume_ld[jj]:
                # P0-6 ②：停牌/当日无行情 → 废单撤销（A股停牌日委托无效；
                # 买入挂单不存在"终态必须成交"语义，一律撤销而非顺延）
                logger.info(f"[执行模型] {dt} {p['sym']} 买入挂单废单撤销（停牌/无行情）")
                continue
            si = p["si"]
            if pos_shares[si] > 0:
                continue  # 已持仓 → 撤销
            # P0-11：数据首日（无前收）禁买——无法确定涨跌停基准的标的不可成交，
            # 避免"无涨跌停限制"假豁免产生虚高收益（原实现按无限制豁免可买）
            # P3 审计修复：原守卫查 _prev_bar 恒不触发——前收刷新（:1319）早于
            # 本日撮合（_flush_pending），守卫读到的恒是当日数据；改查刷新前
            # 快照 _prev_seen：标的从未出现过前收（数据起点/新股首日）才禁买
            if _prev_seen.get(p["sym"]) is None:
                logger.info(
                    f"[执行模型] {dt} {p['sym']} 数据首日无前收 → 买入撤销（涨跌停基准缺失）"
                )
                continue
            if resume_gap_up_ld is not None and resume_gap_up_ld[jj]:
                # P2-5 修复：复牌跳空高开买入从"直接撤销"改为按集合竞价成交率部分成交
                # （复牌日买方抢筹激烈，集合竞价可成交量有限）
                _resume_auction_ratio = float(getattr(engine_cfg, "resume_auction_fill_ratio", 0.15))
                # P0.3 修复：前日无数据时不再回退当日量（前视），改用保守默认值。
                _pv_resume = _prev_volume.get(p["sym"], 0.0)
                _vol_ref_resume = _pv_resume if _pv_resume > 0 else _DEFAULT_VOLUME_FALLBACK
                lot_resume = lot_size_for(p["sym"])
                _avail_resume = int(_vol_ref_resume * _resume_auction_ratio) // lot_resume * lot_resume
                if _avail_resume < lot_resume:
                    logger.info(f"[执行模型] {dt} {p['sym']} 复牌高开 → 买入未成交（可成交量不足）")
                    continue
                # 按可成交比例缩小挂单金额
                _resume_fill_pct = min(_avail_resume * _exec_price_for(day_data_ld, jj, close_raw_ld) / max(float(p["tv"]), 1), 1.0)
                p["tv"] = float(p["tv"]) * _resume_fill_pct
                _sink_inc("resume_buy_partial", 1)
                logger.info(
                    f"[执行模型] {dt} {p['sym']} 复牌高开 → 买入按 resume_auction_fill_ratio 部分成交 "
                    f"(fill_ratio={_resume_fill_pct:.3f})"
                )
                # 继续下方正常撮合流程（tv 已按成交率缩小）
            if _is_seal_up(jj):
                # P1-2 修复：一字涨停不再一律拒绝，改为按 seal_buy_ratio 部分成交
                # （涨停板买方排队极难，逆势买入成交率最低）
                px_seal = _exec_price_for(day_data_ld, jj, close_raw_ld)
                if not (np.isfinite(px_seal) and px_seal > 0):
                    remaining_buys.append(p)
                    continue
                _auction_ratio_buy_seal = _auction_fill_for(p["sym"], jj, side="buy")
                _sink_inc("seal_buy_partial", 1)
                # P0.3 修复：前日无数据时不再回退当日量（前视），改用保守默认值。
                _pv_seal = _prev_volume.get(p["sym"], 0.0)
                _vol_ref_seal = _pv_seal if _pv_seal > 0 else _DEFAULT_VOLUME_FALLBACK
                lot_seal = lot_size_for(p["sym"])
                _avail_seal = int(_vol_ref_seal * _auction_ratio_buy_seal) // lot_seal * lot_seal
                if _avail_seal < lot_seal:
                    _sink_inc("seal_buy_rejected", 1)
                    _sink_val("seal_buy_rejected_value", float(p["tv"]))
                    _u_buy += float(p["tv"])
                    logger.info(f"[执行模型] {dt} {p['sym']} 一字涨停 → 买入未成交（可成交量不足）")
                    continue
                # 按可成交量折算买入金额
                _seal_buy_value = _avail_seal * px_seal
                _seal_cst = _buy_cost(
                    p["sym"], _seal_buy_value, float(_avail_seal),
                    amount_ma20=float(amount_ma20_ld[jj]) if amount_ma20_ld is not None else None,
                    dt=str(dt),
                    volatility_multiplier=float(_vol_mult_ld[jj]),
                )
                if cash < _seal_buy_value + _seal_cst:
                    _sink_inc("seal_buy_cash_cancel", 1)
                    _u_buy += float(p["tv"])
                    logger.info(f"[执行模型] {dt} {p['sym']} 一字涨停 → 买入现金不足（撤销）")
                    continue
                cash -= (_seal_buy_value + _seal_cst)
                pos_shares[si] += _avail_seal
                pos_value[si] += _seal_buy_value
                _sink_inc("buy_limit_orders", 1)
                _sink_val("buy_limit_fill_value", _seal_buy_value)
                trade_log.append({
                    "time": dt,
                    "symbol": p["sym"],
                    "action": "buy_partial",
                    "price": px_seal,
                    "value": _round_half_up(_seal_buy_value),
                    "cost": _round_half_up(_seal_cst),
                    "qty": int(_avail_seal),
                    "close_adj": px_seal,
                    "exec_open": px_seal,
                    "limit": "up",
                    "fill_ratio": round(_auction_ratio_buy_seal, 3),
                })
                logger.info(
                    f"[执行模型] {dt} {p['sym']} 一字涨停 → 买入按 seal_buy_ratio 部分成交 "
                    f"(fill_ratio={_auction_ratio_buy_seal:.3f}, qty={_avail_seal}股)"
                )
                continue
            px = _exec_price_for(day_data_ld, jj, close_raw_ld)
            if not np.isfinite(px) or px <= 0:
                remaining_buys.append(p)
                continue
            lot = lot_size_for(p["sym"])
            shares = int(float(p["tv"]) / px) // lot * lot
            if shares < lot:
                continue
            # ── P3-3：ADV 维度注释 + P1-5 分档 ──
            # _adv_val 来自 _current_adv()，返回 20 日平均成交量（股数）。
            # P1-5 修复：按ADV成交额（股数×当日价）选择分档 max_order_pct 上限
            _adv_val = _current_adv(p["sym"])
            if _adv_val > 100:
                _adv_amount = _adv_val * px  # 日均成交额（元）
                if _adv_amount >= _adv_threshold_high:
                    _applicable_pct = _max_order_pct_high  # 高流动性
                elif _adv_amount <= _adv_threshold_low:
                    _applicable_pct = _max_order_pct_low   # 低流动性
                else:
                    _applicable_pct = _max_order_pct_default  # 中流动性（默认档）
                max_shares_vol = int(_adv_val * _applicable_pct) // lot * lot
                shares = min(shares, max_shares_vol)
                if shares < lot:
                    continue
            # ── #4 审计修复：ADV 约束后检查余量是否够再买一手 ──
            # 原始 tv 取整后可能浪费 0-99 股 × 股价的资金；若余量足够再买一手则补加
            # 在 ADV 约束之后检查，避免补加后突破 ADV 上限
            if _adv_val > 100:
                _headroom = max_shares_vol - shares
            else:
                _headroom = shares  # 无 ADV 上限，用已有股数作为宽松上限
            if _headroom >= lot:
                _one_lot_cost = _buy_cost(
                    p["sym"], px * lot, float(lot),
                    amount_ma20=float(amount_ma20_ld[jj]) if amount_ma20_ld is not None else None,
                    dt=str(dt),
                    volatility_multiplier=float(_vol_mult_ld[jj]),
                )
                if cash >= (shares + lot) * px + _one_lot_cost:
                    shares += lot
            _limit_note = None
            # 0.3 盘中触板：仅在涨停开盘（open ≥ 涨停价）时按日级可成交量折算，
            # 正常开盘的挂单不因盘中触板而受限
            _open_at_limit_up = (
                open_arr_ld is not None and open_arr_ld[jj] >= limit_up_ld[jj] - 1e-9
            )
            # P0-6 ⑥：开盘集合竞价成交率分档（封单量/可成交量代理）——开盘价触板
            # 时，集合竞价可成交量 = 可参考成交量 × min(触板档比例, auction_fill_ratio)。
            # P1-2 修复：触板档比例改用 auction_fill_ratio_for（仅 9:25 已知信息，
            # 当日 high/low/close 不参与档位判定，消除前视）。
            # 技术债修复：经验档（limit_ratio_mode=empirical_*）改查校准表分位数，
            # 固定比例常量仅作经验单元格样本不足时的回退。
            # 假设文档化：成交价 = 开盘价（集合竞价价）；开盘后向限价收敛的盘中成交
            # 不单独建模（对买入保守：涨停开盘成交价=涨停价，且成交率受竞价档上限约束）
            _auction_ratio_buy = (
                _auction_fill_for(p["sym"], jj, side="buy")
                if (_sim_limits and _open_at_limit_up)
                else None
            )
            if _auction_ratio_buy is not None:
                _sink_inc("buy_limit_orders", 1)
            if _auction_ratio_buy is not None and _auction_ratio_buy < 1.0:
                _req = shares
                # P0.3 修复：前日无数据（停牌复牌/新股首日）不再回退当日量（前视），
                # 改用保守默认值 _DEFAULT_VOLUME_FALLBACK（5万手）。
                _pv_buy = _prev_volume.get(p["sym"], 0.0)
                _vol_ref_buy = _pv_buy if _pv_buy > 0 else _DEFAULT_VOLUME_FALLBACK
                _avail = int(_vol_ref_buy * _auction_ratio_buy) // lot * lot
                _updown = "涨停" if _limit_tag_ld[jj] == "up" else "跌停"
                if _avail < lot:
                    _sink_inc("buy_limit_rejected", 1)
                    _sink_val("buy_limit_unfilled_value", float(_req) * px)
                    _u_buy += float(_req) * px
                    logger.info(
                        f"[撮合约束/执行模型] {dt} {p['sym']} {_updown} 可成交量不足 → 未成交（买入） 请求={_req}股 可成交={_avail}股"
                    )
                    continue
                if _avail < shares:
                    _sink_inc("buy_limit_partial", 1)
                    _sink_val("buy_limit_unfilled_value", float(_req - _avail) * px)
                    _u_buy += float(_req - _avail) * px
                    shares = _avail
                    _limit_note = _limit_tag_ld[jj]
                    _day_agg["partial_buy"] += 1
                    logger.debug(
                        f"[撮合约束/执行模型] {dt} {p['sym']} {_updown} 部分成交（买入） 请求={_req}股 成交={shares}股 fill_ratio={_auction_ratio_buy:.3f}"
                    )
            tv = shares * px
            cst = _buy_cost(
                p["sym"], tv, float(shares),
                amount_ma20=float(amount_ma20_ld[jj]) if amount_ma20_ld is not None else None,
                dt=str(dt),
                volatility_multiplier=float(_vol_mult_ld[jj]),
            )
            # P1-3：现金不足时缩减到可用现金能承受的最大整手数，而非整单撤销
            if cash < tv + cst and shares >= lot:
                _affordable_lot = int((cash - cst) / px) // lot * lot
                # 费率随金额变化，迭代一次确保费用不超标
                if _affordable_lot >= lot:
                    _cst_re = _buy_cost(
                        p["sym"], _affordable_lot * px, float(_affordable_lot),
                        amount_ma20=float(amount_ma20_ld[jj]) if amount_ma20_ld is not None else None,
                        dt=str(dt),
                        volatility_multiplier=float(_vol_mult_ld[jj]),
                    )
                    if cash >= _affordable_lot * px + _cst_re:
                        shares = _affordable_lot
                    else:
                        _affordable_lot = int((cash - _cst_re) / px) // lot * lot
                        if _affordable_lot >= lot:
                            _cst_re = _buy_cost(
                                p["sym"], _affordable_lot * px, float(_affordable_lot),
                                amount_ma20=float(amount_ma20_ld[jj]) if amount_ma20_ld is not None else None,
                                dt=str(dt),
                                volatility_multiplier=float(_vol_mult_ld[jj]),
                            )
                            if cash >= _affordable_lot * px + _cst_re:
                                shares = _affordable_lot
                if shares < lot:
                    _day_agg["cash_cancel"] += 1
                    logger.debug(
                        f"[执行模型] {dt} {p['sym']} 现金不足（cash={cash:.0f} < 需要={tv + cst:.0f}）→ 连1手均不可负担，撤销"
                    )
                    continue
                tv = shares * px
                cst = _buy_cost(
                    p["sym"], tv, float(shares),
                    amount_ma20=float(amount_ma20_ld[jj]) if amount_ma20_ld is not None else None,
                    dt=str(dt),
                    volatility_multiplier=float(_vol_mult_ld[jj]),
                )
                logger.info(
                    f"[执行模型] {dt} {p['sym']} 现金不足 → 缩减股数至 {shares} 股"
                )
            elif cash < tv + cst:
                continue
            cash -= tv + cst
            pos_value[si] = tv
            if _auction_ratio_buy is not None:
                _sink_val("buy_limit_fill_value", tv)
            # P0-11：记录成交日复权因子，供除权日持仓股数调整基准（真实价格体系）
            if "adj_factor" in day_data_ld.columns:
                _af_now = float(day_data_ld["adj_factor"].values[jj])
                if np.isfinite(_af_now) and _af_now > 0:
                    _pos_adjf[p["sym"]] = _af_now
            # P0 审计修复：碎股整手防御性约束 — 确保头寸始终为整手，消除浮点精度尾差
            was_held = pos_shares[si] > 0
            pos_shares[si] = (shares // lot) * lot
            assert pos_shares[si] >= lot, f"买入后股数不足一手: {pos_shares[si]} < {lot}"
            buy_val += tv
            # P1.13 修复：部分成交/加仓时保留原始 _buy_date，防止 T+1 解锁日被连续推迟
            # 仅在首次建仓（was_held == False）时写入，部分成交连续填充不重置解锁日
            if not was_held:
                _buy_date[p["sym"]] = str(dt)
                # FIX(P1) Subtask-9：新建仓初始化持有期限计数器
                _held_days[p["sym"]] = 0
                # P0 修复（跟踪止损）：新建仓时初始化 max_close_since_entry 为入场价
                _max_close_since_entry[p["sym"]] = float(px)
                # P2 修复（跟踪止盈）：新建仓时记录入场收盘价
                _entry_close[p["sym"]] = float(close_adj[jj]) if np.isfinite(float(close_adj[jj])) else float(px)
            # P0 修复（跟踪止损）：每笔成交后（含加仓）更新持仓标的的最高收盘价
            if np.isfinite(float(close_adj[jj])):
                _max_close_since_entry[p["sym"]] = max(
                    _max_close_since_entry.get(p["sym"], float(px)),
                    float(close_adj[jj]),
                )
            # P2-2：记录建仓时 buy_score 快照，供 exit_gt 退出比较
            _entry_buy_score[p["sym"]] = float(p.get("buy_score", 0.0))
            _extra_buy = (
                {"limit": _limit_note,
                 "fill_ratio": round(float(_auction_ratio_buy), 3) if _auction_ratio_buy is not None else None}
                if _limit_note is not None
                else {}
            )
            trade_log.append(
                {
                    "time": dt,
                    "symbol": p["sym"],
                    "action": "buy",
                    "price": float(px),
                    "value": _round_half_up(tv),
                    "cost": _round_half_up(cst),
                    "qty": int(shares),
                    # 1.7 执行滞后自检字段：close 模型下为成交日真实收盘（=price）；
                    # next_open/vwap 下为信号日复权收盘价锚点（成交锚点在 exec_open）
                    "close_adj": (
                        float(px)
                        if _exec_model == "close"
                        else float(p["sig_close"])
                    ),
                    "exec_open": float(px),
                    **_extra_buy,
                }
            )
            filled += 1
        _pending_buys[:] = remaining_buys
        _sink_worst("buy_limit_worst_day", dt, _u_buy)
        _sink_worst("sell_limit_worst_day", dt, _u_sell)
        return buy_val, sell_val

    _market_multiplier = 1.0

    for i_day, (dt, grp) in enumerate(date_groups):
        # P3 审计修复：groupby 分组对象只读使用（主循环内无原地写入），不再每日
        # grp.copy() 全量冗余拷贝；PIT 掩码过滤（:1151 布尔索引）已产生新 DataFrame
        day_data = grp

        # Task F 日历轴补全日（全市场无数据的官方交易日）：无成交、无估值变动，
        # 权益按上一日市值结转（保持日轴与交易所日历 100% 对齐）
        if day_data.empty:
            # FIX(P1) Subtask-5：全市场停牌日挂单过期保护——day_data 为空时
            # _flush_pending 不会被调用，挂单 _age 不递增导致无限滞留。
            # 在此处直接遍历两个队列，递增 _age 并移除过期挂单。
            _expiry = engine_cfg.order_expiry_days
            remaining_sells = []
            for p in _pending_sells:
                p["_age"] = p.get("_age", 0) + 1
                if p.get("force"):
                    # 强平单不设过期，逐日重挂
                    remaining_sells.append(p)
                elif p["_age"] > _expiry:
                    logger.info(
                        f"[执行模型] {dt} {p['sym']} 卖出挂单过期（全市场停牌日，无行情）"
                        f"（信号日 {p['sig_dt']}，已顺延 {p['_age'] - 1} 个交易日）→ 撤销"
                    )
                else:
                    remaining_sells.append(p)
            _pending_sells[:] = remaining_sells

            remaining_buys = []
            for p in _pending_buys:
                p["_age"] = p.get("_age", 0) + 1
                if p["_age"] > _expiry:
                    logger.info(
                        f"[执行模型] {dt} {p['sym']} 买入挂单过期（全市场停牌日，无行情）"
                        f"（信号日 {p['sig_dt']}，已等待 {p['_age'] - 1} 个交易日）→ 撤销"
                    )
                else:
                    remaining_buys.append(p)
            _pending_buys[:] = remaining_buys

            equity_curve.append(
                {
                    "time": dt,
                    "portfolio_value": _round_half_up(_last_total_value),
                    "turnover": 0.0,
                }
            )
            continue

        if pit_first is not None:
            # FIX(P0) Subtask-4：PIT双向拦截——上市日前不可见 + 退市日后不可见
            sym_first = day_data["symbol"].astype(str).map(pit_first).fillna(dt)
            sym_last = day_data["symbol"].astype(str).map(pit_last).fillna(dt)
            day_data = day_data[(sym_first <= dt) & (sym_last >= dt)]
            if day_data.empty:
                # FIX(P1) Subtask-5：PIT过滤后无数据同样跳过_flush_pending，挂单需过期检查。
                _expiry_pit = engine_cfg.order_expiry_days
                _remaining_s_pit = []
                for p in _pending_sells:
                    p["_age"] = p.get("_age", 0) + 1
                    if p.get("force"):
                        _remaining_s_pit.append(p)
                    elif p["_age"] > _expiry_pit:
                        logger.info(
                            f"[执行模型] {dt} {p['sym']} 卖出挂单过期（PIT过滤后无行情）"
                            f"（信号日 {p['sig_dt']}，已顺延 {p['_age'] - 1} 个交易日）→ 撤销"
                        )
                    else:
                        _remaining_s_pit.append(p)
                _pending_sells[:] = _remaining_s_pit

                _remaining_b_pit = []
                for p in _pending_buys:
                    p["_age"] = p.get("_age", 0) + 1
                    if p["_age"] > _expiry_pit:
                        logger.info(
                            f"[执行模型] {dt} {p['sym']} 买入挂单过期（PIT过滤后无行情）"
                            f"（信号日 {p['sig_dt']}，已等待 {p['_age'] - 1} 个交易日）→ 撤销"
                        )
                    else:
                        _remaining_b_pit.append(p)
                _pending_buys[:] = _remaining_b_pit
                continue

        syms_str = day_data["symbol"].astype(str).values
        idx = np.array([sym_to_idx[s] for s in syms_str], dtype=np.int32)
        # P3 审计修复：每日本地 symbol→行索引字典（替代停牌 ADV 补 0 处
        # np.flatnonzero(syms_str == s) 嵌套循环，O(n+持仓) 替代 O(持仓×n)）
        _row_idx_day = {s: i for i, s in enumerate(syms_str)}

        # ── 价格空间统一定义（P0-12 价格空间审计修复）─────────────
        # 数据源 (IDataProvider.py) 返回:
        #   close       = close_db              → 不复权原始价
        #   close_raw   = close_db              → 不复权原始价（别名）
        #   close_normal= close_db后复权         → 后复权价（跨除权日连续）
        #
        # 信号计算 (vectorized_signal.py) 基于 close_normal（后复权），
        # 止损价 (prepare.py) 基于 close_normal（后复权），
        # 因此止损比较 / regime 状态 / sig_close 必须同在后复权空间。
        # 不复权原始价仅用于涨跌停模型（交易所限价用真实价）和显示。
        # ───────────────────────────────────────────────────────────

        # 后复权价：用于止损比较、信号比较、regime 状态、sig_close 锚点
        close_adj = day_data["close_normal"].values if "close_normal" in day_data.columns else day_data["close"].values
        # 不复权原始价：用于涨跌停模型（限价判定）、显示
        close_raw = day_data["close_raw"].values if "close_raw" in day_data.columns else close_adj
        # 后复权价合法性：负值/NaN 说明上游数据异常（如 sh600076 2024-06-24 负后复权价），
        # 该标的当日禁止买入/卖出/估值，避免负市值污染净资产
        adj_ok = np.isfinite(close_adj) & (close_adj > 0)
        # 0.4 停牌盯市：记录当日有行情（adj_ok）的真实收盘价，供停牌日估值回退
        for _k, _s in enumerate(syms_str):
            if adj_ok[_k]:
                # P1-15 修复：停牌盯市价使用复权收盘价（close_adj），与信号空间/止损空间一致；
                # 若用不复权价（close_raw），除权日停牌股估值突然跳降（如10送5→因子×1.5），
                # 导致除权日持仓市值人为缩水，净值曲线出现伪回撤。
                if np.isfinite(close_adj[_k]) and close_adj[_k] > 0:
                    _last_close[_s] = float(close_adj[_k])

        # ── P0-6 ⑤ 市场状态仓位倍率（客观状态变量，替代"前日全市场评分中位数"） ──
        # 弃用评分口径的原因：评分经 ML 覆写/阈值截断，且 0 分股拖低中位数 → 状态失真。
        # 纯价格口径（当日收盘可得 → 闭市后决策、次日开盘建仓，PIT 安全）：
        #   市场收益 = 全市场后复权收盘 ret_20d 中位数（"指数 20 日收益"代理）
        #   市场波动率 = 全市场日收益横截面 std → 过去 250 交易日分位（>p80 视为高波）
        # P0-11 注：此处刻意用复权收益（跨除权日无机械跳变，作为状态变量更稳），
        # 与成交/估值/费用的真实价口径无关（收益率为比率，两口径数值一致）。
        if i_day == 0:
            _market_multiplier = engine_cfg.regime_min_multiplier
        else:
            _cur_close: dict[str, float] = {}
            for _k, _s in enumerate(syms_str):
                if adj_ok[_k]:
                    _cur_close[_s] = float(close_adj[_k])
            for _s, _c in _cur_close.items():
                _ch = _close_hist.get(_s)
                if _ch is None:
                    _ch = deque(maxlen=_REGIME_RET_WINDOW + 1)
                    _close_hist[_s] = _ch
                _ch.append(_c)
            _ret_1d_all: list[float] = []
            _ret_20d_all: list[float] = []
            for _s, _c in _cur_close.items():
                _ch = _close_hist.get(_s)
                if _ch is None or len(_ch) < 2:
                    continue
                _ret_1d_all.append(_c / _ch[-2] - 1.0)
                if len(_ch) >= _REGIME_RET_WINDOW + 1:
                    _ret_20d_all.append(_c / _ch[0] - 1.0)
            _mkt_ret20 = float(np.median(_ret_20d_all)) if _ret_20d_all else 0.0
            _mkt_vol = float(np.std(_ret_1d_all)) if len(_ret_1d_all) > 1 else 0.0
            if np.isfinite(_mkt_vol) and _mkt_vol > 0:
                _regime_vol_hist.append(_mkt_vol)
                _vol_pct = _vol_percentile(_regime_vol_hist, _mkt_vol)
            else:
                _vol_pct = 0.0
            _market_multiplier = _regime_multiplier_for(_mkt_ret20, _vol_pct, engine_cfg)
        close = day_data["close"].values
        volume = day_data["volume"].values
        # 1.9 流动性拟真：单日振幅 = (high-low)/前收，>5% 剧烈波动日基础滑点翻倍
        high_arr = day_data["high"].values if "high" in day_data.columns else None
        low_arr = day_data["low"].values if "low" in day_data.columns else None
        # 20 日均成交额（元）：用于流动性分档冲击成本
        amount_ma20 = day_data["AMOUNT_MA20"].values if "AMOUNT_MA20" in day_data.columns else None
        buy_score = day_data["进场评分"].values
        sell_score = day_data["退出评分"].values
        risk_str = day_data["风险等级"].astype(str).values

        if i_day % 20 == 0:
            _bs_nonzero = buy_score[buy_score > 0]
            if len(_bs_nonzero) > 0:
                logger.info(
                    f"[ENGINE-SCORE] {dt}: 进场评分 非零={len(_bs_nonzero)}/{len(buy_score)} mean={_bs_nonzero.mean():.1f} median={float(np.median(_bs_nonzero)):.1f} min={_bs_nonzero.min():.0f} max={_bs_nonzero.max():.0f} >=15={int((buy_score >= 15).sum())} >=60={int((buy_score >= 60).sum())}"
                )
            else:
                logger.info(f"[ENGINE-SCORE] {dt}: 进场评分 全为零 ({len(buy_score)} 只)")

        # ── 涨跌停/停牌检查（首日无 prev_bar 时跳过 limit 过滤） ──
        # 涨跌停价 + 可成交量比例：BackTrading.limit_pricing
        # （主板 10% / ST 5% / 退市整理期首日无限制·期间 10%；创业板/科创板/北交所
        #  已由股票池过滤剔除——run_backtest_pipeline._resolve_symbols 仅保留 60x/00x，
        #  上市初期豁免 + 核准制首日 44%/-36% 由 listing_days 注入驱动）
        open_arr = day_data["open"].values if "open" in day_data.columns else None
        # P0-6 ①：整理期首日（无涨跌幅豁免）与整理期其余日（±10%）分离传递
        _df_first = _delist_first_by_day.get(str(dt), set())
        _df_period = (
            _delist_period_syms_by_day.get(str(dt), set()) - _df_first
        )
        _af_arr = day_data["adj_factor"].values if "adj_factor" in day_data.columns else None
        # P1-2（审计）：快照前日 streak 供当日开盘竞价档使用——当日收盘封板状态
        # 当日开盘不可知，直接读 _limit_streak 会把当日信息衰减当日竞价可成交量；
        # 盘中档（_build_day_limit_model 内部 :232）读取时尚未更新，本就无此问题。
        _limit_streak_prev = dict(_limit_streak)
        (
            limit_up_arr, limit_down_arr,
            at_limit_up, at_limit_down,
            not_touched_up, not_touched_down,
            _touched_up, _touched_down,
            _vol_mult, _limit_tag,
        ) = _build_day_limit_model(
            syms_str, close_raw, high_arr, low_arr,
            _prev_bar, _st_syms_by_day.get(str(dt), set()),
            str(dt), _day_idx, _listing_days_map,
            _limit_streak, _sim_limits,
            delist_first_syms=_df_first,
            delist_period_syms=_df_period,
            adj_factor_today=_af_arr,
            prev_af=_prev_af,
        )
        _have_prev = i_day > 0
        has_volume = volume > 0

        # ── 0.6 复牌跳空识别：当日有 bar 但上一交易日缺失（停牌复牌）→ 相对停牌前收盘的跳空 ──
        # 补涨（高开≥resume_gap_up）→ 开盘兑现卖出 + 当日禁买（追高）；
        # 补跌（低开）→ 日志标记（风控卖出照常）。阈值 0 = 仅识别不动作。
        resume_gap = np.full(len(syms_str), np.nan)
        if i_day > 0:
            _prev_trade_date = str(date_groups[i_day - 1][0])
            for _k, _s in enumerate(syms_str):
                _lbd = _prev_bar_date.get(_s)
                if _lbd is None or _lbd >= _prev_trade_date:
                    continue
                _pc = _prev_bar.get(_s)
                _op = open_arr[_k] if open_arr is not None else None
                if (_pc is None or _pc[0] <= 0 or _op is None
                        or not np.isfinite(_op) or _op <= 0):
                    continue
                resume_gap[_k] = _op / _pc[0] - 1.0
        resume_gap_up = np.zeros(len(syms_str), dtype=bool)
        resume_gap_down = np.zeros(len(syms_str), dtype=bool)
        if _resume_gap_up > 0:
            resume_gap_up = (resume_gap >= _resume_gap_up) & np.isfinite(resume_gap)
            for _k in np.where(resume_gap_up)[0][:20]:
                logger.info(
                    f"[复牌] {dt} {syms_str[_k]} 高开 {resume_gap[_k]*100:.2f}% → 当日禁买（追高）；持仓则开盘兑现"
                )
        if _resume_gap_down > 0:
            resume_gap_down = (resume_gap <= -_resume_gap_down) & np.isfinite(resume_gap)
            for _k in np.where(resume_gap_down)[0][:20]:
                logger.info(
                    f"[复牌] {dt} {syms_str[_k]} 低开 {resume_gap[_k]*100:.2f}% （补跌，风控卖出照常）"
                )

        stop_col = day_data["止损价"].values if "止损价" in day_data.columns else np.zeros(len(day_data))
        # P0 修复（跟踪止损 Trailing Stop）：每日盯市更新持仓标的的最高收盘价
        # 止损基准需要随股价新高而上移——这是 trailing stop 的核心语义
        held_syms = list(_max_close_since_entry.keys())
        if held_syms and adj_ok is not None:
            _syms_set = set(syms_str)
            for _hs in held_syms:
                _hi = int(np.where(syms_str == _hs)[0][0]) if len(np.where(syms_str == _hs)[0]) > 0 else -1
                if _hi >= 0 and adj_ok[_hi] and np.isfinite(float(close_adj[_hi])):
                    _max_close_since_entry[_hs] = max(_max_close_since_entry[_hs], float(close_adj[_hi]))
        # ── P0-1：止损价统一后复权口径 + 昨日止损线语义 ──
        # 止损价按当日 close_adj − ATR×mult 计算（后复权空间），同日均值比较恒不成立，
        # 必须用"上一交易日止损线 vs 今日收盘"判定破位（与 stop_hit_atr/_exit_score 的
        # shift(1) 语义一致）。日频回测无法模拟盘中最低价触及止损价后执行卖出（需要日线 low）。
        # 当前实现：收盘跌破昨日止损线 → 次日开盘卖出（execution_model=next_open 时），
        # 等价于"确认破位"而非"盘中触发"，偏保守但自洽。
        prev_stop_arr = np.array([_prev_stop.get(s, 0.0) for s in syms_str], dtype=np.float64)
        stop_hit_col = (prev_stop_arr > 0) & (close_adj < prev_stop_arr) & adj_ok
        stop_hit_atr = np.zeros(len(day_data), dtype=bool)
        if _atr_stop > 0 and "ATR" in day_data.columns and _have_prev:
            # P2-1：ATR 止损改用复权价比较，避免除权日 close_raw 机械跳降误触发
            # _prev_bar(raw) 供涨跌停模型使用不能改，此处用 _prev_bar_adj(close_adj)
            prev_close_arr = np.array([_prev_bar_adj.get(s, (c, 0))[0] for s, c in zip(syms_str, close_adj)])
            prev_atr_arr = np.array([_prev_bar_adj.get(s, (0, a))[1] for s, a in zip(syms_str, day_data["ATR"].values)])
            # P0 修复（跟踪止损 Trailing Stop）：止损基准从入场收盘价改为"持仓以来最高收盘价"
            # 原逻辑：atr_stop = prev_close_arr - prev_atr_arr * _atr_stop（固定止损，价格反弹不上移）
            # 新逻辑：atr_stop = max_close_since_entry - prev_atr_arr * _atr_stop（随新高上移，保护利润）
            max_close_arr = np.array([_max_close_since_entry.get(s, prev_close_arr[i]) for i, s in enumerate(syms_str)])
            atr_stop = max_close_arr - prev_atr_arr * _atr_stop
            stop_hit_atr = (atr_stop > 0) & (close_adj < atr_stop) & adj_ok
        # P3 审计修复：撮合（_flush_pending）之前快照"昨日可见标的"——下方 _prev_bar/
        # _prev_bar_date 立即被当日数据覆盖，若不快照，撮合内"无前收禁买"守卫
        # 读到的恒是当日数据（死守卫）；快照后守卫可正确识别从未有过前收的标的
        _prev_seen = dict(_prev_bar_date)
        if "ATR" in day_data.columns:
            for i_s, s in enumerate(syms_str):
                _prev_bar[s] = (float(close_raw[i_s]), float(day_data["ATR"].values[i_s]))
                # P2-1：同步维护复权价 prev_bar（仅 adj_ok 时更新，停牌日保留上一有效值）
                if adj_ok[i_s]:
                    _prev_bar_adj[s] = (float(close_adj[i_s]), float(day_data["ATR"].values[i_s]))
                # P1-X：同步维护 adj_factor（除权日涨跌停基准校正）
                if _af_arr is not None:
                    _af_val = _af_arr[i_s]
                    if np.isfinite(_af_val) and _af_val > 0:
                        # 审计修复 P0-1：防御 _prev_af 被异常路径置为 None
                        if _prev_af is None:
                            _prev_af = {}
                        _prev_af[s] = float(_af_val)
                _prev_bar_date[s] = str(dt)
                # P1-1（审计）：_prev_volume 更新已移至当日撮合（_flush_pending）之后
                # P2-1：有行情时重置停牌天数为 0
                if adj_ok[i_s]:
                    _susp_days[s] = 0
                else:
                    _susp_days[s] = _susp_days.get(s, 0) + 1
        else:
            for i_s, s in enumerate(syms_str):
                _prev_bar[s] = (float(close_raw[i_s]), 0.0)
                # P1-X：同步维护 adj_factor（除权日涨跌停基准校正）
                if _af_arr is not None:
                    _af_val = _af_arr[i_s]
                    if np.isfinite(_af_val) and _af_val > 0:
                        # 审计修复 P0-1：防御 _prev_af 被异常路径置为 None
                        if _prev_af is None:
                            _prev_af = {}
                        _prev_af[s] = float(_af_val)
                _prev_bar_date[s] = str(dt)
                # P2-1：有行情时重置停牌天数为 0
                if adj_ok[i_s]:
                    _susp_days[s] = 0
                else:
                    _susp_days[s] = _susp_days.get(s, 0) + 1
        # P0-1：维护上一交易日止损线（后复权空间，仅有效值更新，停牌日保留）
        # 独立于 ATR 列存在与否（无 ATR 时 stop_hit_atr 不生效，但止损价列路径必须工作）
        for i_s, s in enumerate(syms_str):
            if adj_ok[i_s]:
                _stop_v = float(stop_col[i_s])
                if np.isfinite(_stop_v) and _stop_v > 0:
                    _prev_stop[s] = _stop_v
        stop_hit = stop_hit_col | stop_hit_atr

        # ── P0-11 除权除息持仓调整（真实价格体系） ──
        # 复权估值跨除权日连续（无伪盈亏），真实价在除权日跳空。为使"真实价估值"
        # 与"复权价估值"等价，除权日（adj_factor 跳变）按比率调整持仓股数
        # （送转股 + 现金红利再投资近似；A 股现金红利年化 ~2%，再投资免佣金
        # 误差可忽略）。调整以最新一次有行情日的因子为基准，停牌期间除权在
        # 复牌日 bar 上一次性反映。挂单金额（_pending_buys tv）不随除权调整，
        # 除权后按新价自然买入更多股数（真实世界挂单价同样需要调整，影响极小）。
        # P1-20 设计决策：pos_value（持仓总成本市值）不随除权调整——
        # 除权前后总成本守恒（如10送10：100股@10元=1000元 → 200股@5元=1000元），
        # 股数调整后 pos_value/pos_shares 自动降为单位成本，无需额外操作。
        # 若同步调整 pos_value（同比例放大），会导致 pos_value/pos_shares 单位成本
        # 被高估（总成本×股数膨胀比 / 股数 = 总成本×膨胀比 / 股数 × 膨胀比），
        # 停牌估值和止盈止损比较基准失真。
        if "adj_factor" in day_data.columns:
            _af_arr = day_data["adj_factor"].values
            for _k, _s in enumerate(syms_str):
                _af_now = float(_af_arr[_k])
                if not (np.isfinite(_af_now) and _af_now > 0):
                    continue
                _pos_prev_af = _pos_adjf.get(_s)
                _si = idx[_k]
                if _pos_prev_af is not None and abs(_pos_prev_af - _af_now) > 0.01 and pos_shares[_si] > 0:
                    # ── P0-13 审计修复：adj_factor 变化分类处理 ──
                    # ── 根因分析：adj_factor 在后复权体系下有两种变化模式：
                    #    (1) 真实除权事件：因子显著跳升（ratio > 1.01，如 1.10~1.50）
                    #    (2) 数据源修正/重算：因子微调下降（ratio ≈ 1，如 0.995~0.999）
                    #    旧逻辑将两类统一视为除权事件，断言"只能增不能减"，
                    #    在数据微调场景下错误触发崩溃。
                    # ── 修复策略：
                    #    - ratio > 1.01：确认为除权事件，执行碎股清理
                    #    - 0.99 <= ratio <= 1.01：数据修正，跳过调整，更新缓存
                    #    - ratio < 0.99：异常大幅降低（可能误用前复权因子），告警但不断测
                    _ratio = _af_now / _pos_prev_af
                    _pos_adjf[_s] = _af_now  # 无论何种情况都更新缓存
                    if _ratio <= 0.99:
                        # ── 异常场景：因子大幅降低，可能误用了前复权因子 ──
                        logger.warning(
                            f"[adj_factor异常] {dt} {_s}: ratio={_ratio:.6f} "
                            f"({_pos_prev_af:.4f} → {_af_now:.4f}), "
                            f"adj_factor 大幅降低（可能误用前复权因子），"
                            f"跳过股数调整以保障回测连续性"
                        )
                        continue
                    elif _ratio <= 1.01:
                        # ── 数据修正场景：因子微调，不触发股数调整 ──
                        logger.debug(
                            f"[adj_factor修正] {dt} {_s}: ratio={_ratio:.6f} "
                            f"({_pos_prev_af:.4f} → {_af_now:.4f}), "
                            f"数据源微调，跳过股数调整"
                        )
                        continue
                    # else: ratio > 1.01，确认为真实除权事件，继续下方碎股清理逻辑
                    # 记录除权前总市值供不变性检测（碎股现金折算容差 0.5%）
                    _pre_value = pos_value[_si] + cash
                    # ── P0-2 审计修复：除权碎股合规性。
                    # 旧实现四舍五入：int(shares × ratio + 0.5)，向上偏置导致
                    # pos_value/pos_shares 单位成本低于真实成本，虚增净值。
                    # 新实现：向下取整到整手，碎股部分按真实收盘价折算现金（A股现金替代规则）。
                    # pos_value 守恒：碎股对应成本从 pos_value 中扣除，加入 cash。
                    _raw_shares = pos_shares[_si] * _ratio
                    _lot = lot_size_for(_s)
                    _sh_new = int(_raw_shares // _lot) * _lot  # 向下取整到整手
                    if _sh_new <= 0:
                        _sh_new = _lot  # 极端碎股场景兜底至少保留一手
                    # 碎股数量及现金折算
                    _fractional = _raw_shares - _sh_new
                    if _fractional > 0.001:  # 浮点容差
                        _frac_px = close_raw[_k]
                        if np.isfinite(_frac_px) and _frac_px > 0:
                            _frac_cash = _fractional * _frac_px
                            # pos_value 守恒：碎股对应成本从 pos_value 扣除
                            _old_shares = pos_shares[_si]
                            pos_value[_si] -= pos_value[_si] * (_fractional / max(_old_shares, 1))
                            cash += _frac_cash  # cash 在同一作用域，无需 nonlocal
                            # P2-3/P3-2：碎股折算追加结构化 trade_log 记录，供下游分析
                            trade_log.append({
                                "time": dt,
                                "symbol": _s,
                                "action": "fractional_cash",
                                "price": _frac_px,
                                "value": _round_half_up(_frac_cash),
                                "cost": 0.0,
                                "qty": -int(_fractional),  # 负数表示卖出碎股
                                "close_adj": float(close_adj[_k]) if _k < len(close_adj) else _frac_px,
                                "exec_open": _frac_px,
                                "fractional_cash": _round_half_up(_frac_cash),
                            })
                            logger.info(
                                f"[除权碎股现金] {dt} {_s} 碎股 {_fractional:.2f} 股 "
                                f"× {_frac_px:.2f} = {_frac_cash:.2f} 元 → 折算现金"
                            )
                    if _sh_new > 0 and _sh_new != int(pos_shares[_si]):
                        logger.info(
                            f"[除权调整] {dt} {_s} 持仓 {pos_shares[_si]:.0f} 股 "
                            f"×{_af_now / _pos_prev_af:.6f} → {_sh_new} 股"
                            f"（adj_factor {_pos_prev_af:.6f}→{_af_now:.6f}）"
                        )
                        pos_shares[_si] = _sh_new
                        # P0-13 除权日市值不变性检测（碎股现金折算容差 0.5%）
                        _post_value = pos_value[_si] + cash
                        if _pre_value > 0 and abs((_post_value - _pre_value) / _pre_value) > 0.005:
                            logger.warning(
                                f"[除权市值异常] {dt} {_s}: 除权前后市值偏差 "
                                f"{((_post_value - _pre_value) / _pre_value):.2%} "
                                f"({_pre_value:.0f} → {_post_value:.0f})，超过 0.5% 容差"
                            )
                if pos_shares[_si] > 0:
                    _pos_adjf[_s] = _af_now

        close_lookup = dict(zip(syms_str, close_raw))
        total_value = cash + _calc_market_value()
        # 0.1 执行时序：次日开盘撮合昨日挂单（先卖后买；一字板联动）
        daily_buy_value, daily_sell_value = _flush_pending(
            dt, day_data, syms_str, idx, close_adj, close_raw, open_arr,
            volume, at_limit_up, at_limit_down, limit_up_arr, limit_down_arr,
            adj_ok, has_volume,
            amount_ma20, _vol_mult, _limit_tag,
            resume_gap_up,
        )
        if daily_buy_value or daily_sell_value:
            total_value = cash + _calc_market_value()

        # P1-1（审计）：前日量必须在当日撮合（_flush_pending）之后才更新——
        # 开盘竞价可成交量只能引用前日量（当日全天量开盘时不可知，P0-11 承诺）。
        # 原实现在撮合前用当日全天量覆盖 _prev_volume（已实测证实 prev_vol=当日量），
        # 炸板日高估竞价流动性、封板日低估，违反 P0-11 前视合规契约。
        for _s_i, _s in enumerate(syms_str):
            _prev_volume[_s] = float(volume[_s_i])
            # P3 审计修复：随前日量一并记录每标的最新冲击参数，供摘牌末段强平使用
            _last_amp_mult[_s] = float(_vol_mult[_s_i])
            if amount_ma20 is not None:
                _last_amount[_s] = float(amount_ma20[_s_i])

        # 0.6 复牌高开兑现：停牌后跳空高开（补涨）→ 复牌日开盘价全部卖出（先于常规卖出）
        if _resume_gap_up > 0 and np.any(resume_gap_up):
            _resume_held = pos_shares[idx] > 0
            # P1-19 修复：复牌跳空卖出补 T+1 守卫，防止当日建仓当日复牌卖出
            _resume_t1 = np.array([str(dt) != _buy_date.get(s, "") for s in syms_str])
            _resume_sell = _resume_held & resume_gap_up & adj_ok & has_volume & _resume_t1
            si_resume = np.where(_resume_sell)[0]
            if len(si_resume):
                for _k in si_resume:
                    logger.info(
                        f"[复牌] {dt} {syms_str[_k]} 高开 {resume_gap[_k]*100:.2f}% → 开盘兑现卖出"
                    )
                # P0-6 ③：成交价用真实开盘价（与成交/现金/市值/费用统一真实价口径）；
                # 跳空识别用不复权口径（open/前收同尺度比值，不受复权影响）
                # P1-2（审计）：开盘价无效（NaN/<=0，如数据异常）→ 跳过该笔兑现，
                # 防 NaN 污染权益曲线（与卖出挂单撮合同一 _valid_px_mask 口径）
                if open_arr is not None:
                    _resume_px = open_arr[si_resume]
                else:
                    _resume_px = close_raw[si_resume]
                _resume_ok = _valid_px_mask(_resume_px)
                if not _resume_ok.all():
                    for _b in syms_str[si_resume[~_resume_ok]]:
                        logger.warning(
                            f"[复牌] {dt} {_b} 开盘价无效(NaN/<=0) → 跳过当日兑现"
                        )
                    si_resume = si_resume[_resume_ok]
                    _resume_px = _resume_px[_resume_ok]
                if len(si_resume):
                    # P1-6 修复：复牌跳空卖出流动性冲击放大
                    _resume_amp = _vol_mult[si_resume] * _resume_impact_multiplier
                    daily_sell_value += _process_sell(
                        dt,
                        syms_str[si_resume],
                        idx[si_resume],
                        _resume_px,
                        volume[si_resume],
                        partial=False,
                        s_amount=amount_ma20[si_resume] if amount_ma20 is not None else None,
                        s_amp_mult=_resume_amp,
                        s_sig_close=close_adj[si_resume],
                    )
                    close_lookup = dict(zip(syms_str[adj_ok], close_raw[adj_ok]))
                    total_value = cash + _calc_market_value()

        # ── 卖出（含 T+1 检查 + 分批止盈止损 + 摘牌日强平/ST强平） ──
        held = pos_shares[idx] > 0
        # P0-6 ① 退市整理期状态机：整理期日可正常交易（不进任何禁买/强平集合）；
        # 摘牌日（整理期最后交易日）→ 当日收盘价强平；摘牌日之后（K线滞后延伸兜底）
        # → 永久禁买 + 强平；ST/*ST 日 → 禁买 + 强平（仅 _exclude_st=True）
        _delist_last_syms = _delist_last_by_day.get(str(dt))
        _delist_last_idx = (
            np.isin(syms_str, list(_delist_last_syms)) if _delist_last_syms else None
        )
        _post_block_syms = _post_delist_block_by_day.get(str(dt))
        _post_block_idx = (
            np.isin(syms_str, list(_post_block_syms)) if _post_block_syms else None
        )
        _st_block_syms = _st_trade_block_by_day.get(str(dt))
        _st_block_idx = (
            np.isin(syms_str, list(_st_block_syms)) if _st_block_syms else None
        )
        if held.any():
            _t1_ok = np.array([str(dt) != _buy_date.get(s, "") for s in syms_str])
            exit_high = np.isin(risk_str, ["HIGH", "D"])
            # P2-2：exit_gt 与建仓时 buy_score 比较（非当日 buy_score），反映"评分反转"退出语义
            _entry_scores = np.array([_entry_buy_score.get(s, 0.0) for s in syms_str])
            # P0 修复：原 exit_gt 要求 sell_score > entry_score + 20 → 门槛极高导致盈利单几乎不退出
            # 新逻辑：固定阈值 70（信号质量反转）+ 独立跟踪止盈（max_close_since_entry 回撤）
            exit_gt = (sell_score >= 70) & (sell_score > _entry_scores + 5)
            # P0 修复：exit_score_low 从 buy_threshold//3(≈6) 提升至 buy_threshold*0.6(≈12)，
            # 同时要求持仓评分下降超过 15 分才触发（防震荡市误杀）
            exit_score_low = (
                (buy_score > 0) & (buy_score < _buy_threshold * 0.6) &
                (_entry_scores - buy_score > 15)
            )
            # P2 修复（跟踪止盈 Trailing Take-Profit）：
            # 条件：持仓浮盈超过 take_profit_pct 且从最高点回撤超过 trail_profit_ratio
            # _max_close_since_entry 在建仓/每日维护时已更新
            exit_take_profit = np.zeros(len(syms_str), dtype=bool)
            if _take_profit_pct > 0 and _trail_profit_ratio > 0:
                _entry_close_arr = np.array([_entry_close.get(s, close_adj[i]) for i, s in enumerate(syms_str)])
                _max_close_arr = np.array([_max_close_since_entry.get(s, close_adj[i]) for i, s in enumerate(syms_str)])
                # 当前浮盈比例（相对入场价）
                _pnl_pct = (close_adj - _entry_close_arr) / _entry_close_arr
                # 从最高点的回撤比例
                _drawdown_from_max = (_max_close_arr - close_adj) / _max_close_arr
                exit_take_profit = (
                    (_pnl_pct >= _take_profit_pct) &
                    (_drawdown_from_max >= _trail_profit_ratio / 100.0) &
                    adj_ok
                )
            
            # FIX(P1) Subtask-9：持有期限超限退出（无止损保护时避免永久持有）
            # 持仓每日递增 _held_days 计数器，超过 max_hold_days 后重新评估。
            _max_hold_days = int(getattr(engine_cfg, "max_hold_days", 0))
            if _max_hold_days > 0:
                for _i, _s in enumerate(syms_str):
                    if _s in _held_days:
                        _held_days[_s] += 1
                        if _held_days[_s] > _max_hold_days:
                            if not bool(held[_i]):
                                _held_days.pop(_s, None)
                                continue
                            # 超期后若 buy_score 低于阈值或为零 → 触发卖出
                            if (buy_score[_i] == 0) or (buy_score[_i] < _buy_threshold):
                                exit_score_low[_i] = True  # 复用 exit_score_low 出口卖出
                                logger.info(
                                    f"[持有期限] {dt} {_s} 已持仓{_held_days[_s]}日"
                                    f"（上限{_max_hold_days}日）→ 触发卖出"
                                )
                    # 新建仓时初始化持有天数（在卖出判定之后，与买入成交逻辑配合）
            
            # ── P0-6 ① 摘牌日强平：整理期最后交易日当日以收盘价清仓（次日无行情可卖） ──
            # 与常规卖出区分：不挂次日单（次日已无行情），当日收盘价成交是终态事件的
            # 保守近似（摘牌日不交易即按收盘价强制处置，无操纵空间）
            if _delist_last_idx is not None:
                _liq = np.where(
                    held & _delist_last_idx & _t1_ok & adj_ok & has_volume
                )[0]
                if len(_liq):
                    for _k in _liq:
                        logger.info(
                            f"[退市整理] {dt} {syms_str[_k]} 摘牌日 → 当日收盘价强平"
                        )
                    # P1-2（审计）：真实收盘价无效（NaN/<=0）→ 跳过该笔强平
                    # （由 P2-5 无行情兜底或后续 _post_block 永久禁买+强平路径接手），
                    # 防 NaN 污染权益曲线
                    _liq_px = close_raw[_liq]
                    _liq_ok = _valid_px_mask(_liq_px)
                    if not _liq_ok.all():
                        for _b in syms_str[_liq[~_liq_ok]]:
                            logger.warning(
                                f"[退市整理] {dt} {_b} 收盘价无效(NaN/<=0) → 跳过当日强平"
                            )
                        _liq = _liq[_liq_ok]
                        _liq_px = _liq_px[_liq_ok]
                    if len(_liq):
                        daily_sell_value += _process_sell(
                            dt,
                            syms_str[_liq],
                            idx[_liq],
                            # P0-11：成交价用真实收盘（统一真实价口径）
                            _liq_px,
                            volume[_liq],
                            partial=False,
                            s_amount=amount_ma20[_liq] if amount_ma20 is not None else None,
                            s_amp_mult=_vol_mult[_liq],
                            s_sig_close=close_adj[_liq],
                            s_force=True,
                        )
                close_lookup = dict(zip(syms_str[adj_ok], close_raw[adj_ok]))
                total_value = cash + _calc_market_value()
                # P2-5（审计）：摘牌日无 bar（当日行存在但无量/无有效行情）→ 终态
                # 事件强平不依赖挂单队列（挂单遇无行情只会永久滞留）：按最后有效
                # 收盘价（停牌盯市价 _last_close）当日直接清仓，记录 force_exit。
                # 摘牌日当天禁买（_st_ok 排除 _delist_last_idx），持仓必为更早买入，
                # T+1 恒满足，故不再附加 _t1_ok 条件。
                _liq_nobar = np.where(held & _delist_last_idx & ~has_volume)[0]
                if len(_liq_nobar):
                    _nb_syms = np.array([syms_str[_k] for _k in _liq_nobar], dtype=object)
                    _nb_px = np.array(
                        [_last_close.get(syms_str[_k], close_raw[_k]) for _k in _liq_nobar],
                        dtype=np.float64,
                    )
                    for _k, _p in zip(_liq_nobar, _nb_px):
                        logger.warning(
                            f"[退市整理] {dt} {syms_str[_k]} 摘牌日无行情 → 按最后有效"
                            f"收盘价 {_p:.2f} 强平（force_exit，终态事件不依赖挂单队列）"
                        )
                    # P1-2（审计）：最后有效收盘价缺失（_last_close 无记录且 raw 收盘
                    # 也无效）→ 跳过该笔强平，防 NaN 污染权益曲线
                    _nb_ok = _valid_px_mask(_nb_px)
                    if not _nb_ok.all():
                        for _b in _nb_syms[~_nb_ok]:
                            logger.warning(
                                f"[退市整理] {dt} {_b} 无任何有效收盘价 → 跳过当日强平"
                            )
                        _liq_nobar = _liq_nobar[_nb_ok]
                        _nb_syms = _nb_syms[_nb_ok]
                        _nb_px = _nb_px[_nb_ok]
                    if len(_liq_nobar):
                        daily_sell_value += _process_sell(
                            dt, _nb_syms, idx[_liq_nobar], _nb_px,
                            np.ones(len(_liq_nobar)), partial=False,
                            s_amount=amount_ma20[_liq_nobar] if amount_ma20 is not None else None,
                            s_amp_mult=_vol_mult[_liq_nobar],
                            s_force=True,
                        )
                    close_lookup = dict(zip(syms_str[adj_ok], close_raw[adj_ok]))
                    total_value = cash + _calc_market_value()

            # ── P2.1 碎股仓位强制清理守卫 ──
            # 每日本日调仓前，扫描碎股持仓并挂市价卖出单。
            # A股最小交易单位为一手（通常100股，ST/主板科创板特殊1000股）；
            # 若 0 < pos_shares < lot 则视为碎股，当日强制清仓。
            # 避免碎股长期占用 max_holdings 槽位导致策略效率下降。
            # P2-1 修复：pos_has_fractional 是全局数组，需用 idx 映射到当日股票索引
            _frac_mask = pos_has_fractional[idx] & held & adj_ok & _t1_ok & has_volume
            _frac_idx = np.where(_frac_mask)[0]
            if len(_frac_idx):
                for _k in _frac_idx:
                    _lot_frac = lot_size_for(syms_str[_k])
                    _frac_shares = int(pos_shares[_k])
                    _pending_sells.append({
                        "sym": syms_str[_k], "si": idx[_k], "partial": True,
                        "sig_dt": str(dt), "sig_close": float(close_adj[_k]),
                        "fractional_cleanup": True,
                    })
                    logger.info(
                        f"[碎股清理] {dt} {syms_str[_k]} 持仓{_frac_shares}股"
                        f"（不足{_lot_frac}股/手）→ 当日强制清仓"
                    )

# ST/退市后强平：必须离场，但 T+1 无例外（A股硬规则），无成交量则顺延
            # P0-1：修复原实现绕过 T+1 + next_open 模型下以信号日收盘价成交的合规违规
            force_exit = np.zeros(len(held), dtype=bool)
            for _bidx in (_st_block_idx, _post_block_idx):
                if _bidx is not None:
                    force_exit = force_exit | (held & _bidx & adj_ok)
            si_force = np.where(force_exit & _t1_ok & has_volume)[0]
            if len(si_force):
                # next_open/vwap：挂单次日开盘成交，与常规卖出统一执行时序；
                # 强平单遇一字跌停/停牌逐日重挂（_flush_pending 内 force 分支）
                for _k in si_force:
                    _pending_sells.append({
                        "sym": syms_str[_k], "si": idx[_k], "partial": False,
                        "sig_dt": str(dt), "sig_close": float(close_adj[_k]),
                        "force": True,
                    })
            sel_all = (
                held
                & (exit_high | exit_gt | exit_score_low | stop_hit | exit_take_profit)
                & (not_touched_down | _sim_limits)
                & has_volume
                & _t1_ok
                & adj_ok
                & ~force_exit
            )
            si_all = np.where(sel_all)[0]
            if _sim_limits:
                # 跌停无量 → 未成交（撮合约束日志，可追溯）
                _zv_sell = np.where(
                    held & _touched_down & ~has_volume & _t1_ok & adj_ok & ~force_exit
                )[0]
                for _j in _zv_sell[:20]:
                    logger.info(
                        f"[撮合约束] {dt} {syms_str[_j]} 跌停无量 → 未成交（卖出）"
                    )
            if len(si_all):
                sel_stop = held & stop_hit & (not_touched_down | _sim_limits) & has_volume & _t1_ok & adj_ok & ~force_exit
                si_stop = np.where(sel_stop)[0]
                si_partial = np.setdiff1d(si_all, si_stop)
                # 0.1 执行时序：信号日收盘决策 → 挂单次日开盘成交
                for _k in si_stop:
                    _pending_sells.append({
                        "sym": syms_str[_k], "si": idx[_k], "partial": False,
                        "sig_dt": str(dt), "sig_close": float(close_adj[_k]),
                    })
                for _k in si_partial:
                    _pending_sells.append({
                        "sym": syms_str[_k], "si": idx[_k], "partial": True,
                        "sig_dt": str(dt), "sig_close": float(close_adj[_k]),
                    })
                close_lookup = dict(zip(syms_str[adj_ok], close_raw[adj_ok]))
                total_value = cash + _calc_market_value()

        # 买入阈值判定：固定 buy_threshold（移除百分位动态门槛）。
        # FIX(P1)：去除 70 分位数动态阈值。旧逻辑在 800+ 候选池中 pct70 总是 >17，
        # 导致 _effective_threshold 被抬到 18~20+ → 每天 0 买入 → Sharpe 负值。
        # 固定使用配置 buy_threshold，保持策略行为可预测、可校准。
        _non_zero = buy_score[buy_score > 0]
        _effective_threshold = _buy_threshold
        # P0-6 ① 禁买集合：ST/*ST 日（exclude_st）+ 摘牌日 + 摘牌日之后兜底；
        # 退市整理期其余日可正常买入（整理期股票可交易）
        _st_ok = np.ones(len(syms_str), dtype=bool)
        for _bidx in (_st_block_idx, _post_block_idx, _delist_last_idx):
            if _bidx is not None:
                _st_ok = _st_ok & ~_bidx
        buy_ok = (
            (buy_score >= _effective_threshold)
            & (pos_shares[idx] == 0)
            & (~np.isin(risk_str, ["HIGH", "D", "E"]))
            & (not_touched_up | _sim_limits)
            & (~resume_gap_up)  # 0.6 复牌高开当日禁买（追高）
            & has_volume
            & adj_ok
            & _st_ok
        )
        bi = np.where(buy_ok)[0]
        if _sim_limits:
            # 涨停无量 → 未成交（撮合约束日志，可追溯）
            _zv_buy = np.where(
                _touched_up
                & ~has_volume
                & (buy_score >= _effective_threshold)
                & (pos_shares[idx] == 0)
                & (~np.isin(risk_str, ["HIGH", "D", "E"]))
                & adj_ok
                & _st_ok
            )[0]
            for _j in _zv_buy[:20]:
                logger.info(
                    f"[撮合约束] {dt} {syms_str[_j]} 涨停无量 → 未成交（买入）"
                )
        # P2-6（审计）：删除"daily_buy_value = daily_buy_value"自赋值死代码。
        # 口径说明：daily_buy_value / daily_sell_value 已由本日 _flush_pending()
        # （上方 :1311）赋值——当日挂单实际成交买入/卖出毛额；收盘模型（close）
        # 下 _flush_pending 恒返回 (0.0, 0.0)（无挂单队列，成交在当日直接记账）。
        # 后续摘牌强平/复牌兑现/本段买入均通过 daily_sell_value += / _process_sell
        # 与 _pending_buys 追加同步，故 :1567 的 turnover 统计口径自洽。
        if len(bi) == 0 and len(date_groups) > 100 and np.any(buy_score >= _buy_threshold):
            _diag_score = int((buy_score >= _effective_threshold).sum())
            _diag_pos = int((pos_shares[idx] == 0).sum())
            _diag_risk = int((~np.isin(risk_str, ["HIGH", "D", "E"])).sum())
            _diag_limit = int(not_touched_up.sum())
            _diag_vol = int(has_volume.sum())
            # P3-3：诊断日志受 optimizer_verbose 控制，WFO 路径下降噪
            _diag_log = logger.info if getattr(engine_cfg, "optimizer_verbose", False) else logger.debug
            _diag_log(
                f"[ENGINE-DIAG] {dt}: 评分≥{_effective_threshold}={_diag_score} 空仓={_diag_pos} 低风险={_diag_risk} 非涨停={_diag_limit} 有量={_diag_vol} 总={len(buy_ok)}"
            )
        # ═══════════════════════════════════════════════════════════════════
        # Market Filter (A3): 基于股票池 MA20 多头比例的市场风控开关
        # ═══════════════════════════════════════════════════════════════════
        if engine_cfg.market_filter_enabled and i_day >= 20:
            _ma20_col = day_data.get("MA_20")
            if _ma20_col is not None:
                _close_arr = day_data["close"].values
                _ma20_arr = _ma20_col.values
                _valid_mask = np.isfinite(_close_arr) & np.isfinite(_ma20_arr) & (_close_arr > 0)
                _total_valid = int(_valid_mask.sum())
                if _total_valid >= engine_cfg.market_filter_min_stocks:
                    _bull_count = int(np.sum(_close_arr[_valid_mask] > _ma20_arr[_valid_mask]))
                    _bull_ratio = _bull_count / _total_valid
                    if _bull_ratio < engine_cfg.market_filter_bull_ratio:
                        logger.info(
                            f"[市场过滤] {dt} 多头比 {_bull_ratio:.1%} < {engine_cfg.market_filter_bull_ratio:.0%} "
                            f"({_bull_count}/{_total_valid}) → 熊市模式：按多头比缩放买入仓位 "
                            f"（允许仓位比例: {_bull_ratio:.0%}，实际可买入: {max(0, int(_max_holdings * _bull_ratio))} 只）"
                        )
                        # P3 修复：从硬关断改为仓位缩放——多头比越低允许买入数越少
                        _scaled_holdings = max(1, int(_max_holdings * (_bull_ratio / engine_cfg.market_filter_bull_ratio)))
                        _prev_bi_len = len(bi)
                        if len(bi) > _scaled_holdings:
                            # 按 buy_score 降序排序后截取最强的 _scaled_holdings 只
                            _sorted_bis = bi[np.argsort(buy_score[bi])[::-1]]
                            bi = _sorted_bis[:_scaled_holdings]
                            logger.debug(f"[市场过滤] {dt} 买入信号从 {_prev_bi_len} 只缩至 {len(bi)} 只")
        # ── 买入决策 ──
        if len(bi):
            b_syms = syms_str[bi]
            b_idx = idx[bi]
            # P0-11：挂单价格基准用真实收盘（成交/现金/费用统一真实口径）
            b_close = close_raw[bi]
            b_vol = volume[bi]
            b_amount = amount_ma20[bi] if amount_ma20 is not None else None
            # A4：提取候选标的 ATR（用于风险驱动仓位控制）
            # P3-6（修复）：day_data 是 groupby 过滤后的 DataFrame，保留全量原始行号
            # 作索引（如 [120, 642, ...]）；Series[bi] 对整数数组走 label 索引，bi 是
            # 当日局部位置（0..n-1）→ KeyError。先 to_numpy() 再按位置索引，与
            # close_raw[bi]/volume[bi] 等列处理保持一致。
            _atr_col = day_data.get("ATR")
            _atr_arr = _atr_col.to_numpy()[bi] if _atr_col is not None else None

            # ── P4 组合优化器接入（数学规划驱动，替代 Top-K 等权） ──
            _use_optimizer = engine_cfg.optimizer_method != "topk_equal"

            if _use_optimizer:
                # 构建优化器配置
                _opt_cfg = OptimizerConfig(
                    method=engine_cfg.optimizer_method,
                    risk_aversion=engine_cfg.optimizer_risk_aversion,
                    turnover_penalty=engine_cfg.optimizer_turnover_penalty,
                    max_weight=engine_cfg.optimizer_max_weight,
                    cov_lookback=engine_cfg.optimizer_cov_lookback,
                    shrinkage=engine_cfg.optimizer_shrinkage,
                    max_holdings=engine_cfg.optimizer_max_holdings,
                    target_cash_ratio=engine_cfg.optimizer_target_cash,
                    max_industry_deviation=engine_cfg.optimizer_industry_deviation,
                    solve_timeout=engine_cfg.optimizer_solve_timeout,
                    verbose=getattr(engine_cfg, "optimizer_verbose", False),
                )

                # 1. 计算当前持仓权重
                _current_weights: dict[str, float] = {}
                _total_value_for_w = cash + _calc_market_value()
                if _total_value_for_w > 0:
                    for _si in range(n_syms):
                        if pos_shares[_si] > 0:
                            _px = close_lookup.get(symbols[_si]) or _last_close.get(symbols[_si], 0)
                            _current_weights[symbols[_si]] = pos_shares[_si] * _px / _total_value_for_w

                # 2. 构建收益率历史 (截至当日 T，严格 PIT；全样本尾部窗口会泄露未来)
                _candidate_list = b_syms.tolist()
                _returns_df = _build_returns_history(
                    date_groups, _candidate_list, engine_cfg.optimizer_cov_lookback, end_index=i_day
                )
                if _returns_df.empty:
                    global _OPT_EMPTY_WARNED
                    if not _OPT_EMPTY_WARNED:
                        _OPT_EMPTY_WARNED = True
                        logger.warning(
                            f"[ENGINE-OPT] {dt} 收益率历史为空（候选 {len(_candidate_list)} 只, "
                            f"lookback={engine_cfg.optimizer_cov_lookback}），优化器将回退等权"
                        )

                # 3. 行业映射 (若启用行业中性)
                _industry_map: dict[str, str] | None = None
                _bench_industry_weights: dict[str, float] | None = None
                if engine_cfg.optimizer_industry_neutral:
                    _industry_map = _load_industry_map_from_cache(_candidate_list)
                    # P1-4 修复：industry_neutral=True 但行业映射缺失时显式报错
                    if not _industry_map:
                        raise ValueError(
                            f"[ENGINE-OPT] {dt} optimizer_industry_neutral=True 但行业映射缓存为空 "
                            f"（候选 {len(_candidate_list)} 只，命中 0 只）。"
                            f"原因：_db_engine 未传入或 stock_basic_info_sw_l1 表无数据。"
                            f"修复：确保 runner 注入 db_engine 参数，或关闭 optimizer_industry_neutral"
                        )
                    # 告警：覆盖率不足但不阻断（允许部分股票无行业分类）
                    _missing_ind = len(_candidate_list) - len(_industry_map)
                    if _missing_ind > 0:
                        logger.warning(
                            f"[ENGINE-OPT] {dt} 行业映射覆盖不足: {len(_industry_map)}/{len(_candidate_list)} 只 "
                            f"（{_missing_ind} 只无行业分类，不参与行业约束）"
                        )

                # 4. Alpha 信号
                _alpha = buy_score[bi].astype(np.float64)

                # 5. 执行优化
                _optimizer = PortfolioOptimizer(_opt_cfg)
                _target_weights = _optimizer.optimize(
                    candidate_symbols=_candidate_list,
                    alpha_signals=_alpha,
                    returns_history=_returns_df,
                    current_weights=_current_weights,
                    industry_map=_industry_map,
                    benchmark_industry_weights=_bench_industry_weights,
                )

                # 6. 将优化权重映射到挂单 (权重 → 金额 → 股数)
                n_candidates = len(b_syms)
                existing = int((pos_shares > 0).sum())
                _opt_max_holdings = engine_cfg.optimizer_max_holdings
                max_new = (
                    max(0, _opt_max_holdings - existing)
                    if _opt_max_holdings > 0
                    else len(_candidate_list)
                )

                equal_weight = 0.0  # 用于 diag
                bought = 0
                for j in range(n_candidates):
                    if bought >= max_new:
                        break
                    si = b_idx[j]
                    if pos_shares[si] > 0:
                        continue
                    price = float(b_close[j])
                    if not np.isfinite(price) or price <= 0:
                        continue
                    lot = lot_size_for(b_syms[j])
                    # 优化器权重 → 目标金额
                    _w_target = _target_weights.get(b_syms[j], 0.0)
                    tv = min(
                        _total_value_for_w * _w_target,
                        total_value * max_pos_pct * _market_multiplier,
                    )
                    shares = int(tv / price) // lot * lot if price > 0 else 0
                    if shares < lot:
                        continue
                    _pending_buys.append({
                        "sym": b_syms[j], "si": si, "tv": tv,
                        "sig_dt": str(dt), "sig_close": float(b_close[j]),
                        "buy_score": float(buy_score[bi[j]]),
                    })
                    bought += 1
                    continue

                if bought == 0 and len(date_groups) > 100:
                    _p0 = float(b_close[0]) if n_candidates > 0 else 0
                    _tv0 = (
                        min(_total_value_for_w * max(_target_weights.values(), default=0),
                            total_value * max_pos_pct * _market_multiplier)
                        if n_candidates > 0
                        else 0
                    )
                    _s0 = int(_tv0 / _p0) // 100 * 100 if _p0 > 0 else 0
                    # P3-3：优化器日志受 optimizer_verbose 控制，WFO 路径下降噪
                    _opt_log = logger.info if getattr(engine_cfg, "optimizer_verbose", False) else logger.debug
                    _opt_log(
                        f"[ENGINE-DIAG/OPT] {dt}: {len(bi)}候选→{n_candidates} 0买入  "
                        f"cash={cash:.0f}  tv[0]={_tv0:.0f}  p[0]={_p0:.0f}  s[0]={_s0}  max_pos_pct={max_pos_pct}"
                    )

            else:
                # ── 兼容旧版 Top-K 等权分配 ──
                # Top-K 等权分配：集中资金到评分最高的 _top_k 只
                if len(bi) > _top_k:
                    b_scores = buy_score[bi]
                    _top_indices = np.argpartition(-b_scores, _top_k)[:_top_k]
                    b_syms = b_syms[_top_indices]
                    b_idx = b_idx[_top_indices]
                    b_close = b_close[_top_indices]
                    b_vol = b_vol[_top_indices]
                n_candidates = len(b_syms)

                existing = int((pos_shares > 0).sum())
                max_new = max(0, _max_holdings - existing) if _max_holdings > 0 else _top_k
                # P1-2：等权分母取 min(候选数, 实际可买入槽位)，避免候选>槽位时资金闲置
                _w_denom = min(n_candidates, max_new)
                equal_weight = 1.0 / _w_denom if _w_denom > 0 else 0.0
                bought = 0
                for j in range(n_candidates):
                    if bought >= max_new:
                        break
                    si = b_idx[j]
                    if pos_shares[si] > 0:
                        continue
                    price = float(b_close[j])
                    if not np.isfinite(price) or price <= 0:
                        continue
                    # 申报数量单位：按板块（科创 200 股/手，其余 100 股/手），一处定义全链路复用
                    lot = lot_size_for(b_syms[j])
                    
                    # ── A4 风险驱动仓位控制 ──
                    # 用 ATR 止损距离反推仓位，确保单笔破位亏损 = 总资金 × risk_per_trade
                    _sig_close = float(b_close[j])
                    _atr_val = _atr_arr[j] if (_atr_arr is not None and j < len(_atr_arr)) else 0.0
                    _risk_distance = _atr_val * _atr_stop if np.isfinite(_atr_val) and _atr_val > 0 else (_sig_close * 0.05)  # ATR缺失回退5%
                    _atr_tv = (cash * engine_cfg.risk_per_trade) / _risk_distance * _sig_close if _risk_distance > 0 else 0
                    
                    # 最终头寸 = min(等权/原限额, ATR风险仓位)
                    _base_tv = min(cash * equal_weight, total_value * max_pos_pct * _market_multiplier)
                    tv = min(_base_tv, _atr_tv) if _atr_tv > 0 else _base_tv
                    shares = int(tv / price) // lot * lot if price > 0 else 0
                    if shares < lot:
                        continue
                    # 0.1 执行时序：挂单次日开盘成交，撮合成本/可成交量在次日判定
                    _pending_buys.append({
                        "sym": b_syms[j], "si": si, "tv": tv,
                        "sig_dt": str(dt), "sig_close": float(b_close[j]),
                        "buy_score": float(buy_score[bi[j]]),
                    })
                    bought += 1
                    continue

                if bought == 0 and len(date_groups) > 100:
                    _p0 = float(b_close[0]) if n_candidates > 0 else 0
                    _tv0 = (
                        min(cash * equal_weight, total_value * max_pos_pct * _market_multiplier)
                        if n_candidates > 0
                        else 0
                    )
                    _s0 = int(_tv0 / _p0) // 100 * 100 if _p0 > 0 else 0
                    # P3-3：诊断日志受 optimizer_verbose 控制，WFO 路径下降噪
                    _diag_log2 = logger.info if getattr(engine_cfg, "optimizer_verbose", False) else logger.debug
                    _diag_log2(
                        f"[ENGINE-DIAG] {dt}: {len(bi)}候选→{n_candidates}TopK 0买入  cash={cash:.0f}  tv[0]={_tv0:.0f}  p[0]={_p0:.0f}  s[0]={_s0}  eq_w={equal_weight:.4f}  max_pos_pct={max_pos_pct}"
                    )

        for i_sym, i_vol in zip(syms_str, volume):
            _update_adv(i_sym, i_vol)
        # P2.7 修复：停牌/零量日 ADV 窗口不再注入 0（改为 forward fill）。
        # _update_adv 对 vol<=0 直接跳过窗口推进；此处保留调用以触发
        # 内部 short-circuit 保护，确保无异常抛出。当日有量标的已在上方正常入账。
        for _hi in np.where(pos_shares > 0)[0]:
            _s_h = symbols[_hi]
            if _s_h not in _adv_state:
                continue
            # P3 审计修复：用每日本地 symbol→行索引字典替代
            # np.flatnonzero(syms_str == _s_h) 嵌套循环（O(持仓×n)/日 → O(n+持仓)/日）
            _row_hit = _row_idx_day.get(_s_h)
            if _row_hit is None or not bool(has_volume[_row_hit]):
                _update_adv(_s_h, 0.0)

        total_value = cash + _calc_market_value()
        _turnover = (daily_buy_value + daily_sell_value) / (2 * total_value) if total_value > 0 else 0.0
        _last_total_value = total_value
        _susp_v = _susp_position_value()
        _ec_rec = {
            "time": dt,
            "portfolio_value": _round_half_up(total_value),
            "turnover": round(_turnover, 6),
        }
        if _susp_v > 0 and total_value > 0:
            # 0.4 流动性风险指标：停牌期持仓市值占比（行业标配）
            _ec_rec["susp_value_ratio"] = round(_susp_v / total_value, 6)
        # P3-2（审计）：撮合约束/现金不足每日一次汇总（仅当日有事件时输出，
        # 避免全市场回测逐笔刷屏；逐笔细节已降级 DEBUG 保留可追溯性）
        if any(_day_agg.values()):
            _agg_parts = [f"{_k}={_v}" for _k, _v in _day_agg.items() if _v]
            logger.info(f"[撮合汇总] {dt}: " + " ".join(_agg_parts))
            _day_agg = {"reject_sell": 0, "partial_sell": 0, "partial_buy": 0, "cash_cancel": 0}
        equity_curve.append(_ec_rec)
    total_value = cash + _calc_market_value()

    # P2-5（审计）：摘牌日无 bar（数据缺失——该股摘牌日当天不在 K 线中，主循环
    # 掩码无法命中）→ 主循环后终态清仓：已摘牌标的仍滞留的持仓按最后有效收盘价
    # 强制清仓（force_exit），杜绝"以最后收盘价估值挂账"的悬挂状态。
    _delisted_all: set[str] = set()
    for _ds in _delist_last_by_day.values():
        _delisted_all |= set(_ds)
    if _delisted_all:
        _tail_syms: list[str] = []
        _tail_px: list[float] = []
        for _s in _delisted_all:
            _si = sym_to_idx.get(_s)
            if _si is None or pos_shares[_si] <= 0:
                continue
            _px = _last_close.get(_s)
            if _px is None or _px <= 0:
                continue
            _tail_syms.append(_s)
            _tail_px.append(_px)
        if _tail_syms:
            _tail_dt = str(date_groups[-1][0])
            for _s, _p in zip(_tail_syms, _tail_px):
                logger.warning(
                    f"[退市整理] {_tail_dt} K线末段 {_s} 摘牌日无行情数据 → 按最后有效"
                    f"收盘价 {_p:.2f} 强制清仓（force_exit）"
                )
            # P3 审计修复：末段强平统一传入冲击参数（主循环内已逐日记录的最新值），
            # 与常规卖出口径一致；从未记录时缺省 None/1.0（与旧行为相同）
            _tail_amounts = (
                np.array([_last_amount[s] for s in _tail_syms], dtype=np.float64)
                if all(s in _last_amount for s in _tail_syms) else None
            )
            _tail_amps = (
                np.array([_last_amp_mult[s] for s in _tail_syms], dtype=np.float64)
                if all(s in _last_amp_mult for s in _tail_syms) else None
            )
            _process_sell(
                _tail_dt,
                np.array(_tail_syms, dtype=object),
                np.array([sym_to_idx[s] for s in _tail_syms], dtype=np.int32),
                np.array(_tail_px),
                np.ones(len(_tail_syms)),
                partial=False,
                s_amount=_tail_amounts,
                s_amp_mult=_tail_amps,
                s_force=True,
            )
            total_value = cash + _calc_market_value()

    # P2-1（审计）：VWAP 空间一致性断言——P0-1 后 low/high 与 amount/volume
    # 同为不复权原始口径，真实 VWAP（成交额/成交量）必落当日 [low, high]；
    # 越界即数据口径不一致（如库内为 hfq 时 VWAP 必然"越界"被误弃、回退典型价，
    # 执行模型漂移）。大量越界时集中告警，提示核对 K 线数据语义。
    if _vwap_reject_count > 0:
        logger.warning(
            f"[VWAP 口径] 本次回测 {_vwap_reject_count} 笔成交额/成交量隐含 VWAP "
            f"越出当日 [low, high] 被弃用（回退典型价）。若占比过高，请核对 K 线口径："
            f"low/high 须与 amount/volume 同为不复权原始值（P0-1 语义）"
        )

    # ── 1.7 信号执行滞后自检：收益乘数对齐（每笔成交已带同日 close_adj 字段，
    # ── 热路径 O(成交数) 校验；任何开盘/盘中价成交会吃掉当日收益） ──
    try:
        from LogicAnalyzer.ml.execution_lag_integrity import check_price_vs_close_adj

        _lag = check_price_vs_close_adj(trade_log, exec_mode=_exec_model)
        if not _lag.passed:
            logger.warning(f"[执行滞后合规] FAIL: {_lag.details[0] if _lag.details else ''}")
        else:
            logger.info(f"[执行滞后合规] 成交时序对齐 PASS（execution_model={_exec_model}，收益自成交日起计）")
    except Exception as _lag_e:
        logger.debug(f"[执行滞后合规] 自检跳过: {_lag_e}")

    # ── 1.8 交易摩擦合规自检（热路径 O(1)：显性成本/滑点下限/动态冲击） ──
    try:
        from LogicAnalyzer.ml.trading_friction_integrity import check_trading_friction_config

        _fric = check_trading_friction_config(engine_cfg)
        if not _fric.passed:
            logger.warning(f"[交易摩擦合规] FAIL: {'；'.join(_fric.details[:3])}")
        else:
            logger.info("[交易摩擦合规] 显性成本+滑点下限+动态冲击全部合规")
    except Exception as _fric_e:
        logger.debug(f"[交易摩擦合规] 自检跳过: {_fric_e}")

    # ── 成本拆解报告：各项占总成本百分比（单一来源 CostModel 汇总） ──
    try:
        _components = ("commission", "stamp", "transfer", "handling", "csrc", "slippage", "impact")
        _total_cost = sum(_cost_accum[k] for k in _components)
        _buy_v, _sell_v = _cost_accum["buy_value"], _cost_accum["sell_value"]
        if _total_cost > 0:
            _labels = {
                "commission": "佣金", "stamp": "印花税", "transfer": "过户费",
                "handling": "经手费", "csrc": "证管费", "slippage": "滑点", "impact": "冲击",
            }
            _lines = [
                f"[成本拆解] 总成本={_total_cost:.2f} 元（买入成交额 {_buy_v:.0f} / 卖出成交额 {_sell_v:.0f}，"
                f"成本率={_total_cost / max(_buy_v + _sell_v, 1e-9):.4%}）"
            ]
            for _k in _components:
                _v = _cost_accum[_k]
                _lines.append(
                    f"   {_labels[_k]}: {_v:.2f} 元 ({_v / _total_cost * 100:.2f}%)"
                )
            logger.info("\n".join(_lines))
    except Exception as _cost_e:
        logger.debug(f"[成本拆解] 报告跳过: {_cost_e}")

    return (total_value / init_cash) - 1
