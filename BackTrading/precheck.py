"""Task: 窗口预检与容错 — Precheck 模块。

在进入指标计算前判断序列是否可计算，并给出处理决策：
    OK                — 可直接计算
    SKIP              — 拒绝计算（硬性无效：全 NaN / 全零成交 / 列缺失 / 行数不足 / 不可修复缺口）
    LOW_CONFIDENCE    — 可计算但置信度低（高停牌比例 / 复权跳变 / 首发日波动 / 高零成交占比）
    NEED_FILL         — 缺口可前/后向限界填充后计算

API:
    precheck(ohlcv, params=None) -> PrecheckResult {status, reasons, metrics}
    apply_precheck(symbol, df_raw, context)  # 指标调用前的统一入口（含 SKIP 快照 + NEED_FILL 填充）
    fill_ohlcv(df, params)                   # 限界填充（每列 NaN 连续缺口 ≤ max_fill_gap）

检查项（每项产出 原因标签 + 指标值）:
    1. 非 NaN 计数            non_nan_ratio / max_consecutive_non_nan
    2. 连续非 NaN 最长段      max_consecutive_non_nan
    3. 成交量全 0 检测        zero_volume_ratio（含"无成交却有价格变动"异常）
    4. 停牌比例               suspension_ratio（日历口径：官方日历缺失日 → SKIP；
                             无日历回退启发式"零成交 + 价格横盘" → LOW_CONFIDENCE）
    5. 复权跳变检测           adj_factor 跳变 / 无量价跳变
    6. 首发日特殊判定         IPO 早期波动（上市初期无涨跌幅限制）

配置: [BACKTEST] precheck_mode = STRICT | RELAX | OFF（默认 RELAX）
    STRICT — 任何可疑一律 SKIP；RELAX — 硬失败 SKIP、可修复 NEED_FILL、软问题 LOW_CONFIDENCE；
    OFF    — 完全绕过（回退开关）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

_REQUIRED_COLS = ("open", "high", "low", "close", "volume")


class PrecheckStatus(str, Enum):
    OK = "OK"
    SKIP = "SKIP"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NEED_FILL = "NEED_FILL"


class PrecheckSeverity(str, Enum):
    """单项检查的严重级：SKIP > NEED_FILL > LOW_CONFIDENCE。"""

    SKIP = "SKIP"
    NEED_FILL = "NEED_FILL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass
class PrecheckParams:
    """预检阈值（全部可被 params dict / 配置覆盖）。"""

    mode: str = "RELAX"                  # STRICT / RELAX / OFF
    min_rows: int = 60                   # 最少行数（不足 → SKIP）
    min_non_nan_ratio: float = 0.95      # 每列非 NaN 占比下限
    min_consecutive_non_nan: int = 30    # 每列连续非 NaN 最长段下限
    max_zero_volume_ratio: float = 0.05  # 零成交占比上限（超过 → LOW_CONFIDENCE）
    max_suspension_ratio: float = 0.20   # 停牌（零成交+横盘）占比上限（超过 → LOW_CONFIDENCE）
    max_price_jump: float = 0.30         # 单日 |回报| 上限（复权跳变 / 首发日判定）
    min_trading_days_since_ipo: int = 5  # 首发保护天数（上市初期波动 → LOW_CONFIDENCE）
    fillable_nan_ratio: float = 0.10     # NEED_FILL：整列缺失占比上限
    max_fill_gap: int = 5                # NEED_FILL：单段连续 NaN 缺口长度上限
    volume_collapse_ratio: float = 0.3   # 复权跳变疑似判定：跳变日成交量为均值比例上限

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> PrecheckParams:
        if not d:
            return cls()
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class _Finding:
    label: str            # 明确原因标签，如 "ZERO_VOLUME_ALL"
    severity: PrecheckSeverity
    metric: Any = None    # 该检查的指标值


@dataclass
class PrecheckResult:
    """precheck API 返回：{status, reasons, metrics}。"""

    status: PrecheckStatus = PrecheckStatus.OK
    reasons: list[str] = field(default_factory=list)       # 全部原因标签（去重、保序）
    metrics: dict[str, Any] = field(default_factory=dict)  # 各检查指标值
    mode: str = "RELAX"
    n_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reasons": self.reasons,
            "metrics": self.metrics,
            "mode": self.mode,
            "n_rows": self.n_rows,
        }


# ── 配置读取（带兜底，配置缺失不阻塞） ────────────────────────────────

def _mode_from_config() -> str:
    try:
        from UtilsManager.ConfigParser import Config
        mode = str(Config().app_config.backtest.PRECHECK_MODE).strip().upper()
        if mode in ("STRICT", "RELAX", "OFF"):
            return mode
    except Exception:
        pass
    return "RELAX"


# ── 小工具 ────────────────────────────────────────────────────────────

def _max_consecutive_non_nan(series: pd.Series) -> int:
    if series.empty:
        return 0
    m = series.notna().astype(np.int8).to_numpy()
    diff = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0 or len(ends) == 0:
        return 0
    return int((ends - starts).max())


def _max_nan_gap(series: pd.Series) -> int:
    """最大连续 NaN 缺口长度（0 表示无缺口）。"""
    m = series.isna().astype(np.int8).to_numpy()
    diff = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0 or len(ends) == 0:
        return 0
    return int((ends - starts).max())


# ── 各检查项 ──────────────────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, findings: list[_Finding], metrics: dict[str, Any]) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    metrics["missing_columns"] = missing
    if missing:
        findings.append(_Finding("MISSING_OHLCV_COLUMNS", PrecheckSeverity.SKIP, missing))


def _check_non_nan(df: pd.DataFrame, p: PrecheckParams,
                   findings: list[_Finding], metrics: dict[str, Any]) -> None:
    col_stats: dict[str, dict[str, Any]] = {}
    for col in _REQUIRED_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        non_nan = int(s.notna().sum())
        ratio = non_nan / len(df) if len(df) else 0.0
        consec = _max_consecutive_non_nan(s)
        max_gap = _max_nan_gap(s)
        col_stats[col] = {"non_nan": non_nan, "ratio": round(ratio, 4),
                          "max_consecutive_non_nan": consec, "max_nan_gap": max_gap}
        if ratio < p.min_non_nan_ratio or consec < p.min_consecutive_non_nan:
            fillable = (max_gap <= p.max_fill_gap) and (1.0 - ratio) <= p.fillable_nan_ratio
            if fillable:
                findings.append(_Finding(f"NAN_FILLABLE_{col.upper()}", PrecheckSeverity.NEED_FILL,
                                         {"ratio": round(ratio, 4), "max_gap": max_gap}))
            else:
                findings.append(_Finding(f"NAN_TOO_HIGH_{col.upper()}", PrecheckSeverity.SKIP,
                                         {"ratio": round(ratio, 4), "max_consecutive": consec}))
    metrics["non_nan"] = col_stats


def _check_zero_volume(df: pd.DataFrame, p: PrecheckParams,
                       findings: list[_Finding], metrics: dict[str, Any]) -> None:
    if "volume" not in df.columns or len(df) == 0:
        return
    vol = df["volume"].fillna(0).to_numpy(dtype=float)
    zero_mask = vol <= 0
    zero_ratio = float(zero_mask.mean())
    metrics["zero_volume"] = {"ratio": round(zero_ratio, 4), "zero_days": int(zero_mask.sum()), "total": len(df)}
    if zero_ratio >= 1.0:
        findings.append(_Finding("ZERO_VOLUME_ALL", PrecheckSeverity.SKIP, round(zero_ratio, 4)))
    elif zero_ratio > p.max_zero_volume_ratio:
        findings.append(_Finding("ZERO_VOLUME_RATIO_HIGH", PrecheckSeverity.LOW_CONFIDENCE, round(zero_ratio, 4)))
    # 无成交却有价格变动 → 数据异常（停牌日价格应横盘）
    if "close" in df.columns and len(df) >= 2:
        close = df["close"].to_numpy(dtype=float)
        price_move_on_zero = np.zeros(len(df), dtype=bool)
        price_move_on_zero[1:] = zero_mask[1:] & ~np.isclose(close[1:], close[:-1], rtol=0, atol=1e-9)
        if price_move_on_zero.any():
            metrics["zero_volume_price_move_days"] = int(price_move_on_zero.sum())
            findings.append(_Finding("ZERO_VOLUME_PRICE_MOVE", PrecheckSeverity.SKIP,
                                     int(price_move_on_zero.sum())))


def _check_suspension(df: pd.DataFrame, p: PrecheckParams,
                      findings: list[_Finding], metrics: dict[str, Any],
                      calendar_stats: dict[str, Any] | None = None,
                      confirmed_suspension_days: set[str] | None = None) -> None:
    """停牌检查：日历口径优先（官方日历缺失日 = 停牌），无日历回退启发式。

    缺失日的两种成因（真实停牌 vs 数据源漏采）无法仅凭 K 线区分：
      - calendar_stats 带 cross_validated（已对官方停牌公告/龙虎榜交叉验证）时：
        确认停牌占比超阈值 → SKIP（真实停牌，硬拒）；仅"总缺失占比"高而确认
        停牌占比低 → UNDER_COLLECTION_SUSPECTED（漏采嫌疑，LOW_CONFIDENCE，
        不误杀可交易股票）。
      - 未交叉验证时维持原行为：总缺失占比超阈值 → SKIP，但原因明细携带
        missing_days / missing_blocks，供快照与质量日志人工复核。
      - confirmed_suspension_days 直接传入时（调用方持有独立停牌口径），对
        calendar_stats 内缺失日做交叉验证。
    无日历（对齐关闭/日历不可用）回退旧启发式：零成交 + 价格横盘 → LOW_CONFIDENCE。
    """
    if calendar_stats and calendar_stats.get("span_trading_days", 0) >= 1:
        span = int(calendar_stats["span_trading_days"])
        susp_days = list(calendar_stats.get("suspended_days") or [])
        ratio = len(susp_days) / span
        cross = bool(calendar_stats.get("cross_validated")) or confirmed_suspension_days is not None
        confirmed = list(calendar_stats.get("confirmed_days") or [])
        under = list(calendar_stats.get("under_collected_days") or [])
        if confirmed_suspension_days is not None and not calendar_stats.get("cross_validated"):
            confirmed = [d for d in susp_days if d in confirmed_suspension_days]
            under = [d for d in susp_days if d not in confirmed_suspension_days]
            cross = True
        eff_ratio = (len(confirmed) / span) if cross else ratio
        metric: dict[str, Any] = {
            "ratio": round(ratio, 4), "days": int(len(susp_days)),
            "total": span, "calendar": True,
            "missing_days": susp_days,
        }
        if calendar_stats.get("missing_blocks"):
            metric["missing_blocks"] = calendar_stats["missing_blocks"]
        if calendar_stats.get("tail_missing_days"):
            metric["tail_missing_days"] = calendar_stats["tail_missing_days"]
        metric["cross_validated"] = cross
        metric["suspension_ratio_confirmed"] = round(len(confirmed) / span, 4) if cross else None
        metric["under_collection_ratio"] = round(len(under) / span, 4) if cross else None
        metrics["suspension"] = metric
        if cross and eff_ratio <= p.max_suspension_ratio and ratio > p.max_suspension_ratio:
            # 高缺失但确认停牌占比低 → 漏采嫌疑：不硬拒，降级告警放行
            findings.append(_Finding("UNDER_COLLECTION_SUSPECTED", PrecheckSeverity.LOW_CONFIDENCE, {
                "ratio": round(ratio, 4), "confirmed_ratio": round(eff_ratio, 4),
                "under_collected_days": under,
            }))
            return
        if eff_ratio > p.max_suspension_ratio:
            _skip_metric: dict[str, Any] = {
                "ratio": round(ratio, 4), "days": int(len(susp_days)),
                "missing_days": susp_days,
            }
            if calendar_stats.get("missing_blocks"):
                _skip_metric["missing_blocks"] = calendar_stats["missing_blocks"]
            if calendar_stats.get("tail_missing_days"):
                _skip_metric["tail_missing_days"] = calendar_stats["tail_missing_days"]
            findings.append(_Finding("SUSPENSION_RATIO_HIGH_CAL", PrecheckSeverity.SKIP, _skip_metric))
        return
    if "volume" not in df.columns or "close" not in df.columns or len(df) < 2:
        return
    vol = df["volume"].fillna(0).to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    zero_mask = vol <= 0
    flat_mask = np.zeros(len(df), dtype=bool)
    flat_mask[1:] = np.isclose(close[1:], close[:-1], rtol=0, atol=1e-9)
    # #6c 审计修复：停牌 = 连续 N 日零成交+价格横盘，而非单天。
    # 低流动性微盘股正常交易日可能零成交，单天误判率偏高。
    _min_consecutive_susp = 3  # 至少连续 3 日零成交+横盘才视为停牌
    susp = np.zeros(len(df), dtype=bool)
    if _min_consecutive_susp > 0:
        _candidate = zero_mask & flat_mask
        # 统计连续段
        _diff = np.diff(np.concatenate([[0], _candidate.astype(np.int8), [0]]))
        _starts = np.where(_diff == 1)[0]
        _ends = np.where(_diff == -1)[0]
        if len(_starts) == len(_ends):
            for s, e in zip(_starts, _ends):
                if e - s >= _min_consecutive_susp:
                    susp[s:e] = True
    else:
        susp = zero_mask & flat_mask
    ratio = float(susp.mean())
    metrics["suspension"] = {"ratio": round(ratio, 4), "days": int(susp.sum()), "total": len(df)}
    if ratio > p.max_suspension_ratio:
        findings.append(_Finding("SUSPENSION_RATIO_HIGH", PrecheckSeverity.LOW_CONFIDENCE, round(ratio, 4)))


def _check_adjust_jump(df: pd.DataFrame, p: PrecheckParams,
                       findings: list[_Finding], metrics: dict[str, Any]) -> None:
    if "adj_factor" in df.columns and len(df) >= 2:
        f = df["adj_factor"].to_numpy(dtype=float)
        valid = ~np.isnan(f)
        # P1-9 审计修复：adj_factor 是累计因子，除权日正常向上跳变（5%~15%）。
        # 3% 阈值会误报所有正常送转/配权事件，改为只检测**回溯跳变**（因子不应该减小）。
        # 正常：f[t] >= f[t-1]；异常：f[t] < f[t-1] × 0.99 表示数据源断裂/混用。
        ratios = np.ones(len(f))
        ratios[1:] = np.where(
            valid[1:] & valid[:-1],
            f[1:] / np.maximum(f[:-1], 1e-12),
            1.0
        )
        # 仅检测向下异常跳变（< 99%），容忍正常除权上行跳变
        jumps_backward = ratios < 0.99
        n_jumps = int(jumps_backward.sum())
        # 同时检测极端向上跳变（> 200%，通常为数据源混用或因子初始化错误）
        jumps_extreme_up = ratios > 2.0
        n_extreme = int(jumps_extreme_up.sum())
        n_jumps += n_extreme
        metrics["adj_factor_jumps"] = n_jumps
        if n_jumps:
            metrics["adj_factor_backward_jumps"] = int(jumps_backward.sum())
            metrics["adj_factor_extreme_up_jumps"] = n_extreme
            findings.append(_Finding("ADJ_FACTOR_JUMP", PrecheckSeverity.LOW_CONFIDENCE, n_jumps))
        return
    # 无 adj_factor：量价跳变启发式（|回报| > 阈值 且 当日成交量塌缩 → 疑似未复权除权跳变）
    if "close" in df.columns and "volume" in df.columns and len(df) >= 3:
        close = df["close"].to_numpy(dtype=float)
        vol = df["volume"].fillna(0).to_numpy(dtype=float)
        prev_close = np.concatenate([[close[0]], close[:-1]])
        ret = np.abs(close / np.maximum(prev_close, 1e-12) - 1.0)
        mean_vol = np.where(vol.cumsum() == 0, 1.0, vol.mean())
        collapse = (vol / np.maximum(mean_vol, 1e-12)) < p.volume_collapse_ratio
        suspicious = (ret > p.max_price_jump) & collapse
        n = int(suspicious.sum())
        metrics["price_jump_suspicious"] = n
        if n:
            findings.append(_Finding("PRICE_JUMP_VOLUME_ANOMALY", PrecheckSeverity.LOW_CONFIDENCE, n))


def _check_first_day(df: pd.DataFrame, p: PrecheckParams,
                     findings: list[_Finding], metrics: dict[str, Any]) -> None:
    """首发日特殊判定：上市初期无涨跌幅限制，大波动属正常，但指标置信度低。"""
    if "close" not in df.columns or len(df) < 2:
        return
    close = df["close"].to_numpy(dtype=float)
    n_guard = min(p.min_trading_days_since_ipo, len(df) - 1)
    if n_guard < 1:
        return
    rets = np.abs(close[1 : n_guard + 1] / np.maximum(close[:n_guard], 1e-12) - 1.0)
    spike_idx = np.where(rets > p.max_price_jump)[0]
    metrics["ipo_early_max_return"] = round(float(rets.max()), 4) if len(rets) else 0.0
    if len(spike_idx):
        findings.append(_Finding("IPO_EARLY_VOLATILITY", PrecheckSeverity.LOW_CONFIDENCE,
                                 int(spike_idx.sum())))


# ── 决策聚合 ──────────────────────────────────────────────────────────

def _aggregate(findings: list[_Finding], mode: str, n_rows: int,
               metrics: dict[str, Any] | None = None) -> PrecheckResult:
    labels = list(dict.fromkeys(f.label for f in findings))  # 去重保序
    agg_metrics = metrics if metrics is not None else {}
    for f in findings:
        agg_metrics.setdefault(f.label, f.metric)

    if mode == "OFF":
        return PrecheckResult(status=PrecheckStatus.OK, reasons=[], metrics=agg_metrics, mode=mode, n_rows=n_rows)
    if n_rows < 1:
        return PrecheckResult(status=PrecheckStatus.SKIP, reasons=["EMPTY_SERIES"], metrics=agg_metrics, mode=mode, n_rows=n_rows)

    hard = [f for f in findings if f.severity == PrecheckSeverity.SKIP]
    if hard:
        return PrecheckResult(status=PrecheckStatus.SKIP, reasons=labels, metrics=agg_metrics, mode=mode, n_rows=n_rows)
    if mode == "STRICT":
        # 严格模式：任何可疑（可修复/软问题）一律拒绝
        if findings:
            return PrecheckResult(status=PrecheckStatus.SKIP, reasons=labels, metrics=agg_metrics, mode=mode, n_rows=n_rows)
        return PrecheckResult(status=PrecheckStatus.OK, reasons=[], metrics=agg_metrics, mode=mode, n_rows=n_rows)

    fillable = [f for f in findings if f.severity == PrecheckSeverity.NEED_FILL]
    if fillable:
        return PrecheckResult(status=PrecheckStatus.NEED_FILL, reasons=labels, metrics=agg_metrics, mode=mode, n_rows=n_rows)
    soft = [f for f in findings if f.severity == PrecheckSeverity.LOW_CONFIDENCE]
    if soft:
        return PrecheckResult(status=PrecheckStatus.LOW_CONFIDENCE, reasons=labels, metrics=agg_metrics, mode=mode, n_rows=n_rows)
    return PrecheckResult(status=PrecheckStatus.OK, reasons=[], metrics=agg_metrics, mode=mode, n_rows=n_rows)


# ── 核心 API ──────────────────────────────────────────────────────────

def precheck(
    ohlcv: pd.DataFrame | None,
    params: dict[str, Any] | PrecheckParams | None = None,
    mode: str | None = None,
    suspension_stats: dict[str, Any] | None = None,
    confirmed_suspension_days: set[str] | None = None,
) -> PrecheckResult:
    """窗口预检：precheck(ohlcv, params) -> {status, reasons, metrics}。

    Args:
        ohlcv: 原始 K 线（含 open/high/low/close/volume，可选 trade_date/symbol/adj_factor）。
        params: 阈值覆盖（dict 或 PrecheckParams；None 时用默认 + 配置 mode）。
        mode: 显式覆盖模式 STRICT/RELAX/OFF（优先于 params/config）。
        suspension_stats: Task F 日历口径停牌统计（compute_suspension_stats 单只结果；
            None 时停牌检查回退"零成交+横盘"启发式）。
        confirmed_suspension_days: 官方停牌公告/龙虎榜等独立口径确认的停牌日集合；
            提供时对缺失日做"漏采 vs 停牌"交叉验证（高缺失但确认停牌低 →
            UNDER_COLLECTION_SUSPECTED，不硬拒）。

    Returns:
        PrecheckResult（status: OK / SKIP / LOW_CONFIDENCE / NEED_FILL）。
    """
    p = params if isinstance(params, PrecheckParams) else PrecheckParams.from_dict(params)
    if mode is None:
        mode = p.mode
    if mode is None or mode not in ("STRICT", "RELAX", "OFF"):
        mode = _mode_from_config()
    p.mode = mode

    if ohlcv is None or len(ohlcv) == 0:
        return PrecheckResult(status=PrecheckStatus.SKIP, reasons=["EMPTY_SERIES"], mode=mode, n_rows=0)

    n_rows = len(ohlcv)
    findings: list[_Finding] = []
    metrics: dict[str, Any] = {}

    if n_rows < p.min_rows:
        findings.append(_Finding("TOO_FEW_ROWS", PrecheckSeverity.SKIP, n_rows))
        return _aggregate(findings, mode, n_rows, metrics)

    _check_columns(ohlcv, findings, metrics)
    _check_non_nan(ohlcv, p, findings, metrics)
    _check_zero_volume(ohlcv, p, findings, metrics)
    _check_suspension(ohlcv, p, findings, metrics, calendar_stats=suspension_stats,
                      confirmed_suspension_days=confirmed_suspension_days)
    _check_adjust_jump(ohlcv, p, findings, metrics)
    _check_first_day(ohlcv, p, findings, metrics)
    return _aggregate(findings, mode, n_rows, metrics)


# ── 容错执行 ──────────────────────────────────────────────────────────

def fill_ohlcv(df: pd.DataFrame, params: dict[str, Any] | PrecheckParams | None = None) -> pd.DataFrame:
    """前/后向限界填充：每列 NaN 缺口长度 ≤ max_fill_gap 才填充（超长缺口保持 NaN）。"""
    p = params if isinstance(params, PrecheckParams) else PrecheckParams.from_dict(params)
    out = df.copy()
    for col in _REQUIRED_COLS:
        if col not in out.columns:
            continue
        s = out[col]
        if s.isna().any():
            out[col] = s.ffill(limit=p.max_fill_gap).bfill(limit=p.max_fill_gap)
    return out


def _suspension_skip_detail(result: PrecheckResult) -> str:
    """SUSPENSION_RATIO_HIGH_CAL SKIP 的缺失日明细（供快照/日志人工复核）。"""
    if "SUSPENSION_RATIO_HIGH_CAL" not in result.reasons:
        return ""
    m = result.metrics.get("SUSPENSION_RATIO_HIGH_CAL") or {}
    md = m.get("missing_days") or []
    tail = m.get("tail_missing_days") or []
    detail = f" | 缺失交易日 {len(md)} 天: {md[:10]}{'…' if len(md) > 10 else ''}"
    if tail:
        detail += f" | 末日缺失(疑似漏采/停牌中) {len(tail)} 天: {tail[:5]}{'…' if len(tail) > 5 else ''}"
    blocks = m.get("missing_blocks") or []
    if blocks:
        loc = ",".join(f"{b['start']}~{b['end']}({b['days']}天)" for b in blocks)
        detail += f" | 缺失块[{loc}]"
    detail += " → 请人工复核（真实停牌 vs 数据源漏采，可对比官方停牌公告/龙虎榜）"
    return detail


def apply_precheck(
    symbol: str,
    df_raw: pd.DataFrame,
    context: str = "",
    params: dict[str, Any] | PrecheckParams | None = None,
    suspension_stats: dict[str, Any] | None = None,
    confirmed_suspension_days: set[str] | None = None,
) -> tuple[pd.DataFrame, PrecheckResult]:
    """指标调用前的统一 precheck 入口（集成到 prepare/indicator_cache）。

    - OK / LOW_CONFIDENCE: 返回原帧继续计算（RELAX 下软问题放行）
    - NEED_FILL: 限界填充后返回
    - SKIP: 写失败快照（precheck_status 进快照 schema），返回空帧 → 调用方跳过该股票
    - suspension_stats: Task F 日历口径停牌统计，超阈值 → SKIP；携带
      cross_validated 时按确认停牌占比判定，漏采嫌疑降级 UNDER_COLLECTION_SUSPECTED
      （无此参数回退启发式）
    - confirmed_suspension_days: 独立停牌口径（官方停牌公告/龙虎榜），交叉验证用

    Returns:
        (处理后 df, PrecheckResult)；df 为空表示调用方应跳过。
    """
    # P1 防御性断言：非主板代码进入 precheck 时立即拦截（fail-fast）
    _sym_clean = symbol.replace("sh", "").replace("sz", "")
    if _sym_clean.startswith(("300", "688")) or (
        len(_sym_clean) >= 1 and _sym_clean[0] in ("8", "4")
    ):
        skip_result = PrecheckResult(
            status=PrecheckStatus.SKIP,
            reasons=[f"NON_MAIN_BOARD_SYMBOL ({symbol})"],
            metrics={},
            n_rows=len(df_raw),
        )
        logger.warning(
            f"[{symbol}] 预检 SKIP：非主板代码不应进入主板策略 precheck "
            f"（RELAX 模式下仍拒绝）"
        )
        return pd.DataFrame(), skip_result

    result = precheck(df_raw, params, suspension_stats=suspension_stats,
                      confirmed_suspension_days=confirmed_suspension_days)
    mode = result.mode
    if mode == "OFF":
        return df_raw, result
    if result.status == PrecheckStatus.OK:
        return df_raw, result
    if result.status == PrecheckStatus.NEED_FILL:
        filled = fill_ohlcv(df_raw, params)
        logger.warning(
            f"[{symbol}] 预检 NEED_FILL（{result.reasons}），已前/后向限界填充（{context}）"
        )
        return filled, result
    if result.status == PrecheckStatus.LOW_CONFIDENCE:
        if "UNDER_COLLECTION_SUSPECTED" in result.reasons:
            _uc = result.metrics.get("UNDER_COLLECTION_SUSPECTED") or {}
            logger.warning(
                f"[{symbol}] 预检漏采嫌疑（{result.reasons}），"
                f"确认停牌占比 {_uc.get('confirmed_ratio')}，非硬拒，RELAX 放行（{context}）"
            )
        else:
            logger.warning(
                f"[{symbol}] 预检 LOW_CONFIDENCE（{result.reasons}），RELAX 模式放行（{context}）"
            )
        return df_raw, result

    # SKIP：失败写快照（A2 schema 的 precheck_status 字段）
    from BackTrading.snapshot import save_failure_snapshot

    _detail = _suspension_skip_detail(result)
    _sid = save_failure_snapshot(
        ohlcv=df_raw,
        symbol=symbol,
        metric_name="precheck",
        error_code="PRECHECK_SKIP",
        error_message=f"预检拒绝: {result.reasons}{_detail}",
        precheck_status=result.to_dict(),
    )
    _suffix = f" | snapshot_id={_sid}" if _sid else ""
    logger.warning(
        f"[{symbol}] 预检 SKIP（{result.reasons}）{_detail}{_suffix}（{context}）"
    )
    return pd.DataFrame(), result


def precheck_summary(
    kline_df: pd.DataFrame,
    params: dict[str, Any] | PrecheckParams | None = None,
) -> dict[str, int]:
    """股票池级预检摘要（回测入口统计）：{status: 数量}。空/异常时返回 {}。"""
    try:
        counts: dict[str, int] = {}
        for _sym, _g in kline_df.groupby("symbol"):
            st = precheck(_g, params).status.value
            counts[st] = counts.get(st, 0) + 1
        return counts
    except Exception as exc:
        logger.warning(f"Precheck 摘要计算失败: {exc}")
        return {}


def suspension_suspects(
    stats_by_symbol: dict[str, dict[str, Any]],
    max_suspension_ratio: float | None = None,
) -> list[dict[str, Any]]:
    """股票池级"停牌-疑似漏采"清单（供质量日志/人工复核）。

    对每只高停牌比例股票输出缺失日明细与分类（tail/interior）、交叉验证结果：
      - 未交叉验证：标注需人工复核（tail 缺失 = 同步漏采或停牌中，无法自证）；
      - 已交叉验证：给出确认停牌日与漏采嫌疑日集合。

    Args:
        stats_by_symbol: compute_suspension_stats 输出。
        max_suspension_ratio: 超过即列为疑似（None → PrecheckParams 默认 0.20）。

    Returns:
        [{"symbol", "ratio", "days", "tail_days", "interior_days",
          "missing_days", "cross_validated", "under_collected_days"?}]
    """
    if not stats_by_symbol:
        return []
    threshold = max_suspension_ratio if max_suspension_ratio is not None else PrecheckParams().max_suspension_ratio
    suspects: list[dict[str, Any]] = []
    for sym, s in stats_by_symbol.items():
        ratio = float(s.get("suspension_ratio", 0.0))
        if ratio <= threshold:
            continue
        rec: dict[str, Any] = {
            "symbol": str(sym),
            "ratio": round(ratio, 6),
            "days": int(len(s.get("suspended_days") or [])),
            "tail_days": list(s.get("tail_missing_days") or []),
            "interior_days": list(s.get("interior_missing_days") or []),
            "missing_days": list(s.get("suspended_days") or []),
            "cross_validated": bool(s.get("cross_validated", False)),
        }
        if rec["cross_validated"]:
            rec["confirmed_days"] = list(s.get("confirmed_days") or [])
            rec["under_collected_days"] = list(s.get("under_collected_days") or [])
        suspects.append(rec)
    suspects.sort(key=lambda r: (-r["ratio"], r["symbol"]))
    return suspects
