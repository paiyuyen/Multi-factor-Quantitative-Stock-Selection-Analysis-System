from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from DataManager.ColumnNames import ColumnNames


# 新增长周期（如 MA200/MA250）时在此追加并同步更新 MAX_ROLLING_WINDOW。
DEFAULT_MA_PERIODS = [5, 10, 20, 30, 60, 90, 120]

# 供 prepare.py 的 MAX_INDICATOR_WINDOW 自动聚合。
MAX_ROLLING_WINDOW = max(DEFAULT_MA_PERIODS)  # 120
# 确保 MA120 等长周期指标有足够预热窗口，避免前 30 天指标统计无效。
MIN_DATA_LENGTH = MAX_ROLLING_WINDOW + 20  # 140


class TrendLevels:
    """多头排列趋势四档定级"""
    FULL_BULL = "完全主升"
    TREND_ACCELERATION = "趋势加速"
    TREND_OSCILLATION = "趋势震荡"
    TREND_WATCH = "趋势观望"

    @classmethod
    def all_levels(cls) -> list[str]:
        return [cls.FULL_BULL, cls.TREND_ACCELERATION, cls.TREND_OSCILLATION, cls.TREND_WATCH]


def calculate_full_bull_score(df: pd.DataFrame, thresholds: dict[str, int] = None) -> dict[str, Any]:
    if thresholds is None:
        thresholds = {}

    _date_col = next((c for c in ColumnNames.DATE_COLUMN_CANDIDATES if c in df.columns), None)
    if _date_col is None:
        return _generate_empty_result(f"缺少日期列，实际列: {list(df.columns)[:10]}")
    if _date_col != "trade_date":
        df = df.rename(columns={_date_col: "trade_date"})

    df["trade_date"] = df["trade_date"].astype(str).str[:10]
    df = df.sort_values("trade_date").copy()

    # 30 天时 MA60/MA90/MA120 均为 NaN，趋势评分静默失效。
    if len(df) < MIN_DATA_LENGTH:
        return _generate_empty_result(f"数据不足 {MIN_DATA_LENGTH} 个交易日（warm-up 期不足）")

    _price_col = ColumnNames.CLOSE_NORMAL if ColumnNames.CLOSE_NORMAL in df.columns else "close"
    if ColumnNames.CLOSE_NORMAL not in df.columns and _price_col == "close":
        # 不复权 close 在除权日跳变导致 MA/趋势评分严重失真，不可接受。
        logger.warning(
            f"[P1.3] Indicators calculate_full_bull_score: 标的缺失 close_normal 列，"
            f"不复权 close 降级将导致除权日指标跳变 → 强制阻断返回空结果"
        )
        return _generate_empty_result("P1.3 审计：缺失复权价格(close_normal)，拒绝使用不复权价格")

    for _period in DEFAULT_MA_PERIODS:
        _col = f"MA{_period}"
        if _col not in df.columns:
            df[_col] = df[_price_col].rolling(window=_period, min_periods=_period).mean()
    if ColumnNames.MA_VOLUME_5 not in df.columns:
        df[ColumnNames.MA_VOLUME_5] = df[ColumnNames.VOLUME_DATA].rolling(window=5, min_periods=5).mean()

    latest = df.iloc[-1]
    close_price = latest[_price_col]  # P1-10：与MA口径一致，使用复权价比较

    def _trend_skeleton_score() -> tuple[int, str]:
        ma30, ma60, ma90 = latest["MA30"], latest["MA60"], latest["MA90"]
        base_mid = (ma30 > ma60 * 0.98) and (ma60 > ma90 * 0.98)
        _ma30_prev = df["MA30"].iloc[-6]
        _ma60_prev = df["MA60"].iloc[-11]
        slope_30 = (ma30 - _ma30_prev) / _ma30_prev if abs(_ma30_prev) > 1e-6 else 0.0
        slope_60 = (ma60 - _ma60_prev) / _ma60_prev if abs(_ma60_prev) > 1e-6 else 0.0
        slope_benefit = 10 if (slope_30 > 0 and slope_60 > 0) else 0
        ma120 = latest["MA120"]
        long_up_prev = df["MA120"].iloc[-22] if len(df) >= 22 else df["MA120"].iloc[-21]
        long_up = 10 if ma120 > long_up_prev else 0
        price_pos = 10 if close_price > latest["MA20"] else 0
        total = (20 if base_mid else 0) + slope_benefit + long_up + price_pos
        desc = f"骨架得分: {total}/40 (中期:{'是' if base_mid else '否'}, 长期:{'上行' if long_up else '横盘'}, 位置:{'上方' if price_pos else '下方'})"
        return total, desc

    def _short_attack_score() -> tuple[int, str]:
        ma5, ma10, ma20 = latest["MA5"], latest["MA10"], latest["MA20"]
        standard = 15 if (ma5 > ma10 > ma20) else 0
        if len(df) >= 5:
            y = df["MA5"].iloc[-3:].values
            x = np.arange(len(y))
            if len(y) > 1:
                z = np.polyfit(x, y, 1)
                slope = z[0]
                momentum = 15 if slope > 0 else 0
            else:
                momentum = 0
        else:
            momentum = 0
        total = standard + momentum
        desc = f"攻击得分: {total}/30 (排列:{standard}, 动能:{momentum})"
        return total, desc

    def _perfect_bonus_score() -> tuple[int, str]:
        weights = [5, 3, 2]
        conditions = [latest["MA5"] > latest["MA10"], latest["MA10"] > latest["MA20"], latest["MA20"] > latest["MA30"]]
        score = sum(w for w, c in zip(weights, conditions) if c)
        above_all = 2 if close_price > max(latest["MA5"], latest["MA10"], latest["MA20"]) else 0
        total = min(10, score + above_all)
        desc = f"完美度: {total}/10 (梯度匹配)"
        return total, desc

    def _oscillation_forgive_score() -> tuple[int, str]:
        ma5, ma10 = latest["MA5"], latest["MA10"]
        vol, vol_ma5 = latest[ColumnNames.VOLUME_DATA], latest["MA_Volume_5"]
        convergence = 0
        if max(ma5, ma10) > 0:
            convergence_ratio = abs(ma5 - ma10) / max(ma5, ma10)
            convergence = 10 if convergence_ratio < 0.03 else (5 if convergence_ratio < 0.05 else 0)
        is_shrinking = (vol < vol_ma5 * 0.8) if vol_ma5 > 0 else False
        is_above_ma20 = close_price > latest["MA20"]
        volume_check = 10 if (is_shrinking and is_above_ma20) else 0
        total = convergence + volume_check
        desc = f"容错分: {total}/20 (收敛:{'达标' if convergence else '未达标'}, 缩量:{'达标' if volume_check else '未达标'})"
        return total, desc

    def _risk_control_check() -> tuple[bool, str]:
        historical_vol_window = 120
        if len(df) < historical_vol_window:
            historical_vol_window = min(len(df), 60)
        historical_volumes = df[ColumnNames.VOLUME_DATA].iloc[-historical_vol_window:]
        mean_vol = historical_volumes.mean()
        current_vol = latest[ColumnNames.VOLUME_DATA]
        vol_ratio_to_mean = current_vol / mean_vol if mean_vol > 0 else float("inf")
        liquidity_ratio_threshold = 0.2
        if vol_ratio_to_mean < liquidity_ratio_threshold:
            return (
                False,
                f"流动性枯竭 (当前量: {current_vol:.0f}, 历史均量: {mean_vol:.0f}, 比例: {vol_ratio_to_mean:.2%})",
            )
        if latest["MA30"] < latest["MA60"]:
            return False, "中期骨架塌陷"
        stock_name = str(latest.get(ColumnNames.NAME_COL, ""))
        if "ST" in stock_name or "*" in stock_name:
            return False, "ST风险"
        return True, "通过"

    is_safe, risk_reason = _risk_control_check()
    if not is_safe:
        return _generate_empty_result(f"风控拦截: {risk_reason}")

    score_trend, desc_trend = _trend_skeleton_score()
    score_attack, desc_attack = _short_attack_score()
    score_bonus, desc_bonus = _perfect_bonus_score()
    score_forgive, desc_forgive = _oscillation_forgive_score()

    total_score = score_trend + score_attack + score_bonus + score_forgive

    full_bull_threshold = thresholds.get("full_bull", 85)
    trend_acceleration_threshold = thresholds.get("trend_acceleration", 65)
    trend_oscillation_threshold = thresholds.get("trend_oscillation", 45)

    if total_score >= full_bull_threshold:
        level = TrendLevels.FULL_BULL
    elif total_score >= trend_acceleration_threshold:
        level = TrendLevels.TREND_ACCELERATION
    elif total_score >= trend_oscillation_threshold:
        level = TrendLevels.TREND_OSCILLATION
    else:
        level = TrendLevels.TREND_WATCH

    return {
        "level": level,
        "factors": {
            "trend_skeleton": {"desc": desc_trend, "score": score_trend},
            "short_attack": {"desc": desc_attack, "score": score_attack},
            "perfect_bonus": {"desc": desc_bonus, "score": score_bonus},
            "oscillation_forgive": {"desc": desc_forgive, "score": score_forgive},
        },
        "status": "SUCCESS",
    }


def _generate_empty_result(reason: str) -> dict[str, Any]:
    return {
        "level": TrendLevels.TREND_WATCH,
        "factors": {},
        "status": "FAILED",
    }
