from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
import warnings
from sqlalchemy import text as sql_text


class FactorCalculator:
    """多因子 Alpha 计算引擎。

    计算质量、估值、动量、资金流四类因子 Z-Score（行业内中性化），
    与现有 MACD 评分加权融合生成新的综合分析评分。

    因子定义由 FactorRegistry（YAML 配置驱动）统一管理，
    修改 config/factor_registry.yaml 即可调整权重和参数，无需改代码。
    """

    def __init__(self, config: Any, db_engine: Any) -> None:  # noqa: ANN401
        self.config = config
        self._engine = db_engine
        from LogicAnalyzer.scoring.factor_registry import FactorRegistry

        config_dir = getattr(config, "CONFIG_DIR", None) or "config"
        registry_path = os.path.join(config_dir, "factor_registry.yaml")
        self._registry = FactorRegistry(config_path=registry_path)
        self._weights: dict[str, float] = dict(self._registry.weights)  # 快照，可被 adjust_weight 更新

    # ── 质量因子 ─────────────────────────────────────────────────

    @staticmethod
    def calc_quality_scores(df: pd.DataFrame, industry_col: str = "行业") -> pd.Series:
        """计算质量因子综合评分。

        公式: ROE × 0.4 + 毛利率 × 0.3 + 净利率 × 0.3
        然后行业内 Z-Score 标准化。
        """
        if df.empty:
            return pd.Series(dtype=float)

        composite = (
            df.get("roe", 0).fillna(0) * 0.4
            + df.get("gross_profit_margin", 0).fillna(0) * 0.3
            + df.get("net_profit_margin", 0).fillna(0) * 0.3
        )
        return FactorCalculator._industry_zscore(
            composite, df.get(industry_col, pd.Series(dtype=str))
        )

    # ── 估值因子 ─────────────────────────────────────────────────

    @staticmethod
    def calc_valuation_scores(df: pd.DataFrame, industry_col: str = "行业") -> pd.Series:
        """计算估值因子 Z-Score（行业内）。

        PE_TTM < 0 或 > 200 视为缺失；PB 同理。
        估值分数 = -(PE_TTM_z + PB_z) / 2 （高估值得低分）
        """
        if df.empty:
            return pd.Series(dtype=float)

        pe = df.get("pe_ttm", pd.Series(dtype=float))
        pb = df.get("pb", pd.Series(dtype=float))
        industry = df.get(industry_col, pd.Series(dtype=str))

        # 剔除异常值
        pe_clean = pe.where((pe > 0) & (pe <= 200), np.nan)
        pb_clean = pb.where((pb > 0) & (pb <= 100), np.nan)

        pe_z = FactorCalculator._industry_zscore(pe_clean, industry).fillna(0)
        pb_z = FactorCalculator._industry_zscore(pb_clean, industry).fillna(0)

        return -(pe_z + pb_z) / 2

    # ── 动量因子 ─────────────────────────────────────────────────

    @staticmethod
    def calc_momentum_scores(symbols: list[str], hist_df: pd.DataFrame,
                             industry_map: dict[str, str] | None = None) -> pd.Series:
        """计算 21 交易日动量（行业内中性化），向量化版本。

        Args:
            symbols: 股票代码列表（纯代码，如 600519）。
            hist_df: K线 DataFrame，必须含 symbol, trade_date, close 列。
            industry_map: {symbol: industry_name} 映射。

        Returns:
            Series: 动量 Z-Score，index 为 symbol。
        """
        if hist_df.empty:
            return pd.Series(0.0, index=symbols)

        # 向量化：过滤目标股票 → 排序 → 每组取最近 21 根 → 计算收益率
        subset = hist_df[hist_df["symbol"].isin(symbols)]
        if subset.empty:
            return pd.Series(0.0, index=symbols)

        sorted_df = subset.sort_values(["symbol", "trade_date"])
        last_21 = sorted_df.groupby("symbol").tail(21)

        first_close = last_21.groupby("symbol")["close"].first()
        last_close = last_21.groupby("symbol")["close"].last()
        momentum = ((last_close - first_close) / first_close).replace([float("inf"), -float("inf")], 0).fillna(0)

        # 确保所有请求的 symbol 都有值
        momentum = momentum.reindex(symbols, fill_value=0.0)

        if industry_map:
            aligned_ind = momentum.index.to_series().map(industry_map)
            return FactorCalculator._industry_zscore(momentum, aligned_ind).fillna(0)
        else:
            std = momentum.std()
            if std == 0:
                return momentum
            return ((momentum - momentum.mean()) / std).fillna(0)

    # ── 流动性因子 ─────────────────────────────────────────────

    @staticmethod
    def calc_liquidity_scores(symbols: list[str], hist_df: pd.DataFrame,
                               industry_map: dict[str, str] | None = None) -> pd.Series:
        """计算流动性因子 Z-Score。

        使用近 20 日平均换手率（volume / mean_volume），
        流动性越高越好（便于中频交易进出）。
        """
        if hist_df.empty:
            return pd.Series(0.0, index=symbols)
        subset = hist_df[hist_df["symbol"].isin(symbols)]
        if subset.empty:
            return pd.Series(0.0, index=symbols)
        sorted_df = subset.sort_values(["symbol", "trade_date"])
        # 相对换手率 = 当日 volume / 20日均 volume
        def _rel_turnover(grp: pd.DataFrame) -> float:
            vols = grp["volume"].tail(20)
            if len(vols) < 5:
                return 0.0
            return float(vols.iloc[-1] / max(vols.mean(), 1))
        rel_to = sorted_df.groupby("symbol").apply(_rel_turnover, include_groups=False)
        rel_to = rel_to.reindex(symbols, fill_value=0.0)
        if industry_map:
            aligned_ind = rel_to.index.to_series().map(industry_map)
            return FactorCalculator._industry_zscore(rel_to, aligned_ind).fillna(0)
        std = rel_to.std()
        return ((rel_to - rel_to.mean()) / (std if std != 0 else 1)).clip(-3, 3).fillna(0)

    # ── 波动率因子 ─────────────────────────────────────────────

    @staticmethod
    def calc_volatility_scores(symbols: list[str], hist_df: pd.DataFrame,
                                industry_map: dict[str, str] | None = None) -> pd.Series:
        """计算波动率因子 Z-Score（低波动得高分 = 防御偏好）。

        使用近 20 日收益率标准差，取负号使低波动=高分。
        """
        if hist_df.empty:
            return pd.Series(0.0, index=symbols)
        subset = hist_df[hist_df["symbol"].isin(symbols)]
        if subset.empty:
            return pd.Series(0.0, index=symbols)
        sorted_df = subset.sort_values(["symbol", "trade_date"])
        def _vol(grp: pd.DataFrame) -> float:
            prices = grp["close"].tail(20)
            if len(prices) < 5:
                return 0.0
            rets = prices.pct_change().dropna()
            return float(rets.std())
        vols = sorted_df.groupby("symbol").apply(_vol, include_groups=False)
        vols = vols.reindex(symbols, fill_value=0.0)
        low_vol_score = -vols  # 低波动→高分
        if industry_map:
            aligned_ind = low_vol_score.index.to_series().map(industry_map)
            return FactorCalculator._industry_zscore(low_vol_score, aligned_ind).fillna(0)
        std = low_vol_score.std()
        return ((low_vol_score - low_vol_score.mean()) / (std if std != 0 else 1)).clip(-3, 3).fillna(0)

    # ── 质量因子（增强版） ─────────────────────────────────────

    @staticmethod
    def calc_quality_scores(df: pd.DataFrame, industry_col: str = "行业") -> pd.Series:
        """计算质量因子综合评分（增强版）。

        公式: ROE × 0.30 + 毛利率 × 0.20 + 净利率 × 0.20 + 营收增长率 × 0.15 + 净利润增长率 × 0.15
        """
        if df.empty:
            return pd.Series(dtype=float)
        _0 = pd.Series(0.0, index=df.index)
        composite = (
            df.get("roe", _0).fillna(0) * 0.30
            + df.get("gross_profit_margin", _0).fillna(0) * 0.20
            + df.get("net_profit_margin", _0).fillna(0) * 0.20
            + df.get("revenue_growth", _0).fillna(0) * 0.15
            + df.get("net_profit_growth", _0).fillna(0) * 0.15
        )
        return FactorCalculator._industry_zscore(
            composite, df.get(industry_col, pd.Series(dtype=str))
        )

    # ── 宏观因子（行业 tilt） ──────────────────────────────────

    @staticmethod
    def calc_macro_scores(df: pd.DataFrame, macro_tilts: dict[str, float] | None = None,
                          industry_col: str = "行业") -> pd.Series:
        """计算宏观因子 Z-Score。

        宏观因子不是传统截面因子。它通过当前经济状态判断行业偏好，
        对 macro-favored 行业的股票给予正分，反之负分。
        """
        if df.empty or not macro_tilts:
            return pd.Series(0.0, index=df.index)
        industry = df.get(industry_col, pd.Series("未知", index=df.index))
        scores = industry.map(macro_tilts).fillna(0.0)
        return scores.clip(-1, 1)

    # ── 财务前瞻因子 ───────────────────────────────────────────

    @staticmethod
    def calc_financial_forward_scores(forward_df: pd.DataFrame,
                                       industry_col: str = "行业") -> pd.Series:
        """计算财务前瞻因子 Z-Score（行业内中性化）。

        使用业绩预告超预期 + 分析师共识的加权综合。
        """
        if forward_df.empty:
            return pd.Series(dtype=float)
        surprise = forward_df.get("业绩超预期分", pd.Series(0.0, index=forward_df.index)).fillna(0)
        analyst = forward_df.get("分析师共识分", pd.Series(0.0, index=forward_df.index)).fillna(0)
        # 分析师分可能存在量级差异，先 rank 归一化
        if analyst.nunique() > 1:
            analyst_rank = analyst.rank(pct=True) * 2 - 1  # [-1, 1]
        else:
            analyst_rank = analyst * 0
        composite = surprise * 0.6 + analyst_rank * 0.4
        if industry_col in forward_df.columns:
            return FactorCalculator._industry_zscore(
                composite, forward_df[industry_col]
            ).fillna(0)
        std = composite.std()
        return ((composite - composite.mean()) / (std if std != 0 else 1)).clip(-3, 3).fillna(0)

    # ── 事件驱动因子 ───────────────────────────────────────────

    @staticmethod
    def calc_event_driven_scores(event_df: pd.DataFrame,
                                  industry_col: str = "行业") -> pd.Series:
        """计算事件驱动因子 Z-Score（行业内中性化）。

        使用回购/增减持/分红的综合事件驱动分。
        """
        if event_df.empty:
            return pd.Series(dtype=float)
        score = event_df.get("事件驱动总分", pd.Series(0.0, index=event_df.index)).fillna(0)
        if industry_col in event_df.columns:
            return FactorCalculator._industry_zscore(
                score, event_df[industry_col]
            ).fillna(0)
        std = score.std()
        return ((score - score.mean()) / (std if std != 0 else 1)).clip(-3, 3).fillna(0)

    # ── 舆情因子（NLP 新闻情感） ────────────────────────────────

    @staticmethod
    def calc_news_sentiment_scores(sentiment_df: pd.DataFrame,
                                    industry_col: str = "行业") -> pd.Series:
        """计算舆情因子 Z-Score（行业内中性化）。

        使用 keyword-based 情感总分 + 新闻覆盖度。
        """
        if sentiment_df.empty:
            return pd.Series(dtype=float)
        score = sentiment_df.get("情感总分", pd.Series(0.0, index=sentiment_df.index)).fillna(0)
        count = sentiment_df.get("总新闻数", pd.Series(0)).fillna(0)
        count_norm = np.log1p(count)
        composite = score * 0.7 + count_norm * 0.3
        if industry_col in sentiment_df.columns:
            return FactorCalculator._industry_zscore(
                composite, sentiment_df[industry_col]
            ).fillna(0)
        std = composite.std()
        return ((composite - composite.mean()) / (std if std != 0 else 1)).clip(-3, 3).fillna(0)

    # ── 龙虎榜因子 ──────────────────────────────────────────────

    @staticmethod
    def calc_top_trader_scores(trader_df: pd.DataFrame, industry_col: str = "行业") -> pd.Series:
        """计算龙虎榜因子 Z-Score（行业内中性化）。

        使用 20 日净买入总额 + 上榜次数综合评分。
        """
        if trader_df.empty:
            return pd.Series(dtype=float)
        net = trader_df.get("龙虎榜净买入总额", trader_df.get("龙虎榜净买入均值", pd.Series(0.0))).fillna(0)
        count = trader_df.get("上榜总次数", pd.Series(0)).fillna(0)
        composite = net + count * 10  # 每次上榜约等于 10 万元净买入信号
        if industry_col in trader_df.columns:
            return FactorCalculator._industry_zscore(
                composite, trader_df[industry_col]
            ).fillna(0)
        std = composite.std()
        return ((composite - composite.mean()) / (std if std != 0 else 1)).clip(-3, 3).fillna(0)

    # ── 资金流因子 ───────────────────────────────────────────────

    @staticmethod
    def calc_moneyflow_scores(df: pd.DataFrame, industry_col: str = "行业") -> pd.Series:
        """从现有资金流数据计算资金流因子 Z-Score。

        使用 5 日/10 日/20 日资金流入的加权平均，行业内中性化。
        """
        if df.empty:
            return pd.Series(dtype=float)

        candidates = ["5日资金流入万元", "10日资金流入万元", "20日资金流入万元",
                       "3日资金流入万元"]
        available = [c for c in candidates if c in df.columns]
        if not available:
            return pd.Series(0.0, index=df.index)

        weights = {"3日资金流入万元": 0.3, "5日资金流入万元": 0.4,
                   "10日资金流入万元": 0.2, "20日资金流入万元": 0.1}
        total = sum(weights[c] for c in available)
        composite = sum(df[c].fillna(0) * weights[c] for c in available) / total
        return FactorCalculator._industry_zscore(
            composite, df.get(industry_col, pd.Series(dtype=str))
        ).fillna(0)

    # ── 行业内 Z-Score ──────────────────────────────────────────

    @staticmethod
    def _industry_zscore(series: pd.Series, industry: pd.Series) -> pd.Series:
        """行业内 Z-Score，clip 到 [-3, 3]。"""
        if industry.isna().all() or industry.nunique() <= 1:
            std = series.std()
            result = (series - series.mean()) / std if std != 0 else pd.Series(0, index=series.index)
            return result.clip(-3, 3).fillna(0)

        def _zscore(x: pd.Series) -> pd.Series:
            s = x.std()
            return (x - x.mean()) / s if s != 0 else pd.Series(0, index=x.index)

        result = series.groupby(industry).transform(_zscore)
        return result.clip(-3, 3).fillna(0)

    # ── 融合评分 ─────────────────────────────────────────────────

    def fuse_scores(
        self,
        report: pd.DataFrame,
        macd_score_col: str = "综合分析评分",
        industry_col: str = "行业",
        hist_df: pd.DataFrame | None = None,
        quality_df: pd.DataFrame | None = None,
        valuation_df: pd.DataFrame | None = None,
        trader_df: pd.DataFrame | None = None,
        macro_tilts: dict[str, float] | None = None,
        forward_df: pd.DataFrame | None = None,
        event_df: pd.DataFrame | None = None,
        sentiment_df: pd.DataFrame | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        """将多维因子评分融合到报告中，更新综合分析评分。

        Args:
            report: 合并处理的 DataFrame（含 MACD 评分）。
            macd_score_col: MACD 评分列名。
            industry_col: 行业列名。
            hist_df: K 线 DataFrame（用于动量因子计算）。
            quality_df: 质量因子 DataFrame（含 symbol, roe, ...）。
            valuation_df: 估值因子 DataFrame（含 symbol, pe_ttm, pb, ...）。

        Returns:
            添加了各因子评分列并更新综合分析评分的 DataFrame。
        """
        if report.empty or not self._weights:
            return report

        result = report.copy()
        logger.info(f"[v] fuse_scores start: result.shape={result.shape}, cols={list(result.columns)}")

        # 1. 将外部因子数据 merge 到 report
        if quality_df is not None and not quality_df.empty:
            q_df = quality_df.set_index("symbol")
            for col in ["roe", "gross_profit_margin", "net_profit_margin"]:
                if col in q_df.columns:
                    result[col] = result["股票代码"].map(q_df[col]).fillna(0)

        if valuation_df is not None and not valuation_df.empty:
            v_df = valuation_df.set_index("symbol")
            for col in ["pe_ttm", "pb"]:
                if col in v_df.columns:
                    result[col] = result["股票代码"].map(v_df[col]).fillna(0)

        # 2. 计算各因子评分
        quality_score = self.calc_quality_scores(result, industry_col)
        valuation_score = self.calc_valuation_scores(result, industry_col)
        moneyflow_score = self.calc_moneyflow_scores(result, industry_col)

        # 动量需要 kline
        symbols = [s for s in result["股票代码"].unique() if s]
        industry_map = (
            result.set_index("股票代码")[industry_col].to_dict()
            if industry_col in result.columns else None
        )
        momentum_score = self.calc_momentum_scores(symbols, hist_df if not hist_df.empty else pd.DataFrame(), industry_map)

        # 流动性/波动率（需要 hist_df，与 momentum 相同依赖）
        liquidity_score = self.calc_liquidity_scores(symbols, hist_df if not hist_df.empty else pd.DataFrame(), industry_map)
        volatility_score = self.calc_volatility_scores(symbols, hist_df if not hist_df.empty else pd.DataFrame(), industry_map)

        # 龙虎榜
        trader_score = pd.Series(dtype=float)
        if trader_df is not None and not trader_df.empty:
            td = trader_df.copy()
            if industry_col in result.columns:
                td["行业"] = td["symbol"].map(result.set_index("股票代码")[industry_col].to_dict()).fillna("未知")
            trader_score = self.calc_top_trader_scores(td, industry_col)
        result["龙虎榜评分"] = trader_score.reindex(result.index).fillna(0)

        # 宏观因子（行业 tilt，不是截面 Z-Score）
        # 时优先使用，否则回退二级 行业（可能多数为 0，调用方已记录降级日志）
        _macro_ind_col = "行业一级" if "行业一级" in result.columns else industry_col
        macro_score = self.calc_macro_scores(result, macro_tilts, _macro_ind_col)
        result["宏观评分"] = macro_score.reindex(result.index).fillna(0)

        # 财务前瞻
        forward_score = pd.Series(dtype=float)
        if forward_df is not None and not forward_df.empty:
            fd = forward_df.copy()
            if industry_col in result.columns:
                fd["行业"] = fd["symbol"].map(result.set_index("股票代码")[industry_col].to_dict()).fillna("未知")
            forward_score = self.calc_financial_forward_scores(fd, industry_col)
        result["财务前瞻评分"] = forward_score.reindex(result.index).fillna(0)

        # 事件驱动
        event_score = pd.Series(dtype=float)
        if event_df is not None and not event_df.empty:
            ed = event_df.copy()
            if industry_col in result.columns:
                ed["行业"] = ed["symbol"].map(result.set_index("股票代码")[industry_col].to_dict()).fillna("未知")
            event_score = self.calc_event_driven_scores(ed, industry_col)
        result["事件驱动评分"] = event_score.reindex(result.index).fillna(0)

        # 舆情因子（NLP 新闻情感）
        sentiment_score = pd.Series(dtype=float)
        if sentiment_df is not None and not sentiment_df.empty:
            sd = sentiment_df.copy()
            if industry_col in result.columns:
                sd["行业"] = sd["symbol"].map(result.set_index("股票代码")[industry_col].to_dict()).fillna("未知")
            sentiment_score = self.calc_news_sentiment_scores(sd, industry_col)
        result["舆情评分"] = sentiment_score.reindex(result.index).fillna(0)

        # 对齐索引
        result["基本面评分"] = quality_score.reindex(result.index).fillna(0)
        result["估值评分"] = valuation_score.reindex(result.index).fillna(0)
        if not result.empty:
            code_idx = result.drop_duplicates(subset="股票代码").set_index("股票代码").index
            aligned = momentum_score.reindex(code_idx)
            result["动量评分"] = result["股票代码"].map(aligned.to_dict()).fillna(0)
            _liq_ali = liquidity_score.reindex(code_idx)
            result["流动性评分"] = result["股票代码"].map(_liq_ali.to_dict()).fillna(0)
            _vol_ali = volatility_score.reindex(code_idx)
            result["波动率评分"] = result["股票代码"].map(_vol_ali.to_dict()).fillna(0)
        else:
            result["动量评分"] = 0
            result["流动性评分"] = 0
            result["波动率评分"] = 0
        result["资金流评分"] = moneyflow_score.reindex(result.index).fillna(0)

        # 3. MACD 原始评分归一化到 [-3, 3]
        raw_macd = pd.to_numeric(result.get(macd_score_col, 0), errors="coerce").fillna(0)
        macd_std = raw_macd.std()
        macd_z = ((raw_macd - raw_macd.mean()) / (macd_std if macd_std != 0 else 1)).clip(-3, 3).fillna(0)
        result["MACD评分"] = macd_z

        # 4. IC 加权融合（动态 IC 代替固定权重）
        _COL_TO_KEY = {
            "MACD评分": "macd", "动量评分": "momentum", "资金流评分": "moneyflow",
            "基本面评分": "quality", "估值评分": "valuation",
            "龙虎榜评分": "top_trader",
            "流动性评分": "liquidity", "波动率评分": "volatility",
            "宏观评分": "macro", "财务前瞻评分": "financial_forward",
            "事件驱动评分": "event_driven", "舆情评分": "news_sentiment",
        }
        _score_cols = [c for c in _COL_TO_KEY if c in result.columns]
        factor_scores = {_COL_TO_KEY[c]: result[c] for c in _score_cols}
        ic_w = self._compute_ic_weights(factor_scores, self._weights, blend_ratio=0.3)
        logger.info(f"[IC加权] 融合权重: {ic_w}")

        result["综合分析评分"] = sum(
            result[c] * ic_w.get(_COL_TO_KEY[c], 0) for c in _score_cols
        )

        # 映射回 0-100 评分：横截面百分位（非 min-max，避免极值压缩）
        result["综合分析评分"] = result["综合分析评分"].rank(pct=True) * 99 + 1

        # 行业截面百分位（用于步骤 14 过滤）
        result = self._add_industry_percentiles(result, industry_col)

        logger.info(
            "[FactorCalculator] 多因子评分融合完成，因子权重: "
            f"MACD={ic_w.get('macd',0):.2f} 动量={ic_w.get('momentum',0):.2f} "
            f"资金流={ic_w.get('moneyflow',0):.2f} 质量={ic_w.get('quality',0):.2f} "
            f"估值={ic_w.get('valuation',0):.2f} "
            f"龙虎榜={ic_w.get('top_trader',0):.2f} 流动性={ic_w.get('liquidity',0):.2f} "
            f"波动率={ic_w.get('volatility',0):.2f} 宏观={ic_w.get('macro',0):.2f} "
            f"财务前瞻={ic_w.get('financial_forward',0):.2f} 事件驱动={ic_w.get('event_driven',0):.2f}"
        )

        # 写入 DW 层宽表
        if trade_date:
            try:
                self._save_to_dwd(result, trade_date)
            except Exception:
                logger.opt(exception=True).warning("[DW层] dwd_factor_daily 写入失败")

        return result

    @staticmethod
    def _add_industry_percentiles(df: pd.DataFrame, industry_col: str = "行业") -> pd.DataFrame:
        """在 DataFrame 中添加行业截面百分位列（0-100），用于步骤 14 过滤。"""
        if industry_col not in df.columns:
            return df
        from DataManager.ColumnNames import ColumnNames as CN
        score_cols = [
            ("综合分析评分", CN.SCORE_PCT_INDUSTRY),
            ("动量评分", CN.MOMENTUM_PCT_INDUSTRY),
            ("基本面评分", CN.QUALITY_PCT_INDUSTRY),
            ("估值评分", CN.VALUATION_PCT_INDUSTRY),
            ("龙虎榜评分", CN.TOP_TRADER_PCT_INDUSTRY),
            ("流动性评分", CN.LIQUIDITY_PCT_INDUSTRY),
            ("波动率评分", CN.VOLATILITY_PCT_INDUSTRY),
            ("宏观评分", CN.MACRO_PCT_INDUSTRY),
            ("财务前瞻评分", CN.FINANCIAL_FORWARD_PCT_INDUSTRY),
            ("事件驱动评分", CN.EVENT_DRIVEN_PCT_INDUSTRY),
        ]
        for src, dst in score_cols:
            if src in df.columns:
                df[dst] = df.groupby(industry_col, observed=True)[src].rank(pct=True) * 100
            else:
                df[dst] = 50.0
        return df

    def _save_to_dwd(self, df: pd.DataFrame, trade_date: str) -> None:
        """将因子评分写入 dwd_factor_daily 宽表。"""
        import json

        FACTOR_KEYS = [
            "momentum", "quality", "valuation", "moneyflow", "macd",
            "top_trader", "liquidity", "volatility",
            "macro", "financial_forward", "event_driven",
        ]
        COL_MAP = {
            "momentum": "动量评分",
            "quality": "基本面评分",
            "valuation": "估值评分",
            "moneyflow": "资金流评分",
            "macd": "MACD评分",
            "top_trader": "龙虎榜评分",
            "liquidity": "流动性评分",
            "volatility": "波动率评分",
            "macro": "宏观评分",
            "financial_forward": "财务前瞻评分",
            "event_driven": "事件驱动评分",
        }

        if "股票代码" not in df.columns:
            return

        rows = []
        for _, r in df.iterrows():
            symbol = str(r.get("股票代码", ""))
            if not symbol:
                continue

            factors = {}
            factor_z = {}
            factor_raw = {}

            for k in FACTOR_KEYS:
                col = COL_MAP.get(k, "")
                if col in df.columns:
                    val = r.get(col)
                    if val is not None:
                        factors[k] = float(val)

            composite = r.get("综合分析评分")
            industry = r.get("行业", "")

            rows.append({
                "trade_date": trade_date,
                "symbol": symbol,
                "industry": str(industry) if pd.notna(industry) else "",
                "composite_score": float(composite) if composite is not None else None,
                "composite_rank": 0,
                "factors": json.dumps(factors),
                "factor_z": json.dumps(factor_z),
                "factor_raw": json.dumps(factor_raw),
            })

        if not rows:
            return

        from sqlalchemy import text as sql_text

        INSERT_SQL = sql_text("""
        INSERT INTO public.dwd_factor_daily
            (trade_date, symbol, industry, composite_score, composite_rank,
             factors, factor_z, factor_raw)
         VALUES
             (:trade_date, :symbol, :industry, :composite_score, :composite_rank,
              CAST(:factors AS jsonb), CAST(:factor_z AS jsonb), CAST(:factor_raw AS jsonb))
        ON CONFLICT (trade_date, symbol) DO UPDATE SET
            industry = EXCLUDED.industry,
            composite_score = EXCLUDED.composite_score,
            factors = EXCLUDED.factors,
            factor_z = EXCLUDED.factor_z,
            factor_raw = EXCLUDED.factor_raw
        """)

        with self._engine.begin() as conn:
            for row in rows:
                conn.execute(INSERT_SQL, row)
        logger.info(f"[DW层] dwd_factor_daily 写入 {len(rows)} 条")

    @staticmethod
    def _compute_ic_weights(
        factor_scores: dict[str, pd.Series],
        prior_weights: dict[str, float],
        blend_ratio: float = 0.3,
    ) -> dict[str, float]:
        """IC 加权融合 — 用 cross-sectional 一致性动态调整因子权重。

        对每个因子，计算其与其余因子等权共识的 Spearman Rank IC，
        weight = max(0, IC) / sum(max(0, IC))，再以 blend_ratio 与先验权重混合。
        """
        keys = [k for k in factor_scores if prior_weights.get(k, 0) > 0]
        if len(keys) < 3:
            return dict(prior_weights)

        scores_df = pd.DataFrame({k: factor_scores[k] for k in keys})
        ic_weights: dict[str, float] = {}

        for factor in keys:
            others = [c for c in scores_df.columns if c != factor]
            if not others:
                ic_weights[factor] = 0.0
                continue
            consensus = scores_df[others].mean(axis=1)
            valid = scores_df[factor].notna() & consensus.notna()
            if valid.sum() < 10:
                ic_weights[factor] = 0.0
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                rho, _ = spearmanr(scores_df.loc[valid, factor], consensus.loc[valid])
            ic_weights[factor] = max(0, 0.0 if np.isnan(rho) else rho)

        total_ic = sum(ic_weights.values())
        if total_ic == 0:
            return dict(prior_weights)
        ic_norm = {k: v / total_ic for k, v in ic_weights.items()}

        blended = {
            k: (1 - blend_ratio) * prior_weights.get(k, 0) + blend_ratio * ic_norm.get(k, 0)
            for k in prior_weights
        }
        bt = sum(blended.values())
        if bt > 0:
            blended = {k: v / bt for k, v in blended.items()}
        return blended

    def adjust_weight(self, factor_key: str, new_weight: float) -> None:
        """动态调整单因子权重并归一化。"""
        self._registry.adjust_weight(factor_key, new_weight)
        self._registry.normalize_weights()
        self._weights = dict(self._registry.weights)
        logger.info(f"[FactorCalculator] 因子权重调整: {factor_key} → {new_weight:.3f}，归一化后权重: {self._weights}")

    def load_quality_from_db(self, symbols: list[str] | None = None,
                             as_of: str | None = None) -> pd.DataFrame:
        """从数据库加载质量因子数据（PIT as-of 语义）。

        Args:
            as_of: 查询日，仅取该日前已披露（disclosure_date <= as_of）的财报。
        """
        try:
            from DataCollection.FinancialQualityFetcher import FinancialQualityFetcher
            fetcher = FinancialQualityFetcher(self.config)
            return fetcher.load_quality(symbols, as_of)
        except Exception as e:
            logger.warning(f"[FactorCalculator] 质量因子加载失败: {e}")
            return pd.DataFrame()

    def load_valuation_from_db(self, symbols: list[str] | None = None,
                                trade_date: str | None = None) -> pd.DataFrame:
        """从数据库加载估值因子数据。"""
        try:
            from DataCollection.FinancialValuationFetcher import FinancialValuationFetcher
            fetcher = FinancialValuationFetcher(self.config)
            return fetcher.load_latest_valuation(symbols, trade_date)
        except Exception as e:
            logger.warning(f"[FactorCalculator] 估值因子加载失败: {e}")
            return pd.DataFrame()
