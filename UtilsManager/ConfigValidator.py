"""
配置文件完整性校验与自动修复器。

在启动阶段对 config.ini 做全量预检并自动修复：
  - 必需 Section → 整段缺失自动追加（含注释）
  - 必需字段 → 缺失或值异常自动替换为硬编码默认值
  - 修复前备份为 config.ini.bak
  - 仅影响 ERROR 级别问题，WARNING 仅提醒不修复
"""

from __future__ import annotations

import configparser
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# ── 硬编码 Section 默认模板（含注释，修复时直接追加） ────────────────────────

DEFAULT_SECTION_TEMPLATES: dict[str, list[str]] = {
    "DATABASE": [
        "[DATABASE]",
        "user = postgres",
        "password = (请设置数据库密码)",
        "host = localhost",
        "port = 5432",
        "db_name = postgres",
    ],
    "SYSTEM": [
        "[SYSTEM]",
        "HOME_DIRECTORY = ~/Downloads/CoreNews_Reports",
        "TEMP_DATA_DIR = .",
        "max_workers = 15",
        "data_fetch_retries = 3",
        "data_fetch_delay = 5",
    ],
    "LOGGING": [
        "[LOGGING]",
        "log_level = INFO",
        "log_dir = Logs",
    ],
    "MULTI_HEAD_ARRANGEMENT": [
        "[MULTI_HEAD_ARRANGEMENT]",
        "full_bull_threshold = 85",
        "trend_acceleration_threshold = 65",
        "trend_oscillation_threshold = 45",
        "moving_average_periods = 5,10,20,30,60",
    ],
    "FILTER_RULES": [
        "[FILTER_RULES]",
        "enable_weak_stock_filter = true",
        "exempt_levels = 完全主升,趋势加速,趋势震荡,趋势观望",
    ],
    "FUND_FLOW": [
        "[FUND_FLOW]",
        "fund_flow_periods = 5,10,20",
    ],
    "TECHNICAL_INDICATORS": [
        "[TECHNICAL_INDICATORS]",
        "macd_params = 12,26,9",
    ],
    "COLUMN_ALIASES": [
        "[COLUMN_ALIASES]",
        "code_aliases = 代码=股票代码,证券代码=股票代码,股票代码=股票代码",
        "name_aliases = 名称=股票简称,股票名称=股票简称,股票简称=股票简称,简称=股票简称",
        "price_aliases = 最新价=最新价,现价=最新价,当前价格=最新价,今收盘=最新价,收盘=最新价,收盘价=最新价",
    ],
    "RESEARCH_REPORT_FILTER": [
        "[RESEARCH_REPORT_FILTER]",
        "enable_research_report_filter = false",
        "research_report_min_count = 1",
    ],
    "USER_FOCUS_STOCKS": [
        "[USER_FOCUS_STOCKS]",
        "user_focus_stocks = ",
    ],
    "FULL_BULL_SCORING": [
        "[FULL_BULL_SCORING]",
        "weight_zero_axis = 20",
        "weight_strategy_golden = 15",
        "weight_momentum = 15",
        "weight_dif_slope = 10",
        "weight_divergence = 10",
        "weight_volume_price = 10",
        "weight_kline_pattern = 10",
        "conclusion_full_bull = 80",
        "conclusion_bullish = 60",
        "conclusion_oscillate = 40",
        "rule_divergence_threshold = 0.3",
        "rule_winner_rate_high = 80",
        "rule_winner_rate_low = 15",
        "rule_cost_resistance_ratio = 0.95",
        "rule_chip_concentrated_ratio = 0.15",
        "rule_price_new_high_days = 20",
    ],
    "ASHAREHUB": [
        "[ASHAREHUB]",
        "api_key = (请设置 AShareHub API 密钥)",
    ],
    "MACRO_FILTER": [
        "[MACRO_FILTER]",
        "enable_macro_filter = true",
    ],
}


# ── 字段默认值映射（修复用） ──────────────────────────────────────────────────

FIELD_DEFAULTS: dict[str, dict[str, str]] = {sec: {} for sec in DEFAULT_SECTION_TEMPLATES}
for sec_name, tmpl_lines in DEFAULT_SECTION_TEMPLATES.items():
    for line in tmpl_lines:
        line = line.strip()
        if "=" in line and not line.startswith(";") and not line.startswith("#"):
            key, val = line.split("=", 1)
            FIELD_DEFAULTS[sec_name][key.strip()] = val.strip()


# ── 校验结果类型 ─────────────────────────────────────────────────────────────

class Severity:
    ERROR = "错误"
    WARNING = "警告"


@dataclass
class Issue:
    section: str
    field: str
    severity: str
    message: str
    actual_value: Any = None
    expected: str | None = None


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)


    def print_summary(self) -> None:
        if not self.issues:
            logger.info("config.ini 全部检查通过")
            return

        errors = [i for i in self.issues if i.severity == Severity.ERROR]
        warnings = [i for i in self.issues if i.severity == Severity.WARNING]

        if errors:
            logger.info(f"\n发现 {len(errors)} 个错误（将自动修复）：")
            for i, iss in enumerate(errors, 1):
                if iss.field:
                    print(f"  [{i}] [{iss.section}] {iss.field}: {iss.message}")
                    if iss.actual_value is not None:
                        print(f"       当前值: {iss.actual_value}")
                else:
                    print(f"  [{i}] [{iss.section}]: {iss.message}")

        if warnings:
            logger.info(f"\n发现 {len(warnings)} 个警告（不影响运行，建议修正）：")
            for i, iss in enumerate(warnings, 1):
                if iss.field:
                    print(f"  [{i}] [{iss.section}] {iss.field}: {iss.message}")
                else:
                    print(f"  [{i}] [{iss.section}]: {iss.message}")
                if iss.actual_value is not None:
                    print(f"       当前值: {iss.actual_value}")
                if iss.expected:
                    print(f"       推荐: {iss.expected}")

        logger.info("")


# ── 字段校验规则定义 ──────────────────────────────────────────────────────────

@dataclass
class FieldRule:
    name: str
    type_hint: str                    # 'int', 'bool', 'str', 'int_list'
    required: bool = True
    default: Any = None
    min_value: int | None = None
    max_value: int | None = None
    allowed_values: list[Any] | None = None
    custom_validator: str | None = None


@dataclass
class SectionRule:
    name: str
    description: str = ""
    fields: list[FieldRule] | None = None
    optional: bool = False


# ── 内置校验器 ────────────────────────────────────────────────────────────────

def _check_fund_flow_periods(value: str, issues: list[Issue], section: str) -> None:
    allowed_combos = [{3, 5, 10}, {3, 5, 20}, {5, 10, 20}, {3, 10, 20}]
    try:
        parts = [int(x.strip()) for x in value.split(",")]
    except ValueError:
        issues.append(Issue(section=section, field="fund_flow_periods", severity=Severity.ERROR,
                            message="无法解析为整数列表", actual_value=value))
        return
    if len(parts) != 3:
        issues.append(Issue(section=section, field="fund_flow_periods", severity=Severity.ERROR,
                            message=f"必须设置三个周期（当前 {len(parts)} 个）", actual_value=value))
        return
    invalid = [p for p in parts if p not in {3, 5, 10, 20}]
    if invalid:
        issues.append(Issue(section=section, field="fund_flow_periods", severity=Severity.ERROR,
                            message=f"含无效值 {invalid}", actual_value=value))
        return
    if set(parts) not in allowed_combos:
        issues.append(Issue(section=section, field="fund_flow_periods", severity=Severity.ERROR,
                            message=f"组合 {parts} 不被允许", actual_value=value))


def _check_moving_average_periods(value: str, issues: list[Issue], section: str) -> None:
    try:
        parts = [int(x.strip()) for x in value.split(",")]
    except ValueError:
        issues.append(Issue(section=section, field="moving_average_periods", severity=Severity.ERROR,
                            message="无法解析为整数列表", actual_value=value))
        return
    if len(parts) < 3 or len(parts) > 5:
        issues.append(Issue(section=section, field="moving_average_periods", severity=Severity.WARNING,
                            message=f"建议 3~5 个（当前 {len(parts)} 个）", actual_value=value))
    for i in range(len(parts) - 1):
        if parts[i] >= parts[i + 1]:
            issues.append(Issue(section=section, field="moving_average_periods", severity=Severity.WARNING,
                                message=f"非递增：{parts[i]} >= {parts[i+1]}", actual_value=value))
            break


def _check_full_bull_weights(section_data: dict[str, str], issues: list[Issue], section: str) -> None:
    weight_fields = [
        "weight_zero_axis", "weight_strategy_golden",
        "weight_momentum", "weight_dif_slope", "weight_divergence",
        "weight_volume_price", "weight_kline_pattern",
    ]
    total = 0
    for f in weight_fields:
        raw = section_data.get(f, "").strip()
        if not raw:
            continue
        try:
            total += int(raw)
        except ValueError:
            pass
    if total != 0 and total != 90:
        issues.append(Issue(section=section, field="(权重合计)", severity=Severity.WARNING,
                            message=f"7 个权重总和 {total}，建议 90", actual_value=str(total)))


def _check_full_bull_thresholds(section_data: dict[str, str], issues: list[Issue], section: str) -> None:
    try:
        full = int(section_data.get("conclusion_full_bull", "0").strip() or "0")
        bullish = int(section_data.get("conclusion_bullish", "0").strip() or "0")
        osc = int(section_data.get("conclusion_oscillate", "0").strip() or "0")
    except ValueError:
        return
    if not (full >= bullish >= osc):
        issues.append(Issue(section=section, field="(阈值大小关系)", severity=Severity.WARNING,
                            message=f"需 full({full}) >= bullish({bullish}) >= oscillate({osc})"))


# ── Section 规则定义 ─────────────────────────────────────────────────────────

SECTION_RULES: list[SectionRule] = [
    SectionRule(name="DATABASE", description="数据库连接配置", fields=[
        FieldRule("user", "str"), FieldRule("password", "str"),
        FieldRule("host", "str"), FieldRule("port", "str"), FieldRule("db_name", "str"),
    ]),
    SectionRule(name="SYSTEM", description="系统运行参数", fields=[
        FieldRule("HOME_DIRECTORY", "str", required=False, default="~/Downloads/CoreNews_Reports"),
        FieldRule("TEMP_DATA_DIR", "str", required=False, default="."),
        FieldRule("max_workers", "int", required=False, default="15", min_value=1, max_value=64),
        FieldRule("data_fetch_retries", "int", required=False, default="3", min_value=1, max_value=10),
        FieldRule("data_fetch_delay", "int", required=False, default="5", min_value=1, max_value=60),
        FieldRule("signal_processing_processes", "int", required=False, default="(auto)", min_value=1, max_value=64),
    ]),
    SectionRule(name="LOGGING", description="日志配置", fields=[
        FieldRule("log_level", "str", required=False, default="INFO", allowed_values=["DEBUG", "INFO", "WARNING", "ERROR"]),
        FieldRule("log_dir", "str", required=False, default="Logs"),
    ]),
    SectionRule(name="MULTI_HEAD_ARRANGEMENT", description="多头排列评分系统配置", fields=[
        FieldRule("full_bull_threshold", "int", required=False, default="85", min_value=0, max_value=100),
        FieldRule("trend_acceleration_threshold", "int", required=False, default="65", min_value=0, max_value=100),
        FieldRule("trend_oscillation_threshold", "int", required=False, default="45", min_value=0, max_value=100),
        FieldRule("moving_average_periods", "int_list", required=False, default="5,10,20,30,60",
                   custom_validator="_check_moving_average_periods"),
    ]),
    SectionRule(name="FILTER_RULES", description="弱势股过滤规则配置", optional=True, fields=[
        FieldRule("enable_weak_stock_filter", "bool", required=False, default="true"),
        FieldRule("exempt_levels", "str", required=False, default="完全主升,趋势加速"),
    ]),
    SectionRule(name="FUND_FLOW", description="资金流分析配置", fields=[
        FieldRule("fund_flow_periods", "str", required=False, default="5,10,20",
                   custom_validator="_check_fund_flow_periods"),
    ]),
    SectionRule(name="TECHNICAL_INDICATORS", description="技术指标信号配置", fields=[
        FieldRule("macd_params", "str", required=False, default="12,26,9"),
    ]),
    SectionRule(name="COLUMN_ALIASES", description="列名别名配置", optional=True, fields=[
        FieldRule("code_aliases", "str", required=False, default="代码=股票代码,证券代码=股票代码,股票代码=股票代码"),
        FieldRule("name_aliases", "str", required=False, default="名称=股票简称,股票名称=股票简称,股票简称=股票简称,简称=股票简称"),
        FieldRule("price_aliases", "str", required=False, default="最新价=最新价,现价=最新价,当前价格=最新价,今收盘=最新价,收盘=最新价,收盘价=最新价"),
    ]),
    SectionRule(name="RESEARCH_REPORT_FILTER", description="研报过滤配置", optional=True, fields=[
        FieldRule("enable_research_report_filter", "bool", required=False, default="false"),
        FieldRule("research_report_min_count", "int", required=False, default="1", min_value=1, max_value=100),
    ]),
    SectionRule(name="USER_FOCUS_STOCKS", description="用户关注股池配置", optional=True, fields=[
        FieldRule("user_focus_stocks", "str", required=False, default=""),
    ]),
    SectionRule(name="API", description="REST API 服务配置", optional=True, fields=[
        FieldRule("enabled", "bool", required=False, default="0"),
        FieldRule("host", "str", required=False, default="127.0.0.1"),
        FieldRule("port", "int", required=False, default="8000", min_value=0, max_value=65535),
        FieldRule("alert_webhook_url", "str", required=False, default=""),
        FieldRule("alert_channel", "str", required=False, default="generic"),
        FieldRule("alert_on_failure", "bool", required=False, default="1"),
        FieldRule("alert_on_success", "bool", required=False, default="0"),
    ]),
    SectionRule(name="ASHAREHUB", description="AShareHub筹码分布数据配置", optional=True, fields=[
        FieldRule("api_key", "str"),
        FieldRule("chip_limit", "int", required=False, default="1", min_value=1, max_value=200),
    ]),
    SectionRule(name="FULL_BULL_SCORING", description="MACD完全多头评分配置", optional=True, fields=[
        FieldRule("weight_zero_axis", "int", required=False, default="20", min_value=0, max_value=100),
        FieldRule("weight_strategy_golden", "int", required=False, default="15", min_value=0, max_value=100),
        FieldRule("weight_tactical_golden", "int", required=False, default="10", min_value=0, max_value=100),
        FieldRule("weight_momentum", "int", required=False, default="15", min_value=0, max_value=100),
        FieldRule("weight_dif_slope", "int", required=False, default="10", min_value=0, max_value=100),
        FieldRule("weight_divergence", "int", required=False, default="10", min_value=0, max_value=100),
        FieldRule("weight_volume_price", "int", required=False, default="10", min_value=0, max_value=100),
        FieldRule("weight_kline_pattern", "int", required=False, default="10", min_value=0, max_value=100),
        FieldRule("conclusion_full_bull", "int", required=False, default="80", min_value=0, max_value=100),
        FieldRule("conclusion_bullish", "int", required=False, default="60", min_value=0, max_value=100),
        FieldRule("conclusion_oscillate", "int", required=False, default="40", min_value=0, max_value=100),
        FieldRule("rule_divergence_threshold", "float", required=False, default="0.3", min_value=0, max_value=1),
        FieldRule("rule_winner_rate_high", "int", required=False, default="80", min_value=0, max_value=100),
        FieldRule("rule_winner_rate_low", "int", required=False, default="15", min_value=0, max_value=100),
        FieldRule("rule_cost_resistance_ratio", "float", required=False, default="0.95", min_value=0, max_value=1),
        FieldRule("rule_chip_concentrated_ratio", "float", required=False, default="0.15", min_value=0, max_value=1),
        FieldRule("rule_price_new_high_days", "int", required=False, default="20", min_value=5, max_value=120),
    ]),
    SectionRule(name="MACRO_FILTER", description="宏观过滤器配置", optional=True, fields=[
        FieldRule("enable_macro_filter", "bool", required=False, default="true"),
    ]),
    SectionRule(name="REGIME_DETECTION", description="市场状态分类参数", optional=True, fields=[
        FieldRule("boll_narrow_ratio", "float", required=False, default="0.8", min_value=0.3, max_value=2.0),
        FieldRule("oscillation_hist_std_ratio", "float", required=False, default="0.1", min_value=0.01, max_value=1.0),
        FieldRule("top_risk_ma20_deviation", "float", required=False, default="0.15", min_value=0.01, max_value=0.5),
        FieldRule("oscillation_min_bars", "int", required=False, default="30", min_value=10, max_value=120),
        FieldRule("reversal_lookback", "int", required=False, default="10", min_value=5, max_value=60),
    ]),
    SectionRule(name="DIVERGENCE", description="背离检测参数", optional=True, fields=[
        FieldRule("base_distance", "int", required=False, default="10", min_value=5, max_value=60),
        FieldRule("strength_threshold", "float", required=False, default="0.15", min_value=0.01, max_value=1.0),
        FieldRule("decay_half_life", "int", required=False, default="8", min_value=2, max_value=60),
        FieldRule("slope_window", "int", required=False, default="5", min_value=3, max_value=30),
    ]),
    SectionRule(name="SCORING_PARAMS", description="评分计算参数", optional=True, fields=[
        FieldRule("atr_stop_mult", "float", required=False, default="1.5", min_value=0.5, max_value=5.0),
        FieldRule("cross_decay_days", "int", required=False, default="30", min_value=5, max_value=120),
        FieldRule("cross_decay_min", "float", required=False, default="0.3", min_value=0.1, max_value=1.0),
        FieldRule("kline_decay_days", "int", required=False, default="10", min_value=2, max_value=60),
        FieldRule("kline_decay_min", "float", required=False, default="0.2", min_value=0.05, max_value=1.0),
        FieldRule("vol_norm_denominator", "float", required=False, default="0.15", min_value=0.01, max_value=1.0),
        FieldRule("trailing_stop_high_ratio", "float", required=False, default="0.98", min_value=0.9, max_value=1.0),
        FieldRule("trailing_stop_lookback", "int", required=False, default="10", min_value=5, max_value=60),
        FieldRule("trailing_stop_high_lookback", "int", required=False, default="20", min_value=10, max_value=120),
        FieldRule("expected_return_lookback", "int", required=False, default="20", min_value=5, max_value=120),
    ]),
    SectionRule(name="TECHNICAL_CONSTANTS", description="标准技术指标参数", optional=True, fields=[
        FieldRule("atr_length", "int", required=False, default="14", min_value=5, max_value=60),
        FieldRule("adx_length", "int", required=False, default="14", min_value=5, max_value=60),
        FieldRule("rsi_length", "int", required=False, default="14", min_value=5, max_value=60),
        FieldRule("boll_length", "int", required=False, default="20", min_value=5, max_value=60),
        FieldRule("boll_std", "float", required=False, default="2.0", min_value=1.0, max_value=4.0),
        FieldRule("stoch_k", "int", required=False, default="9", min_value=3, max_value=30),
        FieldRule("stoch_d", "int", required=False, default="3", min_value=2, max_value=15),
        FieldRule("kline_scan_window", "int", required=False, default="60", min_value=20, max_value=200),
    ]),
    SectionRule(name="POSITION_SIZING", description="仓位管理配置", optional=True, fields=[
        FieldRule("max_single_position", "float", required=False, default="0.33", min_value=0.0, max_value=1.0),
        FieldRule("kelly_fraction", "float", required=False, default="0.25", min_value=0.0, max_value=1.0),
        FieldRule("default_win_rate", "float", required=False, default="0.55", min_value=0.0, max_value=1.0),
        FieldRule("position_a", "float", required=False, default="0.30", min_value=0.0, max_value=1.0),
        FieldRule("position_b", "float", required=False, default="0.15", min_value=0.0, max_value=1.0),
        FieldRule("position_c", "float", required=False, default="0.05", min_value=0.0, max_value=1.0),
        FieldRule("max_industry_exposure", "float", required=False, default="0.30", min_value=0.0, max_value=1.0),
        FieldRule("risk_budget", "float", required=False, default="0.02", min_value=0.001, max_value=0.10),
    ]),
    SectionRule(name="BACKTEST", description="回测系统配置", optional=True, fields=[
        FieldRule("enabled", "bool", required=False, default="false"),
        FieldRule("optimize_frequency", "str", required=False, default="monthly",
                   allowed_values=["initial", "monthly", "quarterly"]),
        FieldRule("backtest_start_date", "str", required=False, default="20230101"),
        FieldRule("out_of_sample_days", "int", required=False, default="120", min_value=20, max_value=504),
        FieldRule("initial_cash", "float", required=False, default="1000000", min_value=10000),
        FieldRule("commission_rate", "float", required=False, default="0.0003", min_value=0, max_value=0.01),
        FieldRule("stamp_tax_rate", "float", required=False, default="0.0005", min_value=0, max_value=0.01),
        FieldRule("slippage", "float", required=False, default="0.001", min_value=0, max_value=0.01),
        FieldRule("transfer_fee_rate", "float", required=False, default="0.00001", min_value=0, max_value=0.001),
        FieldRule("min_commission_per_trade", "float", required=False, default="5.0", min_value=0, max_value=100),
        FieldRule("impact_base", "float", required=False, default="0.002", min_value=0, max_value=0.05),
        FieldRule("impact_threshold", "float", required=False, default="0.01", min_value=0.00001, max_value=1.0),
        FieldRule("impact_cap", "float", required=False, default="0.05", min_value=0, max_value=1.0),
        FieldRule("liquidity_tier_edges", "str", required=False, default="5e6,2e7,1e8"),
        FieldRule("liquidity_tier_impact_base", "str", required=False, default="0.008,0.003,0.0015,0.001"),
        FieldRule("liquidity_tier_threshold", "str", required=False, default="0.005,0.01,0.01,0.02"),
        FieldRule("liquidity_tier_cap", "str", required=False, default="0.10,0.05,0.05,0.03"),
        FieldRule("max_position_pct", "float", required=False, default="0.1", min_value=0.01, max_value=1.0),
        FieldRule("max_order_pct", "float", required=False, default="0.30", min_value=0.01, max_value=1.0),
        FieldRule("max_order_pct_high", "float", required=False, default="0.20", min_value=0.01, max_value=1.0),
        FieldRule("max_order_pct_low", "float", required=False, default="0.10", min_value=0.01, max_value=1.0),
        FieldRule("portfolio_method", "str", required=False, default="score_weighted",
                   allowed_values=["score_weighted", "risk_parity", "min_variance", "mean_variance"]),
        FieldRule("point_in_time", "bool", required=False, default="true"),
        FieldRule("execution_model", "str", required=False, default="next_open",
                   allowed_values=["next_open", "vwap"]),
        FieldRule("simulate_limit_up_down", "bool", required=False, default="true"),
        FieldRule("limit_seal_ratio", "float", required=False, default="0.05", min_value=0, max_value=1),
        FieldRule("limit_tradable_ratio", "float", required=False, default="0.30", min_value=0, max_value=1),
        FieldRule("limit_seal_decay", "float", required=False, default="0.5", min_value=0, max_value=1),
        # P0-6 ⑥：开盘集合竞价成交率分档（封单量/可成交量代理）
        FieldRule("auction_fill_ratio", "float", required=False, default="0.12", min_value=0, max_value=1),
        # 经验填充模型（技术债修复）：历史日线分位数替代固定比例常量
        FieldRule("limit_ratio_mode", "str", required=False, default="fixed",
                  allowed_values=["fixed", "empirical_median", "empirical_p10"]),
        FieldRule("limit_calib_min_samples", "int", required=False, default="20", min_value=1, max_value=1000000),
        FieldRule("limit_stress_enabled", "bool", required=False, default="true"),
        # P0-6 ⑤：市场状态客观变量（指数20日收益 + 波动率分位）
        FieldRule("regime_ret20_full", "float", required=False, default="0.02", min_value=-1, max_value=1),
        FieldRule("regime_ret20_half", "float", required=False, default="-0.02", min_value=-1, max_value=1),
        FieldRule("regime_vol_pct_max", "float", required=False, default="0.8", min_value=0, max_value=1),
        FieldRule("resume_gap_up", "float", required=False, default="0.05", min_value=0, max_value=1),
        FieldRule("resume_gap_down", "float", required=False, default="0.05", min_value=0, max_value=1),
        FieldRule("handling_fee_rate", "float", required=False, default="0.0000341", min_value=0, max_value=0.01),
        FieldRule("csrc_fee_rate", "float", required=False, default="0.00002", min_value=0, max_value=0.01),
        FieldRule("stamp_tax_segments", "str", required=False,
                   default="2023-08-28:0.0005;2000-01-01:0.001"),
    ]),
    SectionRule(name="BACKTEST_CALIBRATED", description="回测校准参数", optional=True, fields=[
        # P0-7 ②：整数参数必须整值落盘（float "17.0" 会在加载端 int() 解析崩溃）
        FieldRule("buy_threshold", "int", required=False, default="17", min_value=1, max_value=100),
        FieldRule("max_holdings", "int", required=False, default="11", min_value=0, max_value=100),
    ]),
    SectionRule(name="MULTI_FACTOR_ALPHA", description="多因子Alpha评分配置", optional=True, fields=[
        FieldRule("enabled", "bool", required=False, default="true"),
        FieldRule("financial_quality_cache_days", "int", required=False, default="90", min_value=1, max_value=365),
        FieldRule("financial_quality_batch_size", "int", required=False, default="100", min_value=1, max_value=5000),
        FieldRule("financial_quality_batch_sleep", "int", required=False, default="10", min_value=0, max_value=600),
        FieldRule("financial_quality_file_cache_days", "int", required=False, default="30", min_value=1, max_value=365),
        FieldRule("fundamentals_retry", "int", required=False, default="3", min_value=1, max_value=10),
    ]),
    SectionRule(name="DISTRIBUTION", description="筹码分布数据配置", optional=True, fields=[
        FieldRule("api_token", "str", required=False, default=""),
    ]),
    SectionRule(name="TRADING_COST", description="交易成本参数", optional=True, fields=[
            FieldRule("commission_rate", "float", required=False, default="0.0003", min_value=0, max_value=0.01),
            FieldRule("stamp_tax_rate", "float", required=False, default="0.0005", min_value=0, max_value=0.01),
            FieldRule("transfer_fee_rate", "float", required=False, default="0.00001", min_value=0, max_value=0.001),
            FieldRule("handling_fee_rate", "float", required=False, default="0.0000341", min_value=0, max_value=0.01),
            FieldRule("csrc_fee_rate", "float", required=False, default="0.00002", min_value=0, max_value=0.01),
            FieldRule("stamp_tax_segments", "str", required=False,
                       default="2023-08-28:0.0005;2000-01-01:0.001"),
    ]),
    SectionRule(name="POSITION_BACKTEST", description="跟仓回测配置", optional=True, fields=[
        FieldRule("pool_file_path", "str", required=False, default="证券交割单.xlsx"),
    ]),
]

_CUSTOM_VALIDATORS = {
    "_check_fund_flow_periods": _check_fund_flow_periods,
    "_check_moving_average_periods": _check_moving_average_periods,
    "_check_full_bull_weights": _check_full_bull_weights,
    "_check_full_bull_thresholds": _check_full_bull_thresholds,
}


# ── 核心校验逻辑 ──────────────────────────────────────────────────────────────

def _type_check(raw: str, rule: FieldRule, issues: list[Issue], section: str) -> bool:
    if rule.type_hint == "int":
        try:
            int(raw)
        except ValueError:
            issues.append(Issue(section=section, field=rule.name, severity=Severity.ERROR,
                                message=f"应为整数，无法解析 '{raw}'", actual_value=raw))
            return False
    elif rule.type_hint == "bool":
        if raw.lower() not in ("true", "false", "1", "0", "yes", "no"):
            issues.append(Issue(section=section, field=rule.name, severity=Severity.WARNING,
                                message="应为 true/false", actual_value=raw))
    return True


def _range_check(raw: str, rule: FieldRule, issues: list[Issue], section: str) -> None:
    if rule.type_hint != "int" or (rule.min_value is None and rule.max_value is None):
        return
    try:
        val = int(raw)
    except ValueError:
        return
    if rule.min_value is not None and val < rule.min_value:
        issues.append(Issue(section=section, field=rule.name, severity=Severity.WARNING,
                            message=f"值 {val} < 最小值 {rule.min_value}", actual_value=str(val)))
    if rule.max_value is not None and val > rule.max_value:
        issues.append(Issue(section=section, field=rule.name, severity=Severity.WARNING,
                            message=f"值 {val} > 最大值 {rule.max_value}", actual_value=str(val)))


def _allowed_check(raw: str, rule: FieldRule, issues: list[Issue], section: str) -> None:
    if rule.allowed_values is None:
        return
    if raw not in rule.allowed_values:
        issues.append(Issue(section=section, field=rule.name, severity=Severity.WARNING,
                            message=f"值 '{raw}' 不在 {rule.allowed_values} 中", actual_value=raw))


def validate(config_path: str = "config.ini") -> ValidationReport:
    """全量校验，返回问题列表。"""
    report = ValidationReport()
    if not os.path.exists(config_path):
        report.issues.append(Issue(section="(文件)", field="", severity=Severity.ERROR,
                                   message=f"配置文件不存在: {os.path.abspath(config_path)}"))
        return report

    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"), interpolation=None)
    parser.read(config_path, encoding="utf-8")
    known_section_names = {sr.name for sr in SECTION_RULES}

    for sec_name in parser.sections():
        if sec_name not in known_section_names:
            report.issues.append(Issue(section=sec_name, field="", severity=Severity.WARNING,
                                       message="未知配置段，系统将忽略"))

    for sr in SECTION_RULES:
        if sr.name not in parser.sections():
            if sr.optional:
                continue
            report.issues.append(Issue(section=sr.name, field="", severity=Severity.ERROR,
                                       message=f"缺少必需配置段 [{sr.name}] ({sr.description})"))
            continue

        sec_data = parser[sr.name]
        if sr.fields is None:
            continue

        for rule in sr.fields:
            if rule.name not in sec_data or not sec_data[rule.name].strip():
                if rule.required:
                    report.issues.append(Issue(section=sr.name, field=rule.name, severity=Severity.ERROR,
                                               message="缺少必需字段", expected=rule.default))
                continue

            raw = sec_data[rule.name].strip()

            if rule.custom_validator:
                validator = _CUSTOM_VALIDATORS.get(rule.custom_validator)
                if validator and rule.type_hint == "int_list":
                    validator(raw, report.issues, sr.name)
                    continue

            if not _type_check(raw, rule, report.issues, sr.name):
                continue

            _range_check(raw, rule, report.issues, sr.name)
            _allowed_check(raw, rule, report.issues, sr.name)

        if sr.name == "FULL_BULL_SCORING":
            _check_full_bull_weights(dict(sec_data), report.issues, sr.name)
            _check_full_bull_thresholds(dict(sec_data), report.issues, sr.name)

    return report


# ── 自动修复逻辑 ──────────────────────────────────────────────────────────────

def _parse_raw_sections(lines: list[str]) -> dict[str, dict[str, Any]]:
    """解析原始行，返回 {section_name: {start, end, field_lines: {key: idx}}}"""
    sections: dict[str, dict[str, Any]] = {}
    cur = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            name = s[1:-1]
            sections.setdefault(name, {"start": i, "end": i, "field_lines": {}})
            cur = name
        elif "=" in s and not s.startswith(("#", ";")) and cur and cur in sections:
            key = s.split("=", 1)[0].strip()
            if key:
                sections[cur]["field_lines"][key] = i
    # 设置 end（下一个 section 前一行或文件末）
    names = list(sections.keys())
    for j, name in enumerate(names):
        if j + 1 < len(names):
            sections[name]["end"] = sections[names[j + 1]]["start"] - 1
        else:
            sections[name]["end"] = len(lines) - 1
    return sections


def _find_insert_line(lines: list[str], section_end: int) -> int:
    """在 section 末尾找到合适的插入行（在最后一个空行前插入）"""
    idx = section_end
    while idx >= 0 and lines[idx].strip() == "":
        idx -= 1
    return idx + 1


def _replace_field_value(line: str, new_value: str) -> str:
    """替换 'key = old_value' 中的值，保留原缩进"""
    if "=" in line:
        key_part = line.split("=", 1)[0]
        return f"{key_part}= {new_value}\n"
    return line


def auto_repair(config_path: str = "config.ini") -> int:
    """
    自动修复 config.ini：
      - 整段缺失的必需 Section → 追加
      - 缺失/不可解析的字段 → 写入默认值
      - 修复前备份为 config.ini.bak
    返回修复的操作数。
    """
    report = validate(config_path)
    if not report.has_errors:
        return 0

    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections = _parse_raw_sections(lines)
    ops: list[str] = []

    # 收集需修复的 ERROR
    missing_sections: set[str] = set()
    bad_fields: set[tuple[str, str]] = set()
    for iss in report.issues:
        if iss.severity != Severity.ERROR:
            continue
        if iss.field:
            bad_fields.add((iss.section, iss.field))
        else:
            missing_sections.add(iss.section)

    # 1. 补充缺失 Section（确保文件以换行结尾）
    need_newline = lines and not lines[-1].endswith("\n")

    for sr in SECTION_RULES:
        if sr.name not in missing_sections:
            continue
        tmpl = DEFAULT_SECTION_TEMPLATES.get(sr.name)
        if not tmpl:
            continue
        if need_newline:
            lines.append("\n")
            need_newline = False
        lines.append("\n")
        for tline in tmpl:
            lines.append(tline + "\n")
        ops.append(f"[{sr.name}] 整段追加")

    # 2. 修复缺失/错误的字段
    for sec_name, field_name in sorted(bad_fields):
        if sec_name not in sections:
            continue
        default_val = FIELD_DEFAULTS.get(sec_name, {}).get(field_name)
        if default_val is None:
            continue
        sec = sections[sec_name]
        field_lines = sec["field_lines"]

        if field_name in field_lines:
            # 替换已有行
            idx = field_lines[field_name]
            lines[idx] = _replace_field_value(lines[idx], default_val)
            ops.append(f"[{sec_name}] {field_name} → {default_val}")
        else:
            # 行末插入
            insert_at = _find_insert_line(lines, sec["end"])
            # 确保前有空行
            need_blank = (insert_at > 0 and lines[insert_at - 1].strip() != "")
            indent = ""
            if need_blank:
                lines.insert(insert_at, "\n")
                insert_at += 1
            lines.insert(insert_at, f"{indent}{field_name} = {default_val}\n")
            ops.append(f"[{sec_name}] {field_name} = {default_val} (新增)")

    if not ops:
        return 0

    # 3. 备份
    bak_path = config_path + ".bak"
    try:
        shutil.copy2(config_path, bak_path)
    except (OSError, shutil.Error):
        logger.info(f"备份失败: {bak_path}")
        return 0

    # 4. 写回
    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info(f"已备份原文件至 {bak_path}")
    for op in ops:
        print(f"  + {op}")
    logger.info(f"共修复 {len(ops)} 项\n")
    return len(ops)


# ── 一站式入口 ────────────────────────────────────────────────────────────────

def validate_and_repair(config_path: str = "config.ini") -> ValidationReport:
    """校验 → 打印 → 自动修复 → 返回报告。"""
    report = validate(config_path)
    if report.has_errors:
        report.print_summary()
        auto_repair(config_path)
    else:
        report.print_summary()
    return report
