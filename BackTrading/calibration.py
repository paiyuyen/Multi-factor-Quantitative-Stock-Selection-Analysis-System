from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    p = Path(__file__).resolve().parent  # Backtesting/
    for _ in range(10):
        if (p / "config.ini").exists():
            return p
        parent = p.parent
        if parent == p:
            break
        p = parent
    return Path.cwd()


PROJECT_ROOT = _project_root()

import pandas as pd
from loguru import logger

from BackTrading.bayesian.space import build_spaces, split_by_cost, describe

# config.ini 中参数名 → (section, key) 映射
# ── 寻优参数（bayesian optimizer 搜索，写回闭环）──
# ── 受控静态参数（POSITION_SIZING；回测引擎等权不消费，但复盘 PositionSizer 使用）──
# 受控参数写在 [BACKTEST_CALIBRATED] 同一分区，由 ConfigParser.apply_backtest_calibrated_override() 读取；
# 此处不加入 bayesian 搜索空间（见 bayesian/space.py DEAD_KEYS），但保留写回路径，
# 以便 apply_calibration_to_config 和 write_calibration_to_ini 能处理手动写入的校准值。
CALIB_PARAM_MAP: dict[str, tuple[str, str]] = {
    # 寻优参数
    "atr_stop_mult": ("BACKTEST_CALIBRATED", "atr_stop_mult"),
    "boll_narrow_ratio": ("BACKTEST_CALIBRATED", "boll_narrow_ratio"),
    "cross_decay_days": ("BACKTEST_CALIBRATED", "cross_decay_days"),
    "conclusion_full_bull": ("BACKTEST_CALIBRATED", "conclusion_full_bull"),
    "golden_cross_bonus": ("BACKTEST_CALIBRATED", "golden_cross_bonus"),
    "divergence_penalty": ("BACKTEST_CALIBRATED", "divergence_penalty"),
    "buy_threshold": ("BACKTEST_CALIBRATED", "buy_threshold"),
    "max_holdings": ("BACKTEST_CALIBRATED", "max_holdings"),
    # ── P4 组合优化器超参数 ──
    "optimizer_risk_aversion": ("BACKTEST_CALIBRATED", "optimizer_risk_aversion"),
    "optimizer_turnover_penalty": ("BACKTEST_CALIBRATED", "optimizer_turnover_penalty"),
    "optimizer_max_weight": ("BACKTEST_CALIBRATED", "optimizer_max_weight"),
    "optimizer_cov_lookback": ("BACKTEST_CALIBRATED", "optimizer_cov_lookback"),
    # ── 受控静态参数（Position Sizer 消费；回测引擎等权不消费）──
    # 注：position_d 不在此处 —— PositionSizingConfig 无 POSITION_D 字段（D 级恒为 0 仓位）
    "kelly_fraction": ("BACKTEST_CALIBRATED", "kelly_fraction"),
    "position_a": ("BACKTEST_CALIBRATED", "position_a"),
    "position_b": ("BACKTEST_CALIBRATED", "position_b"),
    "position_c": ("BACKTEST_CALIBRATED", "position_c"),
    "risk_none_multiplier": ("BACKTEST_CALIBRATED", "risk_none_multiplier"),
    # ── 波动率自适应退出参数（VAEO 学习产出；复盘/跟盘单元消费）──
    "learned_t1_mult": ("BACKTEST_CALIBRATED", "learned_t1_mult"),
    "learned_t2_mult": ("BACKTEST_CALIBRATED", "learned_t2_mult"),
}


@dataclass
class CalibrationResult:
    params: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_vol: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    timestamp: str = ""
    git_commit: str = ""
    config_hash: str = ""
    pbo: float = 0.0
    dsr: float = 0.0
    num_trials: int = 0
    # ── 成本模型快照（方案C：回测验证假设的持久化记录）──
    cost_model_snapshot: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


CALIBRATION_FILE = PROJECT_ROOT / "calibration_result.json"
CONFIG_INI = PROJECT_ROOT / "config.ini"

# 写入 config.ini 时需取整的整数参数
# 一旦加载端 int() 解析即崩溃；现强制整值落盘（写入前另有类型断言）。
_INT_KEYS = frozenset({
    "cross_decay_days", "conclusion_full_bull",
    "golden_cross_bonus", "divergence_penalty",
    "buy_threshold", "max_holdings",
})


def run_bayesian_walk_forward(
    kline_df: pd.DataFrame,
    train_period: int = 120,
    test_period: int = 60,
    num_paths: int = 3,
    initial_cash: float = 1_000_000.0,
    spaces: dict | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """贝叶斯 Walk-Forward 优化入口。

    Args:
        kline_df: K 线数据
        train_period: IS 训练窗口（交易日）
        test_period: OOS 验证窗口（审计强制 ≥ 60 天，低于此值拒绝执行）
        num_paths: 多路径数（≥ 5，路径间偏移 ≥ 40 天以降低相关性）
        initial_cash: 初始资金
        spaces: 预构建的 ParamSpace dict（None 时从 config 自动构建）
        **kwargs: 透传给引擎的额外参数

    Returns:
        DataFrame, 每行一个 WFO 窗口，与旧 walk_forward 返回格式兼容。
    """
    # P0 审计修复：OOS 窗口硬约束 ≥ 60 天
    if test_period < 60:
        raise ValueError(
            f"OOS 验证窗口 {test_period} 天 < 60 天最小要求，统计效力不足（Sharpe 标准误 ≈ 1.96/√{test_period-1}）"
            " — 请缩短 IS 窗口或增加数据跨度以提供至少 60 天 OOS。"
        )
    # P1 审计修复：路径数 ≥ 5 以降低路径间相关性
    if num_paths < 5:
        logger.warning(
            f"路径数 {num_paths} < 5，WFO 中位数聚合统计效力不足，建议 ≥ 5 且路径起始偏移 ≥ 40 天"
        )

    from UtilsManager.ConfigParser import Config as _Config

    if spaces is None:
        full_cfg = _Config()
        bt_cfg = full_cfg.app_config.backtest
        po_cfg = getattr(full_cfg.app_config, "portfolio_optimizer", None)
        spaces = build_spaces(bt_cfg, portfolio_optimizer_config=po_cfg)

    signal_sp, portfolio_sp = split_by_cost(spaces)
    logger.info(f"贝叶斯 WFO 参数空间:\n{describe(spaces)}")
    logger.info(f"  信号参数: {len(signal_sp)} | 组合参数: {len(portfolio_sp)}")

    n_dates = len(kline_df["trade_date"].unique())
    logger.info(f"  交易日数: {n_dates} | IS={train_period} | OOS={test_period}")

    # ── 正式调用贝叶斯 WFO 引擎 ─────────────────────────
    from BackTrading.bayesian.meta_optimizer import bayesian_walk_forward_multi

    result = bayesian_walk_forward_multi(
        kline_df=kline_df,
        train_period=train_period,
        test_period=test_period,
        num_paths=num_paths,
        initial_cash=initial_cash,
        spaces=spaces,
        **kwargs,
    )
    return result



def save_calibration(result: CalibrationResult) -> None:
    tmp_path = CALIBRATION_FILE.with_name(CALIBRATION_FILE.name + ".tmp")
    tmp_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, CALIBRATION_FILE)


def load_calibration() -> CalibrationResult | None:
    if not CALIBRATION_FILE.exists():
        return None
    try:
        data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
        return CalibrationResult.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def audit_cost_model_vs_calibration() -> list[str]:
    """运行时审计：比对当前成本假设与回测校准快照。

    在复盘/实盘启动时调用，加载 calibration_result.json 中持久化的
    回测验证假设，与当前 CostModel 默认值（或 config.ini 配置）比对，
    若发现实盘成本假设低于回测值则返回告警列表。

    Returns:
        告警信息字符串列表。空列表表示无差异或无校准数据可供比对。
    """
    warnings_list: list[str] = []

    cal = load_calibration()
    if cal is None:
        warnings_list.append("[成本审计] calibration_result.json 不存在，跳过比对")
        return warnings_list

    snapshot = cal.cost_model_snapshot
    if not snapshot:
        warnings_list.append("[成本审计] calibration_result.json 中无 cost_model_snapshot，跳过比对")
        return warnings_list

    # 获取当前 CostModel 默认值用于比对
    try:
        from BackTrading.domain.models import CostModel
    except ImportError:
        warnings_list.append("[成本审计] 无法导入 CostModel，跳过比对")
        return warnings_list

    current = CostModel()

    # 关键审计字段：实盘值不得低于回测值（否则收益会被高估）
    # 即 commission/slippage/impact 等 "成本类" 参数，实盘 >= 回测
    critical_fields = [
        "commission_rate", "stamp_tax_rate", "market_slippage",
        "limit_slippage", "impact_threshold", "impact_base",
        "min_commission_per_trade", "transfer_fee_rate",
        "handling_fee_rate", "csrc_fee_rate",
    ]

    for field in critical_fields:
        snap_val = snapshot.get(field)
        curr_val = getattr(current, field, None)
        if snap_val is None or curr_val is None:
            continue
        # 数值化比对（容忍浮点精度差异）
        try:
            s = float(snap_val)
            c = float(curr_val)
        except (TypeError, ValueError):
            continue

        tolerance = 1e-10  # 绝对容差
        if c < s - tolerance:
            warnings_list.append(
                f"[成本审计告警] {field}: 当前值({c}) < 回测验证值({s}) — "
                f"实盘摩擦成本假设过低，策略收益可能不及预期"
            )

    # 布尔字段检查（commission_includes_fees 变更会导致总佣金跳变）
    bool_fields = ["commission_includes_fees"]
    for field in bool_fields:
        snap_val = snapshot.get(field)
        curr_val = getattr(current, field, None)
        if snap_val is not None and curr_val is not None and snap_val != curr_val:
            warnings_list.append(
                f"[成本审计告警] {field}: 当前值({curr_val}) != 回测验证值({snap_val})"
            )

    if not warnings_list:
        logger.info("[成本审计] 当前成本模型与回测校准快照一致，通过检查")

    return warnings_list



def apply_calibration_to_config(config: object) -> None:
    from UtilsManager.ConfigParser import Config

    assert isinstance(config, Config), f"需要 Config 实例，收到 {type(config).__name__}"
    cfg = config
    result = load_calibration()
    if result is None:
        return
    overrides = result.params.copy()
    if not overrides:
        return

    rd = cfg.app_config.regime_detection
    sc = cfg.app_config.scoring_params
    fr = cfg.app_config.filter_rules
    ps = cfg.app_config.position_sizing

    for key, val in overrides.items():
        attr = key.upper()
        if key == "boll_narrow_ratio":
            rd.BOLL_NARROW_RATIO = val
        elif key == "cross_decay_days":
            sc.CROSS_DECAY_DAYS = int(val)
        elif key == "atr_stop_mult":
            setattr(sc, attr, val)
        elif key == "conclusion_full_bull":
            cfg.app_config.full_bull_scoring.CONCLUSION_FULL_BULL = int(val)
        elif key == "golden_cross_bonus":
            sc.GOLDEN_CROSS_BONUS = int(val)
        elif key == "divergence_penalty":
            sc.DIVERGENCE_PENALTY = int(val)
        # P0-7 ②：校准闭环补齐 —— buy_threshold/max_holdings 曾只写不读，
        # 现覆写到 backtest 配置（与 ConfigParser [BACKTEST_CALIBRATED] 覆写同目标）
        elif key == "buy_threshold":
            cfg.app_config.backtest.BUY_THRESHOLD = int(val)
        elif key == "max_holdings":
            cfg.app_config.backtest.MAX_HOLDINGS = int(val)
        # 受控静态参数（Position Sizer 消费；回测引擎等权不消费）
        # 若校准结果中出现了这些键（如手动写入 calibration_result.json），
        # 则写回至 PositionSizingConfig，确保复盘与配置一致。
        elif key == "kelly_fraction":
            ps.KELLY_FRACTION = float(val)
        elif key == "position_a":
            ps.POSITION_A = float(val)
        elif key == "position_b":
            ps.POSITION_B = float(val)
        elif key == "position_c":
            ps.POSITION_C = float(val)
        elif key == "risk_none_multiplier":
            ps.RISK_NONE_MULTIPLIER = float(val)


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _format_val(key: str, val: Any) -> str:
    """按字段语义格式化配置值：整数参数取整，浮点去尾零，避免 int('37.0') 崩溃。"""
    if key in _INT_KEYS:
        return str(int(round(float(val))))
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    s = f"{v:.10f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def write_calibration_to_ini(params: dict) -> dict | None:
    """将校准参数写入 config.ini 的 [BACKTEST_CALIBRATED]。

    已有键原位替换（保留行尾注释），新键追加到 section 末尾；
    整型参数取整写入，避免下次加载时 int() 解析崩溃；原子写防半截文件。
    返回实际落盘参数（整数参数已取整），未写入时返回 None。
    """
    if not params:
        logger.info("无校准参数，跳过写入")
        return None
    # P0-1 修复：整数参数统一取整 —— WFO 回退参数 = 参数空间中点，会产生
    # 31.5/17.5/11.5 等非整值；旧实现在此 fail-fast 抛 ValueError 后被调用方
    # except 吞掉，导致 config.ini 实际未写但日志谎报"已写入"。现在仅对
    # 无法转换为数值的值 fail-fast，非整数值 round 后落盘。
    sanitized = dict(params)
    for k in sorted(_INT_KEYS & set(sanitized)):
        v = sanitized[k]
        try:
            sanitized[k] = int(round(float(v)))
        except (TypeError, ValueError):
            raise ValueError(
                f"[校准写回] 整数参数 {k} = {v!r} 无法转换为数值，拒绝写入 config.ini（防 int() 解析崩溃）"
            ) from None
    params = sanitized
    ini_path = CONFIG_INI
    if not ini_path.exists():
        logger.warning(f"config.ini 不存在: {ini_path}")
        return None

    text = ini_path.read_text(encoding="utf-8")
    section_header = "[BACKTEST_CALIBRATED]"
    lines = text.splitlines(keepends=True)

    # 定位 section（或在其后插入）
    sec_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("["):
            if s == section_header:
                sec_idx = i
                break
            if sec_idx is not None:
                break
    if sec_idx is None:
        if text and not text.endswith("\n"):
            text += "\n"
        text += f"\n{section_header}\n"
        lines = text.splitlines(keepends=True)
        sec_idx = len(lines) - 1

    written: set[str] = set()
    out: list[str] = []
    in_section = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("["):
            in_section = s == section_header
            out.append(ln)
            continue
        if in_section and "=" in s and not s.startswith(("#", ";")):
            key = s.split("=", 1)[0].strip().lower()
            if key in params:
                comment = ""
                cm = re.search(r"#.*$", ln)
                if cm:
                    comment = cm.group(0)
                body = f"{key} = {_format_val(key, params[key])}"
                ln = body + ("  " + comment if comment else "") + ("\n" if ln.endswith("\n") else "")
                written.add(key)
        out.append(ln)

    # 追加未写出的键（插到 section 内容末尾）
    missing = {k for k in params if k not in written}
    if missing:
        insert_at = len(out)
        for i in range(len(out) - 1, -1, -1):
            if out[i].strip().startswith("["):
                insert_at = i + 1
                break
        extra = "".join(f"{k} = {_format_val(k, params[k])}\n" for k in sorted(missing))
        out.insert(insert_at, extra)

    tmp_path = ini_path.with_name(ini_path.name + ".tmp")
    tmp_path.write_text("".join(out), encoding="utf-8")
    os.replace(tmp_path, ini_path)
    logger.info(f"校准参数已写入 {ini_path} [{section_header}]")
    return params
