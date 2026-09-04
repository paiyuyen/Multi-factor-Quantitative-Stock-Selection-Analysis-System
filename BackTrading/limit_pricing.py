"""涨跌停价计算 + 可成交量规则 — 撮合层约束建模（Limit Matching）。

目标：在撮合层引入涨跌停约束与可成交量规则，提升成交模拟真实度。

业务范围：仅沪深主板（60x/00x），创业板/科创板/北交所已从系统中剔除。

涨跌停规则（基于前收盘与当期涨跌幅规则）：
    - 主板（60x/00x）：±10%；ST/*ST 一律 ±5%
    - 退市整理期（P0-6）：整理期首日无涨跌幅限制，次日起 ±10%
      （进入退市整理期后涨跌幅独立于 ST 5% 规则，按 10% 计算）
    - 上市初期豁免：
        主板注册制（2023-04-10 起）：上市前 5 个交易日无涨跌幅限制
        主板核准制（2023-04-10 前）：上市首日 ±44%/-36%，次日起 ±10%
    - 限价取整：前收 × (1±比例) 四舍五入到分

可成交量规则（simulate_limit_up_down=true 时撮合层消费）：
    - 非涨跌停日：可成交量比例 1.0（不限）
    - 涨停/跌停日：可成交量比例 = 一字板(开盘=收盘=限价) 用 seal_ratio，
      开盘触板后炸板(open≥限价, close<限价) 用 tradable_ratio，
      盘中冲板(open<限价, high≥限价) 用 intraday_ratio；
      连续板数 ≥ 2 时按 seal_decay 逐板衰减
    - 可成交量 = 当日成交量 × 比例，向下取整到最小交易单位；
      小于一手 → 未成交（日志 [撮合约束] 可追溯）

配置（[BACKTEST]，simulate_limit_up_down=true 开启本模型，false 回退简化撮合）：
    simulate_limit_up_down = true
    limit_seal_ratio = 0.05       # 一字板可成交量比例
    limit_tradable_ratio = 0.30   # 开盘触板后炸板可成交量比例
    limit_intraday_ratio = 0.10   # 盘中冲板可成交量比例
    limit_seal_decay = 0.5        # 连续板每板衰减系数
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import numpy as np
from loguru import logger

# 涨跌停价取整精度（分）
LIMIT_PX_PRECISION = 100.0
# 主板注册制首批上市日（此后主板新股前 5 日无涨跌幅限制）
MAIN_BOARD_REFORM_DATE = "2023-04-10"
# ST/*ST 涨跌幅限制
ST_LIMIT_RATIO = 0.05
# 主板常规涨跌幅限制
MAIN_BOARD_LIMIT_RATIO = 0.10
# 退市整理期涨跌幅（P0-6）：首日无涨跌幅限制，次日起 ±10%
# （进入退市整理期后独立于 ST 5% 规则；引擎按 is_delisting 状态机判定首日/期间）
DELISTING_PERIOD_LIMIT_RATIO = 0.10
# 核准制主板上市首日限制（44% / -36%）
MAIN_BOARD_FIRST_DAY_UP = 0.44
MAIN_BOARD_FIRST_DAY_DOWN = 0.36
# 主板注册制后上市前 5 个交易日无涨跌幅限制
MAIN_BOARD_REGISTERED_EXEMPT_DAYS = 5
# 数值比较容差（浮点取整后相等判断）
_LIMIT_EPS = 1e-9

_CODE_RE = re.compile(r"(?:[a-zA-Z]*)(\d+)")


def listing_exempt_days(trade_date: Optional[str] = None) -> int:
    """主板上市初期无涨跌幅限制的交易日数。

    Args:
        trade_date: "YYYY-MM-DD"；主板注册制改革(2023-04-10)前后规则不同。

    Returns:
        豁免交易日数（listing_days <= 该值即豁免）。
        注册制后前 5 日豁免；核准制无豁免（首日用 44%/36%）。
    """
    if trade_date is not None and str(trade_date)[:10] >= MAIN_BOARD_REFORM_DATE:
        return MAIN_BOARD_REGISTERED_EXEMPT_DAYS
    return 0


def _round_half_up_scalar(x: float) -> float:
    """交易所"四舍五入到分"（ROUND_HALF_UP）。

    Python round / np.round 采用银行家舍入（round-half-even）：第三位小数恰为 5
    时舍向偶数。交易所（沪深交易规则）为 ROUND_HALF_UP——第三位恰为 5 一律进位。
    例：前收 9.55 → 涨停 9.55×1.1 = 10.505 → 交易所 10.51，round(10.505,2)=10.5。
    用 Decimal(str(x)) 规避二进制浮点表示误差（10.505 实际 ≈10.5049999...）。
    """
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round_half_up_vec(x: np.ndarray) -> np.ndarray:
    """向量化 half-up（等效 _round_half_up_scalar）。

    前收×比例是 0.01 精度的有限十进制，第三位小数为精确 0-9；仅二进制表示有
    尾差。np.nextafter 将二进制略低于整数的边界值推过 0.5 进位阈值：第三位=5
    （浮点 10.505→1050.4999...）→ 进位；第三位=0（恰为整数）→ 不进位。
    数值结果为 {0, 0.005, 0.01, ...} 的半格边界判定，无 ROUND_HALF_UP 误判。
    """
    y = np.nextafter(x * 100.0, np.inf)
    return np.floor(y + 0.5) / 100.0


def exright_reference_price(
    prev_close_raw: float,
    adj_factor_today: float,
    adj_factor_yesterday: float,
) -> float:
    """计算交易所"除权除息参考价"用作涨跌停基准。

    涨跌停价以交易所除权除息参考价为基准，而非原始前收盘。
    除权日若直接用原始前收计算涨跌停价，偏差 = 每股分红 / 前收，
    一字板判定随之失真（close ≥ limit_up 误判 / 漏判）。

    原理：adj_factor 为不复权→后复权的累积乘数，
    adj_factor_today / adj_factor_yesterday = 1（普通日）
    adj_factor_today / adj_factor_yesterday < 1（除权日：比值 = 除权系数）

    ref_price = prev_close_raw × (adj_factor_today / adj_factor_yesterday)

    等价于交易所公式：(前收 − 每股现金红利) / (1 + 送转比例)

    Args:
        prev_close_raw: 前收盘价（不复权原始价）
        adj_factor_today: 当日 adj_factor（后复权因子）
        adj_factor_yesterday: 前日 adj_factor

    Returns:
        涨跌停基准参考价；无法计算时退化为 prev_close_raw。
    """
    if (np.isfinite(adj_factor_today) and np.isfinite(adj_factor_yesterday)
            and adj_factor_yesterday > 0 and adj_factor_today > 0):
        ratio = adj_factor_today / adj_factor_yesterday
        # 除权系数应在合理范围内（防止脏数据导致极端偏差）
        if 0.1 < ratio < 5.0:
            return prev_close_raw * ratio
    return prev_close_raw


def calc_limit_prices(
    prev_close: float,
    ratio_up: float,
    ratio_down: float,
    precision: float = LIMIT_PX_PRECISION,
) -> tuple[float, float]:
    """涨跌停价 = 前收 × (1±比例)，四舍五入到分。

    P0-11 审计修复：以交易所 ROUND_HALF_UP（四舍五入）替代 Python round 的
    银行家舍入（round-half-even）——第三位小数恰为 5 时 round 会舍向偶数，
    与交易所规则系统性偏差 0.01 元，影响封板/一字板/可成交量判定。

    P2.1 修复：中间计算使用 Decimal 精确十进制算术，避免高价位股票（如茅台 1800+）
    浮点乘法误差导致 10.505 → 10.50499999... → 低位舍入偏差。
    """
    # Decimal 精确计算替代 float 乘法
    pc_d = Decimal(str(prev_close))
    up = float((pc_d * (1 + Decimal(str(ratio_up)))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))
    down = float((pc_d * (1 - Decimal(str(ratio_down)))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))
    return up, down


def calc_limit_prices_batch(
    prev_close: np.ndarray,
    ratio_up: np.ndarray,
    ratio_down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """P2 审计修复：向量化涨跌停价批量计算，替代逐股循环调用 calc_limit_prices。

    P0-11：取整采用 ROUND_HALF_UP（_round_half_up_vec），与标量版一致，
    不再使用 np.round（银行家舍入）。

    Args:
        prev_close: 前收盘价数组（n,）。除权日应为"除权除息参考价"。
        ratio_up: 涨停比例数组（n,）
        ratio_down: 跌停比例数组（n,）

    Returns:
        (limit_up, limit_down) 各 (n,)
    """
    up = _round_half_up_vec(prev_close * (1.0 + ratio_up))
    down = _round_half_up_vec(prev_close * (1.0 - ratio_down))
    return up, down


def lot_size_for(symbol: str) -> int:
    """A 股沪深主板每手申报单位 = 100 股（一处定义，引擎买入/卖出/可成交量撮合全链路复用）。

    注：系统仅覆盖沪深主板（60x/00x），100 股/手。科创板 200 股/手已随板块剔除。
    """
    return 100


@dataclass(frozen=True)
class LimitPriceInfo:
    """单只股票当日涨跌停信息（供撮合层消费）。"""

    ratio_up: float
    ratio_down: float
    limit_up: float
    limit_down: float
    exempt: bool = False
    is_st: bool = False

    def at_limit_up(self, close: float) -> bool:
        return close >= self.limit_up - _LIMIT_EPS

    def at_limit_down(self, close: float) -> bool:
        return close <= self.limit_down + _LIMIT_EPS


def limit_prices_for(
    prev_close: float,
    symbol: str,
    *,
    is_st: bool = False,
    is_delisting: bool = False,
    listing_days: Optional[int] = None,
    trade_date: Optional[str] = None,
) -> LimitPriceInfo:
    """计算单只主板股票当日涨跌停价。

    Args:
        prev_close: 前收盘（不复权）。
        symbol: 股票代码（sh600000 / 600000.SH / 600000）— 保留参数兼容签名，
                 系统仅覆盖沪深主板，不再按代码分派板块规则。
        is_st: 当日是否为 ST/*ST（涨跌幅 5%）。
        is_delisting: 当日是否处于退市整理期（涨跌幅 10%，独立于 ST 规则）。
        listing_days: 上市第 N 个交易日（1=首日）；None 视为已上市多日（无豁免）。
        trade_date: "YYYY-MM-DD"，主板注册制前后规则切换用。

    Returns:
        LimitPriceInfo：exempt=True 时 limit_up/down 用 ±100% 近似无限制。
    """
    # P1 审计修复 + P1.10 精确前缀匹配：非主板代码进入时直接拦截（fail-fast）
    # 使用精确前缀集合替代 startswith，避免 8/4 单字符前缀误匹配其他代码段。
    _MAIN_BOARD_PREFIXES = {"600", "601", "603", "605", "000", "001", "002", "003"}
    _GEM_PREFIXES = {"300", "301"}
    _STAR_PREFIXES = {"688"}
    _BSE_PREFIXES = {"83", "87", "43", "89"}

    _digits = _CODE_RE.search(symbol)
    if _digits is not None:
        _code = _digits.group(1)
        _prefix = _code[:3]

        # P1.10：精确前缀集合拦截，不依赖单字符 startswith
        if _prefix in _GEM_PREFIXES:
            raise ValueError(
                f"[涨跌停] {symbol} 为创业板（{_prefix}），不适用主板涨跌停规则。"
                f"请确认 stock_basic_info 过滤正确或关闭 main_board_only。"
            )
        elif _prefix in _STAR_PREFIXES:
            raise ValueError(
                f"[涨跌停] {symbol} 为科创板（{_prefix}），不适用主板涨跌停规则。"
                f"请确认 stock_basic_info 过滤正确或关闭 main_board_only。"
            )
        elif _code[:2] in _BSE_PREFIXES:
            raise ValueError(
                f"[涨跌停] {symbol} 为北交所（{_code[:2]}），不适用主板涨跌停规则。"
                f"请确认 stock_basic_info 过滤正确或关闭 main_board_only。"
            )

    exempt = False
    if listing_days is not None:
        exempt = listing_days <= listing_exempt_days(trade_date)

    if exempt:
        # P1.9 修复：新股豁免期返回 inf 涨跌停价，确保判定逻辑自动放行。
        # 原实现 ratio=1.0 导致所有价位均触发涨跌停守卫，误杀正常买入/卖出。
        return LimitPriceInfo(
            ratio_up=float("inf"),
            ratio_down=float("inf"),
            limit_up=float("inf"),
            limit_down=float("-inf"),
            exempt=True,
            is_st=False,
        )
    elif is_delisting:
        # P0.5 修复：退市整理期涨跌幅 ±10%，独立于 ST ±5% 规则
        ratio_up = ratio_down = MAIN_BOARD_LIMIT_RATIO
    elif (
        listing_days == 1
        and (trade_date is None or str(trade_date)[:10] < MAIN_BOARD_REFORM_DATE)
    ):
        # 核准制主板首日：44% / -36%
        ratio_up, ratio_down = MAIN_BOARD_FIRST_DAY_UP, MAIN_BOARD_FIRST_DAY_DOWN
    else:
        ratio = ST_LIMIT_RATIO if is_st else MAIN_BOARD_LIMIT_RATIO
        ratio_up = ratio_down = ratio

    up, down = calc_limit_prices(prev_close, ratio_up, ratio_down)
    return LimitPriceInfo(
        ratio_up=ratio_up,
        ratio_down=ratio_down,
        limit_up=up,
        limit_down=down,
        exempt=exempt,
        is_st=is_st,
    )


def fill_ratio_for(
    close: float,
    open_price: Optional[float],
    high: Optional[float],
    low: Optional[float],
    limit_up: float,
    limit_down: float,
    board_streak: int = 1,
    *,
    seal_ratio: float = 0.05,
    tradable_up_ratio: float = 0.30,
    tradable_down_ratio: float = 0.30,
    tradable_ratio: float = 0.30,  # [deprecated] 保留兼容，默认与 up 一致
    intraday_ratio: float = 0.10,
    seal_decay: float = 0.5,
) -> float:
    """可成交量比例（撮合层消费）——三档触板口径。

    注意：此函数依赖当日 close/high/low（未来信息），P2 审计已将其从引擎
    核心路径移除。保留仅供校准参考 / 离线分析使用。

    触板判定（日线近似，无需分钟数据）：
        - 涨停方向：high >= limit_up 或 open >= limit_up（一字/高开触及涨停价）
        - 跌停方向：low <= limit_down 或 open <= limit_down
    三档口径（按触板方式区分，全天量 × 比例 = 可成交量）：
        1. 一字板（开=收=限价）：seal_ratio（默认 5%）—— 全天封死，流动性最差
        2. 开盘触板后炸板（open≥限价 但 close<限价）：
           涨停方向用 tradable_up_ratio（默认 30%），跌停方向用 tradable_down_ratio
           —— 开盘即封板、随后打开，早盘限价附近有大量挂单
        3. 盘中冲板（open<限价, high≥限价）：intraday_ratio（默认 10%）
           —— 盘中才触板，全天大部分成交量不在限价附近
    - 非触板日：1.0（不限制）
    - 连续板数 n 板 → 比例 × seal_decay^(n-1)
    """
    touched_up = limit_up is not None and (
        (high is not None and high >= limit_up - _LIMIT_EPS)
        or (open_price is not None and open_price >= limit_up - _LIMIT_EPS)
    )
    if touched_up:
        sealed = (
            open_price is not None and open_price >= limit_up - _LIMIT_EPS
            and close is not None and close >= limit_up - _LIMIT_EPS
        )
        if sealed:
            r = seal_ratio
        elif open_price is not None and open_price >= limit_up - _LIMIT_EPS:
            # 开盘涨停后炸板：open≥limit 但 close<limit
            r = tradable_up_ratio
        else:
            # 盘中冲板：open<limit, high≥limit
            r = intraday_ratio
        return r * (seal_decay ** max(0, board_streak - 1))

    touched_down = limit_down is not None and (
        (low is not None and low <= limit_down + _LIMIT_EPS)
        or (open_price is not None and open_price <= limit_down + _LIMIT_EPS)
    )
    if touched_down:
        sealed = (
            open_price is not None and open_price <= limit_down + _LIMIT_EPS
            and close is not None and close <= limit_down + _LIMIT_EPS
        )
        if sealed:
            r = seal_ratio
        elif open_price is not None and open_price <= limit_down + _LIMIT_EPS:
            # 开盘跌停后炸板：open≤limit 但 close>limit
            r = tradable_down_ratio
        else:
            # 盘中触跌停：open>limit, low≤limit
            r = intraday_ratio
        return r * (seal_decay ** max(0, board_streak - 1))

    return 1.0


def auction_fill_ratio_for(
    open_price: Optional[float],
    limit_up: float,
    limit_down: float,
    *,
    side: str = "sell",
    tradable_up_ratio: float = 0.30,
    tradable_down_ratio: float = 0.30,
    board_streak: int = 1,
    seal_decay: float = 0.5,
) -> float:
    """开盘集合竞价时点（9:25）的可成交量比例 — P1-2 前视修复 + P0.4 方向区分。

    P0.4 修复：涨停开盘买卖方向成交率区分——
        - 涨停开盘时买方排队深，买入成交极难（~5%）
        - 涨停开盘时卖方提供流动性，卖出相对容易（~60%）
        - 跌停开盘反之，卖出极难（~5%），买方相对容易接盘（~60%）
    side="buy" 时买入方向取保守比，"sell" 时取宽松比。
    """
    touched_up = open_price is not None and open_price >= limit_up - _LIMIT_EPS
    touched_down = open_price is not None and open_price <= limit_down + _LIMIT_EPS

    decay = seal_decay ** max(0, board_streak - 1)

    if touched_up:
        # P0.4：按方向拆分成交率
        if side == "buy":
            # 涨停开盘买入极难——买方排队深，仅少量成交
            _ratio = 0.05
        else:
            # 涨停开盘卖出相对容易——卖方向市场提供流动性
            _ratio = tradable_up_ratio
        return _ratio * decay
    if touched_down:
        if side == "sell":
            # 跌停开盘卖出极难——卖方恐慌抛压
            _ratio = 0.05
        else:
            # 跌停开盘买入相对容易——逆势抄底有流动性
            _ratio = tradable_down_ratio
        return _ratio * decay
    return 1.0
