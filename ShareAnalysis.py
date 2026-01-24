import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Any, List
import numpy as np
import ParallelUtils as  utils


class Config:
    """程序配置类"""

    def __init__(self):
        self.HOME_DIRECTORY = os.path.expanduser('~')
        self.SAVE_DIRECTORY = os.path.join(self.HOME_DIRECTORY, 'Downloads', 'CoreNews_Reports')
        self.TEMP_DATA_DIRECTORY = os.path.join(self.SAVE_DIRECTORY, 'ShareData')
        self.DATA_FETCH_RETRIES = 3
        self.DATA_FETCH_DELAY = 5  #
        self.MAX_WORKERS = 15
        self.CODE_ALIASES = {'代码': '股票代码', '证券代码': '股票代码', '股票代码': '股票代码'}
        self.NAME_ALIASES = {'名称': '股票简称', '股票名称': '股票简称', '股票简称': '股票简称', '简称': '股票简称',
                             '简': '股票简称', '证券名称': '股票简称'}
        self.PRICE_ALIASES = {'最新价': '最新价', '现价': '最新价', '当前价格': '最新价', '今收盘': '最新价',
                              '收盘': '最新价', '收盘价': '最新价'}


def format_stock_code(code: str) -> str:
    """根据股票代码的开头数字，添加市场前缀。"""
    code_str = str(code).zfill(6)
    if code_str.startswith('6'):
        return 'sh' + code_str
    elif code_str.startswith(('0', '3')):
        return 'sz' + code_str
    elif code_str.startswith(('4', '8')):
        return 'bj' + code_str
    return code_str


class StockAnalyzer:

    def __init__(self):
        self.config = Config()
        self.today_str = datetime.now().strftime("%Y%m%d")
        self.temp_dir = self.config.TEMP_DATA_DIRECTORY
        os.makedirs(self.temp_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS)
        self.start_time = time.time()

    def _get_file_path(self, base_name: str, cleaned: bool = False) -> str:
        """
        生成临时数据文件的完整路径。
        如果 cleaned=True, 则添加 "_经清洗" 后缀。
        """
        suffix = "_经清洗" if cleaned else ""
        file_name = f"{base_name}{suffix}_{self.today_str}.txt"
        return os.path.join(self.temp_dir, file_name)

    def _load_data_from_cache(self, file_path: str) -> pd.DataFrame:
        """从缓存加载数据。"""
        if os.path.exists(file_path):
            try:
                # 使用 '|'
                # 分隔符，并确保股票代码是字符串格式
                df = pd.read_csv(file_path, sep='|', encoding='utf-8', dtype={'股票代码': str})
                print(f"  - 发现缓存，加载: {os.path.basename(file_path)}")
                return df
            except Exception as e:
                print(f"[WARN] 加载缓存 {os.path.basename(file_path)} 失败: {e}，将重新获取。")
        return pd.DataFrame()

    def _save_data_to_cache(self, df: pd.DataFrame, file_path: str):
        """保存数据到缓存。"""
        try:
            df.to_csv(file_path, sep='|', index=False, encoding='utf-8')
        except Exception as e:
            print(f"[ERROR] 保存数据到缓存 {os.path.basename(file_path)} 失败: {e}")

    def _safe_ak_fetch(self, fetch_func: Callable, file_base_name: str, **kwargs: Any) -> pd.DataFrame:

        # 1. 尝试从【清洗后的缓存】加载数据
        cleaned_file_path = self._get_file_path(file_base_name, cleaned=True)
        cached_df = self._load_data_from_cache(cleaned_file_path)
        if not cached_df.empty:
            return cached_df

        # 2. 如果清洗后的缓存不存在，则尝试从原始获取
        df = pd.DataFrame()
        for i in range(self.config.DATA_FETCH_RETRIES):
            try:
                print(f"  - 正在尝试第 {i + 1}/{self.config.DATA_FETCH_RETRIES} 次获取数据: {file_base_name}...")
                df = fetch_func(**kwargs)
                if df is not None and not df.empty:
                    break  # 成功获取数据，跳出重试循环
                else:
                    print(f"[WARN] 数据返回为空或无效: {file_base_name}，重试中。")
                    time.sleep(self.config.DATA_FETCH_DELAY)
            except Exception as e:
                print(f"[ERROR] 获取 {file_base_name} 时出错: {e}，将在 {self.config.DATA_FETCH_DELAY} 秒后重试。")
                time.sleep(self.config.DATA_FETCH_DELAY)

        if df.empty:
            print(f"[FATAL] 所有重试均失败，返回空 DataFrame: {file_base_name}")
            return pd.DataFrame()

        # 3. 清洗数据并保存到带有 "_经清洗" 后缀的缓存文件
        cleaned_df = self._clean_and_standardize(df, file_base_name)
        if not cleaned_df.empty:
            self._save_data_to_cache(cleaned_df, cleaned_file_path)

        return cleaned_df

    def _clean_and_standardize(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """通用数据清洗和列名标准化。"""
        if df.empty: return df
        df = utils._normalize_fund_data(df)
        # 1. 标准化列名：使用 Config 中的别名映射
        for old, new in self.config.CODE_ALIASES.items():
            if old in df.columns: df.rename(columns={old: new}, inplace=True)
        for old, new in self.config.NAME_ALIASES.items():
            if old in df.columns: df.rename(columns={old: new}, inplace=True)
        for old, new in self.config.PRICE_ALIASES.items():
            if old in df.columns: df.rename(columns={old: new}, inplace=True)

        if '股票代码' not in df.columns:
            return pd.DataFrame()

        # 2. 清洗数据
        df.dropna(subset=['股票代码'], inplace=True)
        df.drop_duplicates(subset=['股票代码'], inplace=True)
        df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)

        if '最新价' in df.columns:
            # 使用 errors='coerce' 将任何非数字值强制转换为 NaN，确保列类型为 float
            df['最新价'] = pd.to_numeric(df['最新价'], errors='coerce')

        # 3. 过滤ST股 (依赖 '股票简称' 列)
        if '股票简称' not in df.columns:
            # 如果没有简称，就无法过滤 ST 股，但仍返回清洗过代码格式的 DF
            cleaned_df = df.copy()
        else:
            cleaned_df = df[~df['股票简称'].str.contains('ST|st|退市|bj|BJ', case=False, na=False)].copy()

        if len(cleaned_df) == 0:
            print(f"[WARN] {df_name} 数据清洗后为空。")
            return pd.DataFrame()

        return cleaned_df

    def _fetch_individual_industry_parallel(self, codes: List[str]) -> pd.DataFrame:
        """重构：并行获取个股行业信息"""

        # 1. 缓存检查 (保持原逻辑)
        cache_name = f"Individual_Industry_Cache"
        file_path = self._get_file_path(cache_name, cleaned=True)
        cached_df = self._load_data_from_cache(file_path)
        if not cached_df.empty:
            return cached_df

        # 2. 定义单个 worker 函数 (闭包)
        def fetch_worker(code):
            for _ in range(self.config.DATA_FETCH_RETRIES):
                try:
                    info_df = ak.stock_individual_info_em(symbol=code)
                    if not info_df.empty:
                        industry = info_df[info_df['item'] == '行业']['value'].values[0]
                        return {'股票代码': code, '行业': industry}
                except Exception:
                    time.sleep(1)
            return {'股票代码': code, '行业': 'N/A'}

        # 3. 调用通用并发工具
        results = utils.run_with_thread_pool(
            items=codes,
            worker_func=fetch_worker,
            max_workers=self.config.MAX_WORKERS,
            desc="获取个股行业信息"
        )

        # 4. 汇总并保存
        final_df = pd.DataFrame(results)
        if not final_df.empty:
            self._save_data_to_cache(final_df, file_path)
        return final_df

    def _get_all_raw_data(self) -> Dict[str, pd.DataFrame]:
        """集中获取所有数据源。"""
        print("\n>>> 正在初始化数据获取和缓存检查...")

        # 1. 基础行情和研报数据
        data = {
            'spot_data_all': self._safe_ak_fetch(ak.stock_zh_a_spot, "A股实时行情"),
            'main_report_raw': self._safe_ak_fetch(ak.stock_profit_forecast_em, "主力研报盈利预测"),
            'financial_abstract_raw': self._safe_ak_fetch(ak.stock_financial_abstract, "财务摘要数据"),
            'market_fund_flow_raw': self._safe_ak_fetch(ak.stock_fund_flow_individual, "5日市场资金流向",
                                                        symbol="5日排行"),
            'market_fund_flow_raw_10': self._safe_ak_fetch(ak.stock_fund_flow_individual, "10日市场资金流向",
                                                           symbol="10日排行"),
            'market_fund_flow_raw_20': self._safe_ak_fetch(ak.stock_fund_flow_individual, "20日市场资金流向",
                                                           symbol="20日排行"),
            'strong_stocks_raw': self._safe_ak_fetch(ak.stock_zt_pool_strong_em, "强势股池",
                                                     date=datetime.now().strftime('%Y%m%d')),
            'consecutive_rise_raw': self._safe_ak_fetch(ak.stock_rank_lxsz_ths, "连续上涨"),
            'ljqs_raw': self._safe_ak_fetch(ak.stock_rank_ljqs_ths, "量价齐升"),
            'cxfl_raw': self._safe_ak_fetch(ak.stock_rank_cxfl_ths, "持续放量"),
        }

        # 2. 均线突破数据 (Akshare接口参数不同，需分开获取)
        data['xstp_10_raw'] = self._safe_ak_fetch(ak.stock_rank_xstp_ths, "向上突破10日均线", symbol="10日均线")
        data['xstp_30_raw'] = self._safe_ak_fetch(ak.stock_rank_xstp_ths, "向上突破30日均线", symbol="30日均线")
        data['xstp_60_raw'] = self._safe_ak_fetch(ak.stock_rank_xstp_ths, "向上突破60日均线", symbol="30日均线")

        # 3. 行业板块数据
        print("\n>>> 正在获取行业板块名称并保存至本地...")
        industry_board_df = ak.stock_board_industry_name_em()
        industry_info_filename = f"行业板块信息_{self.today_str}.txt"
        industry_info_path = os.path.join(self.temp_dir, industry_info_filename)

        try:
            industry_board_df.to_csv(industry_info_path, sep='|', index=False, encoding='utf-8-sig')
            print(f"  - 成功打印行业板块信息至: {industry_info_filename}")
        except Exception as e:
            print(f"[ERROR] 保存行业板块信息失败: {e}")

        data['top_industry_cons_df'] = self._get_top_industry_constituents(industry_board_df)
        data['industry_board_df'] = industry_board_df
        return data

    def _safe_fetch_constituents(self, symbol: str) -> pd.DataFrame:
        """
        带重试机制获取单个行业板块的成分股。
        """
        df = pd.DataFrame()
        for i in range(self.config.DATA_FETCH_RETRIES):
            try:
                # print(f"  - 尝试获取 {symbol} 成分股 (第 {i + 1}/{self.config.DATA_FETCH_RETRIES} 次)")
                df = ak.stock_board_industry_cons_em(symbol=symbol)
                if df is not None and not df.empty:
                    return df
                else:
                    time.sleep(self.config.DATA_FETCH_DELAY)
            except Exception:
                time.sleep(self.config.DATA_FETCH_DELAY)
        return pd.DataFrame()

    def _get_top_industry_constituents(self, industry_board_df: pd.DataFrame) -> pd.DataFrame:
        """重构：获取涨幅前10板块的成分股"""
        if industry_board_df.empty or '板块名称' not in industry_board_df.columns:
            return pd.DataFrame()

        # 1. 缓存检查
        cache_name = "前十板块成分股"
        cleaned_file_path = self._get_file_path(cache_name, cleaned=True)
        cached_df = self._load_data_from_cache(cleaned_file_path)
        if not cached_df.empty:
            return cached_df

        top_industries = industry_board_df.sort_values(by='涨跌幅', ascending=False).head(10)
        # 将 DataFrame 的行转为字典列表，便于传给 worker
        industry_list = top_industries.to_dict('records')

        # 2. 定义 worker
        def fetch_worker(row):
            industry_name = row['板块名称']
            # 调用原本类中定义的 _safe_fetch_constituents
            constituents_df = self._safe_fetch_constituents(symbol=industry_name)

            if constituents_df is not None and not constituents_df.empty:
                if '代码' in constituents_df.columns:
                    constituents_df.rename(columns={'代码': '股票代码'}, inplace=True)

                if '股票代码' in constituents_df.columns:
                    constituents_df['股票代码'] = constituents_df['股票代码'].astype(str).str.zfill(6)
                    constituents_df['所属板块'] = industry_name
                    # 返回清洗后的小 DataFrame
                    return constituents_df[['股票代码', '所属板块']].drop_duplicates()
            return None

        # 3. 调用并发工具
        results = utils.run_with_thread_pool(
            items=industry_list,
            worker_func=fetch_worker,
            max_workers=self.config.MAX_WORKERS,
            desc="获取板块成分股"
        )

        # 4. 合并结果
        if results:
            final_df = pd.concat(results, ignore_index=True).drop_duplicates(subset=['股票代码'])
            self._save_data_to_cache(final_df, cleaned_file_path)
            return final_df
        return pd.DataFrame()

    def _fetch_hist_data_parallel(self, codes: List[str], days: int) -> pd.DataFrame:
        """重构：并行获取历史数据"""

        # 1. 准备日期参数
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")

        # 2. 缓存检查 (保持原逻辑)
        cache_name = f"MACD_hist_data_cache"
        file_path = self._get_file_path(cache_name, cleaned=True)
        cached_df = self._load_data_from_cache(file_path)

        if not cached_df.empty:
            if 'close' in cached_df.columns and 'open' in cached_df.columns:
                return cached_df
            else:
                print("[WARN] 历史数据缓存文件缺少关键英文列，将重新获取。")

        # 3. 定义 worker
        def fetch_worker(code):
            try:
                hist_df = ak.stock_zh_a_hist_tx(
                    symbol=format_stock_code(code),
                    start_date=start_date_str,
                    end_date=end_date_str,
                    adjust="hfq"
                )
                if hist_df is not None and not hist_df.empty:
                    # 标准化列名
                    hist_df.rename(columns={
                        'date': 'date', '日期': 'date',
                        'open': 'open', '开盘': 'open',
                        'close': 'close', '收盘': 'close',
                        'high': 'high', '最高': 'high',
                        'low': 'low', '最低': 'low',
                        'volume': 'volume', '成交量': 'volume', 'amount': 'volume',
                    }, inplace=True, errors='ignore')

                    hist_df['股票代码'] = code
                    if 'date' in hist_df.columns:
                        hist_df['date'] = pd.to_datetime(hist_df['date']).dt.strftime('%Y-%m-%d')

                    cols_to_keep = ['date', 'open', 'close', 'high', 'low', 'volume', '股票代码']
                    return hist_df[[col for col in cols_to_keep if col in hist_df.columns]]
            except Exception:
                pass
            return None

        # 4. 调用并发工具
        results = utils.run_with_thread_pool(
            items=codes,
            worker_func=fetch_worker,
            max_workers=self.config.MAX_WORKERS,
            desc=f"下载 {days} 天历史数据"
        )

        # 5. 合并结果
        if results:
            merged_df = pd.concat(results, ignore_index=True)
            self._save_data_to_cache(merged_df, file_path)
            return merged_df
        return pd.DataFrame()

    def _custom_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [自定义实现] 同时计算 MACD 标准周期 (12, 26, 9) 和加速周期 (6, 13, 5) 的快慢线金叉信号。
        **改进：区分零轴上金叉和零轴下金叉。**
        """
        if 'close' not in df.columns:
            return df

        close = df['close']

        # 定义需要计算的MACD周期组合
        macd_periods = {
            '12269': (12, 26, 9),  # 标准中长线周期
            '6135': (6, 13, 5)  # 短线/加速周期
        }

        for name, (fast, slow, signal) in macd_periods.items():
            # 1. 计算快线和慢线
            ema_fast_col = f'EMA_{fast}_{name}'
            ema_slow_col = f'EMA_{slow}_{name}'

            # adjust=False 确保权重符合技术分析的传统公式
            df[ema_fast_col] = close.ewm(span=fast, adjust=False).mean()
            df[ema_slow_col] = close.ewm(span=slow, adjust=False).mean()

            # 2. 计算 DIF
            dif_col = f'DIF_{name}'
            df[dif_col] = df[ema_fast_col] - df[ema_slow_col]

            # 3. 计算 DEA
            dea_col = f'DEA_{name}'
            df[dea_col] = df[dif_col].ewm(span=signal, adjust=False).mean()

            # 4. 判断 MACD 金叉信号 (DIF 上穿 DEA) 和零轴位置
            cross_col = f'MACD_{name}_CROSS'
            signal_col = f'MACD_{name}_SIGNAL_DETAIL'  # <== 新增的列

            # 判定金叉的布尔序列
            is_cross = (df[dif_col] > df[dea_col]) & \
                       (df[dif_col].shift(1).fillna(0) <= df[dea_col].shift(1).fillna(0))

            # **核心修改：区分零轴上/下金叉**
            df[signal_col] = np.where(
                is_cross,
                np.where(
                    # 零轴上金叉条件: 金叉发生，且 DIF 和 DEA 均大于 0
                    (df[dif_col] > 0) & (df[dea_col] > 0),
                    '零轴上金叉',
                    # 零轴下金叉条件: 金叉发生，且 DIF 或 DEA 小于等于 0
                    '零轴下金叉'
                ),
                ''  # 非金叉，返回空字符串
            )

            # 保持原有的 CROSS 标志位 (非必须，但保持兼容)
            df[cross_col] = np.where(is_cross, 1, 0)

            # 5. 清理中间计算列 (保留 DIF 和 DEA)
            df.drop(columns=[ema_fast_col, ema_slow_col],
                    inplace=True, errors='ignore')

        return df


    def _calculate_macd_momentum(self, df: pd.DataFrame, dif_col: str, dea_col: str) -> str:
        """
        计算 MACD 动能状态: 加速上涨/减速上涨/加速下跌/减速下跌
        """
        if len(df) < 2:
            return "N/A (数据不足)"

        # 获取最新的 DIF, DEA 值和前一天的 DIF 值
        latest_dif = df[dif_col].iloc[-1]
        latest_dea = df[dea_col].iloc[-1]
        prev_dif = df[dif_col].iloc[-2]

        # DIF 线的变化 (MACD 柱的变化方向)
        dif_change = latest_dif - prev_dif

        momentum_state = ""

        if latest_dif >= latest_dea:
            # DIF 在 DEA 之上 (多头区域/红柱)
            if dif_change > 0:
                momentum_state = "加速上涨 (红柱加长)"
            elif dif_change <= 0:
                momentum_state = "减速上涨 (红柱缩短)"
        else:
            # DIF 在 DEA 之下 (空头区域/绿柱)
            if dif_change < 0:
                momentum_state = "加速下跌 (绿柱加长)"
            elif dif_change >= 0:
                momentum_state = "减速下跌 (绿柱缩短)"

        return momentum_state


    def _classify_cci_level(self, cci_value: float) -> str:
        """根据CCI值，将其分类为专业的市场状态术语。"""
        if pd.isna(cci_value):
            return 'N/A'

        # 1. 极度超买区 (Ultra-Overbought)
        if cci_value > 200:
            return f'极度超买 ({cci_value:.2f})'

        # 2. 强势超买区 (Strong Overbought)
        elif cci_value >= 100:
            return f'强势超买 ({cci_value:.2f})'

        # 3. 常态波动/整理区 (Normal Fluctuation/Consolidation)
        elif cci_value > -100:
            # 在报告中我们只关注超买/超卖，常态波动直接返回空字符串
            return ''

        # 4. 弱势超卖区 (Weak Oversold)
        elif cci_value >= -200:
            return f'弱势超卖 ({cci_value:.2f})'

        # 5. 极度超卖区 (Ultra-Oversold)
        else:  # cci_value < -200
            return f'极度超卖 ({cci_value:.2f})'

    # ==========================================================================

    def _save_ta_signals_to_txt(self, ta_signals: Dict[str, pd.DataFrame]):
        """
        将技术指标信号结果保存到独立的 TXT 文件。
        """
        print("\n>>> 正在保存技术指标信号到本地 TXT 文件...")

        save_dir = self.config.TEMP_DATA_DIRECTORY
        today_str = self.today_str

        for indicator_name, df in ta_signals.items():
            if df is None or df.empty:
                continue

            # 命名格式: MACD_12269_Signals_YYYYMMDD.txt
            file_name = f"{indicator_name}_Signals_{today_str}.txt"
            file_path = os.path.join(save_dir, file_name)

            try:
                # 使用与缓存文件相同的格式：'|'
                # 分隔符，utf-8 编码，不包含索引
                df.to_csv(file_path, sep='|', index=False, encoding='utf-8')
                print(f"  - 成功保存 {indicator_name} 信号文件: {file_name}")
            except Exception as e:
                print(f"[ERROR] 保存 {indicator_name} 信号文件失败: {e}")


    def _process_ta_signals(self, all_codes: List[str], hist_df_all: pd.DataFrame, spot_df: pd.DataFrame) -> Dict[
        str, pd.DataFrame]:
        """计算并提取所有技术指标信号。"""
        print(f"\n正在对 {len(all_codes)} 只股票进行技术分析...")

        # MACD指标增加两种周期，键名不变，但内容将包含零轴信息
        # MACD指标增加两种周期，键名不变，但内容将包含零轴信息
        ta_signals = {'MACD_12269': [], 'MACD_6135': [], 'KDJ': [], 'CCI': [], 'RSI': [], 'BOLL': [],
                      'MACD_DIF_MOMENTUM': []}  # <== 新增这一项

        if hist_df_all.empty:
            print("[WARN] 历史数据为空，跳过技术分析。")
            # 返回包含空 DataFrame 的字典
            return {key: pd.DataFrame(columns=['股票代码', f'{key}_Signal']) for key in ta_signals.keys()}

        # 确保数据是连续的且已排序 (使用标准化的 'date' 列)
        hist_df_all.sort_values(['股票代码', 'date'], inplace=True)

        for code in all_codes:
            # df 现在应该已经包含了 'open', 'close', 'high', 'low' 等标准列名
            df = hist_df_all[hist_df_all['股票代码'] == code].copy()

            if df.empty or len(df) < 30:  # 至少需要30条数据用于BOLL和KDJ
                continue

            # 确保所需列是数字类型 (使用英文列名)
            for col in ['close', 'open', 'high', 'low']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(subset=['close'], inplace=True)
            if df.empty: continue

            # 确保关键列存在 (现在只检查英文小写)
            if 'close' not in df.columns or 'open' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
                print(f"[ERROR] 股票 {code}: 历史数据中缺少必要的 OHLC 列，跳过 TA 计算。")
                continue

            try:
                # === [新增] 量比计算与过滤逻辑 ===
                # 计算 5 日均量（不含当日）
                df['vol_ma5'] = df['volume'].rolling(window=5).mean().shift(1)

                curr_vol = df['volume'].iloc[-1]
                prev_vol_ma5 = df['vol_ma5'].iloc[-1]

                # 计算今日涨跌幅 (用于过滤)
                curr_pct_change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) if len(df) > 1 else 0

                # 计算量比数值
                vol_ratio = 1.0 if pd.isna(prev_vol_ma5) or prev_vol_ma5 == 0 else curr_vol / prev_vol_ma5

                # 判定状态标签
                vol_status = "缩量" if vol_ratio < 0.8 else "温和" if vol_ratio < 1.5 else "放量"
                if vol_ratio > 2.5: vol_status = "巨量"

                # 【核心过滤】：如果股价上涨 且 量比 < 0.7 (严重无量上涨)，剔除该股
                if curr_pct_change > 0 and vol_ratio < 0.7:
                    continue  # 关键点：直接跳过，不参与后续计算，不计入报表


                # === [量比逻辑结束] ===
                df = self._custom_macd(df)

                try:
                    latest_row = df.iloc[-1]
                    # 计算标准周期动能
                    mom_12269 = self._calculate_macd_momentum(df, 'DIF_12269', 'DEA_12269')
                    # 计算加速周期动能
                    mom_6135 = self._calculate_macd_momentum(df, 'DIF_6135', 'DEA_6135')

                    ta_signals['MACD_DIF_MOMENTUM'].append({
                        '股票代码': code,
                        'MACD_12269_DIF': latest_row.get('DIF_12269', 0),
                        'MACD_12269_动能': mom_12269,
                        'MACD_6135_DIF': latest_row.get('DIF_6135', 0),
                        'MACD_6135_动能': mom_6135,
                    })
                except Exception as e:
                    print(f"[WARN] {code} 动能计算失败: {e}")

                # 提取 MACD 12269 信号 (使用新的详细列)
                detail_col_12269 = 'MACD_12269_SIGNAL_DETAIL'
                if detail_col_12269 in df.columns:
                    signal_detail = df[detail_col_12269].iloc[-1]
                    if signal_detail != '':
                        ta_signals['MACD_12269'].append({'股票代码': code, 'MACD_12269_Signal': signal_detail})

                # 提取 MACD 6135 信号 (使用新的详细列)
                detail_col_6135 = 'MACD_6135_SIGNAL_DETAIL'
                if detail_col_6135 in df.columns:
                    signal_detail = df[detail_col_6135].iloc[-1]
                    if signal_detail != '':
                        ta_signals['MACD_6135'].append({'股票代码': code, 'MACD_6135_Signal': signal_detail})
                        # --- MACD 自定义计算结束 ---

                        # 2. KDJ 增强版 (集成 J值、背离、趋势过滤)

                        df.ta.stoch(append=True, close='close', high='high', low='low')
                        kdj_cols = [col for col in df.columns if col.startswith('STOCHk_') or col.startswith('STOCHd_')]

                        if len(kdj_cols) >= 2:
                            k_col = kdj_cols[0]
                            d_col = kdj_cols[1]
                            j_col = 'KDJ_J'

                            # 1. 手动计算 J 值: J = 3K - 2D
                            df[j_col] = 3 * df[k_col] - 2 * df[d_col]

                            # 2. 定义基础状态
                            # 金叉：K上穿D
                            kdj_cross = (df[k_col] > df[d_col]) & (df[k_col].shift(1) <= df[d_col].shift(1))
                            # 极度超卖：J值曾在过去3天内小于 0
                            j_oversold = df[j_col].shift(1).rolling(window=3).min() < 0
                            # 普通超卖：K和D都在20以下
                            kd_oversold = (df[k_col] < 20) & (df[d_col] < 20)

                            # 3. 定义高级信号：底背离 (Divergence)
                            # 逻辑：价格(Low)创最近10天新低，但K值没有创最近10天新低
                            # 简化算法：当前是金叉点，且价格低于前一个金叉点，但K值高于前一个金叉点 (这里用简化窗口法)
                            window = 10
                            curr_low = df['low'].iloc[-1]
                            min_low_window = df['low'].iloc[-window:-1].min()
                            curr_k = df[k_col].iloc[-1]
                            min_k_window = df[k_col].iloc[-window:-1].min()

                            # 背离判断：当前价格接近窗口最低价，但K值明显高于窗口最低K值
                            is_divergence = (curr_low <= min_low_window * 1.02) & (curr_k > min_k_window * 1.1)

                            # 4. 辅助过滤：股价站上5日线 (右侧确认)
                            ma5 = df['close'].rolling(window=5).mean()
                            above_ma5 = df['close'] > ma5

                            # --- 信号提取逻辑 ---
                            current_idx = df.index[-1]  # 最后一行的索引
                            last_row = df.iloc[-1]
                            prev_row = df.iloc[-2]

                            signal_msg = ""

                            # 场景 A: 极值 J 线反转 (最强短线信号)
                            # 条件：昨日 J < 0, 今日 J > 5, 且今日金叉
                            if prev_row[j_col] < 0 and last_row[j_col] > 5 and kdj_cross.iloc[-1]:
                                signal_msg = "极值J线反转"

                            # 场景 B: 底背离金叉 (波段反转信号)
                            # 条件：金叉 + 判定背离 + K<30 (低位)
                            elif kdj_cross.iloc[-1] and is_divergence and last_row[k_col] < 30:
                                signal_msg = "底背离金叉"

                            # 场景 C: 超卖区金叉 + 趋势确认 (稳健信号)
                            # 条件：最近5天有过超卖 + 金叉 + 站上5日线
                            elif (kd_oversold.iloc[-5:-1].sum() > 0) and kdj_cross.iloc[-1] and above_ma5.iloc[-1]:
                                signal_msg = "趋势确认金叉"

                            # 场景 D: 普通低位金叉 (保留原有逻辑作为兜底)
                            elif (kd_oversold.iloc[-5:-1].sum() > 0) and kdj_cross.iloc[-1]:
                                signal_msg = "低位超卖金叉"

                            # 如果有信号，则记录
                            if signal_msg:
                                ta_signals['KDJ'].append({
                                    '股票代码': code,
                                    'KDJ_Signal': f"{signal_msg} (K={last_row[k_col]:.1f}, J={last_row[j_col]:.1f})"
                                })

                # 3. CCI (专业分类) - 使用 pandas_ta
                df.ta.cci(append=True, close='close', high='high', low='low')
                cci_cols = [col for col in df.columns if col.startswith('CCI_')]
                if cci_cols:
                    cci_col = cci_cols[0]

                    # **核心改进点：使用专业分类函数**
                    current_cci = df[cci_col].iloc[-1]
                    cci_signal = self._classify_cci_level(current_cci)

                    # 只有当CCI状态是超买或超卖时才记录，跳过常态波动/整理
                    if cci_signal:
                        ta_signals['CCI'].append(
                            {'股票代码': code, 'CCI_Signal': cci_signal})

                # 4. RSI (超卖低位) - 使用 pandas_ta
                df.ta.rsi(append=True, close='close', length=14)
                rsi_cols = [col for col in df.columns if col.startswith('RSI_')]
                if rsi_cols:
                    rsi_col = rsi_cols[0]
                    curr_rsi = df[rsi_col].iloc[-1]
                    window = 10
                    curr_low = df['low'].iloc[-1]
                    min_low_window = df['low'].iloc[-window:-1].min()
                    min_rsi_window = df[rsi_col].iloc[-window:-1].min()
                    is_price_low = curr_low <= (min_low_window * 1.02)
                    is_rsi_divergence = (is_price_low) and \
                                        (curr_rsi > min_rsi_window * 1.05) and \
                                        (curr_rsi < 50)
                    if is_rsi_divergence:
                        ta_signals['RSI'].append({'股票代码': code,
                                                  'RSI_Signal': f"RSI底背离 (Price低={curr_low}, RSI={curr_rsi:.1f}, 前RSI低={min_rsi_window:.1f})"})

                # 5. BOLL (低波/缩口) - 使用 pandas_ta
                df.ta.bbands(append=True, length=20, std=2, close='close')
                boll_cols = [col for col in df.columns if col.startswith('BBL_')]
                if boll_cols:
                    lower_band = boll_cols[0]
                    upper_band = [col for col in df.columns if col.startswith('BBU_')][0]
                    # 计算带宽
                    df['BOLL_BANDWIDTH'] = (df[upper_band] - df[lower_band]) / df['close']
                    # 最近 N 天的平均带宽低于历史平均带宽（表示缩口）
                    if df['BOLL_BANDWIDTH'].iloc[-5:].mean() < df['BOLL_BANDWIDTH'].mean():
                        # 仅记录信号，不包含评分
                        ta_signals['BOLL'].append({'股票代码': code, 'BOLL_Signal': '低波/缩口'})

            except KeyError as ke:
                # 捕获 MACDH_12_26_9 或其他指标列名找不到的错误
                print(f"[ERROR] 计算 {code} 的 TA 指标时出错: 关键指标列名丢失 ({ke})")
            except Exception as e:
                print(f"[ERROR] 计算 {code} 的 TA 指标时出错: {e}")
                continue

        # 将结果列表转换为 DataFrame
        final_ta_dfs = {}
        for key, value in ta_signals.items():
            final_ta_dfs[key] = pd.DataFrame(value)

        return final_ta_dfs

    def _process_xstp_and_filter(self, raw_data: Dict[str, pd.DataFrame], spot_df: pd.DataFrame) -> pd.DataFrame:
        """处理并合并均线突破数据，并进行多头排列筛选。"""
        print("正在处理并合并均线突破数据...")

        # 1. 清洗均线数据
        processed_df10 = raw_data['xstp_10_raw'].rename(columns={'最新价': '10日均线最新价'})
        processed_df30 = raw_data['xstp_30_raw'].rename(columns={'最新价': '30日均线最新价'})
        processed_df60 = raw_data['xstp_60_raw'].rename(columns={'最新价': '60日均线最新价'})

        # 2. 合并
        # 使用concat时只保留代码和简称，避免价格列冲突
        merged_df = pd.concat([
            processed_df10[['股票代码', '股票简称']].dropna(subset=['股票代码']),
            processed_df30[['股票代码', '股票简称']].dropna(subset=['股票代码']),
            processed_df60[['股票代码', '股票简称']].dropna(subset=['股票代码'])
        ]).drop_duplicates(subset=['股票代码'])

        # 3. 重新合并均线价格，确保同一行有所有数据
        xstp_base = merged_df[['股票代码', '股票简称']].drop_duplicates()
        xstp_base = pd.merge(xstp_base, processed_df10[['股票代码', '10日均线最新价']], on='股票代码', how='left')
        xstp_base = pd.merge(xstp_base, processed_df30[['股票代码', '30日均线最新价']], on='股票代码', how='left')
        xstp_base = pd.merge(xstp_base, processed_df60[['股票代码', '60日均线最新价']], on='股票代码', how='left')

        # 4. 合并实时价格 (此处仍然按代码合并，以便于均线计算的准确性)
        xstp_base = pd.merge(xstp_base, spot_df[['股票代码', '最新价']], on='股票代码', how='left')

        # 5. 类型转换和过滤
        cols_to_convert = [col for col in xstp_base.columns if '最新价' in col or col == '最新价']
        for col in cols_to_convert:
            # 导入 numpy 是为了使用 np.inf，所以在这里重新导入或使用 float('-inf')
            xstp_base[col] = pd.to_numeric(xstp_base[col], errors='coerce')

        # 过滤条件: 1. 最新价>10日均线 2. 多头排列 (10>30 或 30>60)
        filtered_df = xstp_base[
            (xstp_base['最新价'] > xstp_base['10日均线最新价']) &
            (
                    (xstp_base['10日均线最新价'] > xstp_base['30日均线最新价'].fillna(float('-inf'))) |
                    (xstp_base['30日均线最新价'] > xstp_base['60日均线最新价'].fillna(float('-inf')))
            )
            ].copy()

        # 添加完全多头排列标记
        filtered_df['完全多头排列'] = filtered_df.apply(
            # 需要确保在计算前处理 NaN 值，这里使用一个较大的负数代替 NaN
            lambda row: '是' if row['10日均线最新价'] > row['30日均线最新价'] and row['30日均线最新价'] > row[
                '60日均线最新价'] else '否',
            axis=1
        )

        # 重新命名 '最新价' 为 '当前价格' 以避免与均线价混淆
        filtered_df.rename(columns={'最新价': '当前价格'}, inplace=True)
        return filtered_df.fillna('N/A')


    def _consolidate_data(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """合并所有数据源和信号，生成最终汇总报告。"""
        print("\n>>> 正在汇总所有数据和信号 (技术指标作为独立列)...")

        # 1. 提取所有有信号的股票代码作为基础 (
        all_codes = set()
        data_sources = [
            processed_data.get('processed_main_report'),
            processed_data.get('processed_xstp_df'),
            processed_data.get('market_fund_flow_raw'),
            processed_data.get('strong_stocks_raw'),
            processed_data.get('consecutive_rise_raw'),
            processed_data.get('ljqs_raw'),
            processed_data.get('cxfl_raw'),
            processed_data.get('MACD_12269', pd.DataFrame()),  # MACD 标准周期
            processed_data.get('MACD_6135', pd.DataFrame()),  # MACD 加速周期
            processed_data.get('KDJ', pd.DataFrame()),
            processed_data.get('CCI', pd.DataFrame()),
            processed_data.get('RSI', pd.DataFrame()),
            processed_data.get('BOLL', pd.DataFrame()),
        ]

        for df in data_sources:
            if df is not None and not df.empty and '股票代码' in df.columns:
                all_codes.update(df['股票代码'].unique())

        if not all_codes: return pd.DataFrame()
        final_df = pd.DataFrame(list(all_codes), columns=['股票代码'])

        # 强制代码列表为字符串类型
        final_df['股票代码'] = final_df['股票代码'].astype(str)

        # 2. 鲁棒地合并基础信息 (简称, 价格)
        spot_df = processed_data.get('spot_data_all', pd.DataFrame())
        report_df_base = processed_data.get('processed_main_report', pd.DataFrame())

        # 确保实时行情代码列表是字符串类型
        if '股票代码' in spot_df.columns:
            spot_df['股票代码'] = spot_df['股票代码'].astype(str)

        # 2a.
        # 构建名称源：从实时行情和研报数据中收集名称 (仍以代码为基础确保唯一性)
        name_source_spot = spot_df[['股票代码', '股票简称']].drop_duplicates(
            subset=['股票代码']) if '股票简称' in spot_df.columns else pd.DataFrame()
        name_source_report = report_df_base[['股票代码', '股票简称']].drop_duplicates(
            subset=['股票代码']) if '股票简称' in report_df_base.columns else pd.DataFrame()

        # 2b.
        # 合并所有名称源：优先使用实时行情 (spot_df) 的名称
        all_names = pd.concat([name_source_spot, name_source_report]).drop_duplicates(subset=['股票代码'], keep='first')

        # 2c.
        # 合并名称到最终报告 (按代码合并，保证基础表的简称是正确的)
        final_df = pd.merge(final_df, all_names, on='股票代码', how='left')

        # 1. 从实时行情数据中提取用于价格合并的列 (简称和最新价)
        if '股票简称' not in spot_df.columns:
            # 严重警告：如果实时行情数据中没有股票简称，则回退到使用股票代码。
            print("[FATAL] 实时行情数据中缺少 '股票简称' 列，无法按要求按简称关联。回退到按代码关联。")
            price_source_key = '股票代码'
            price_source = spot_df[['股票代码', '最新价']].copy()
        else:
            price_source_key = '股票简称'
            price_source = spot_df[['股票简称', '最新价']].copy()

            # 2. 严格筛选：对价格数据进行数值转换，并只保留非空、非零的价格
            price_source['最新价'] = pd.to_numeric(price_source['最新价'], errors='coerce')
            price_source = price_source[
                (price_source['最新价'].notna()) &
                (price_source['最新价'] > 0)
                ].copy()

            # 3. 关键修改点：根据用户要求，按 "股票简称" 进行去重（处理非唯一性问题）
            price_source = price_source.drop_duplicates(subset=[price_source_key], keep='first')

        # 4. 移除旧的 '最新价' 列 (来自XSTP或之前的合并)
        final_df.drop(columns=['最新价'], errors='ignore', inplace=True)

        # 5. 合并价格 (使用 '股票简称' 或回退键)
        if price_source_key in final_df.columns:
            # 确保合并键是字符串类型
            final_df[price_source_key] = final_df[price_source_key].astype(str)
            price_source[price_source_key] = price_source[price_source_key].astype(str)

            final_df = pd.merge(final_df, price_source, on=price_source_key, how='left')

        # 打印诊断信息
        valid_prices_count = final_df['最新价'].notna().sum() if '最新价' in final_df.columns else 0
        print(
            f"  - 实时行情数据 (最新价) 成功通过 '{price_source_key}' 关联的有效价格数量: {valid_prices_count} / {len(final_df)}")

        # 6. 填充空值
        final_df['股票简称'] = final_df['股票简称'].fillna('N/A')
        # 将价格的 NaN 填充为 'N/A'
        final_df['最新价'] = final_df['最新价'].fillna('N/A')

        report_df = processed_data['processed_main_report'][['股票代码', '机构投资评级(近六个月)-买入']]
        report_df.rename(columns={'机构投资评级(近六个月)-买入': '研报买入次数'}, inplace=True)
        final_df = pd.merge(final_df, report_df, on='股票代码', how='left')
        final_df['研报买入次数'] = pd.to_numeric(final_df['研报买入次数'], errors='coerce').fillna(0).astype(int)

        # B. 均线信号 (多头排列)
        xstp_df = processed_data['processed_xstp_df']
        xstp_cols = ['股票代码', '完全多头排列', '当前价格', '10日均线最新价', '30日均线最新价', '60日均线最新价']


        if not xstp_df.empty and '股票代码' in xstp_df.columns:
            # 1. 筛选出实际存在的列
            cols_present = [col for col in xstp_cols if col in xstp_df.columns]

            # 2. 执行去重和合并
            # 此时 xstp_df 肯定包含 '股票代码'，所以 drop_duplicates 是安全的
            merge_df = xstp_df[cols_present].drop_duplicates(subset=['股票代码'])

            # 3. 执行合并
            final_df = pd.merge(final_df, merge_df, on='股票代码', how='left')


        if '完全多头排列' not in final_df.columns:
            final_df['完全多头排列'] = '否'
        else:
            # 如果存在，则将合并产生的 NaN 填充为 '否'
            final_df['完全多头排列'] = final_df['完全多头排列'].fillna('否')

        # C. 资金流向-5day

        fund_flow_df = processed_data.get('market_fund_flow_raw', pd.DataFrame())
        if not fund_flow_df.empty and '股票简称' in fund_flow_df.columns and '资金流入净额' in fund_flow_df.columns:
            merge_df = fund_flow_df[['股票简称', '资金流入净额']].drop_duplicates(subset=['股票简称'])
            final_df = pd.merge(final_df, merge_df, on='股票简称', how='left')
            final_df['5日资金流入'] = final_df['资金流入净额']
            final_df.drop(columns=['资金流入净额'], errors='ignore', inplace=True)

        fund_flow_df_10 = processed_data.get('market_fund_flow_raw_10', pd.DataFrame())
        if not fund_flow_df_10.empty and '股票简称' in fund_flow_df_10.columns and '资金流入净额' in fund_flow_df_10.columns:
            merge_df_10 = fund_flow_df_10[['股票简称', '资金流入净额']].drop_duplicates(subset=['股票简称'])
            final_df = pd.merge(final_df, merge_df_10, on='股票简称', how='left')
            final_df['10日资金流入'] = final_df['资金流入净额']
            final_df.drop(columns=['资金流入净额'], errors='ignore', inplace=True)

        fund_flow_df_20 = processed_data.get('market_fund_flow_raw_20', pd.DataFrame())
        if not fund_flow_df_20.empty and '股票简称' in fund_flow_df_20.columns and '资金流入净额' in fund_flow_df_20.columns:
            merge_df_20 = fund_flow_df_20[['股票简称', '资金流入净额']].drop_duplicates(subset=['股票简称'])
            final_df = pd.merge(final_df, merge_df_20, on='股票简称', how='left')
            final_df['20日资金流入'] = final_df['资金流入净额']
            final_df.drop(columns=['资金流入净额'], errors='ignore', inplace=True)

        f5_col, f10_col, f20_col = '5日资金流入', '10日资金流入', '20日资金流入'
        if all(col in final_df.columns for col in [f5_col, f10_col, f20_col]):
            def calculate_trend(row):
                # 因为之前清洗过，这里直接取 float 即可
                v5 = float(row[f5_col]) if not pd.isna(row[f5_col]) else 0
                v10 = float(row[f10_col]) if not pd.isna(row[f10_col]) else 0
                v20 = float(row[f20_col]) if not pd.isna(row[f20_col]) else 0

                # 业务逻辑：(5日>10日 或 5日>20日) 且 5日为正流入
                if (v5 > v10 or v5 > v20) and v5 > 0:
                    return "动能增强"
                elif v5 > 0:
                    return "流入"
                else:
                    return "谨慎观察"

            # 创建新列
            final_df['资金趋势向上'] = final_df.apply(calculate_trend, axis=1)

            # 调整列顺序，将参考列放到资金列旁边便于对比
            cols = list(final_df.columns)
            if '资金趋势向上' in cols:
                target_idx = cols.index(f5_col)
                cols.insert(target_idx + 1, cols.pop(cols.index('资金趋势向上')))
                final_df = final_df[cols]



        # D. 强势股池
        if not processed_data['strong_stocks_raw'].empty:
            strong_codes = processed_data['strong_stocks_raw']['股票代码'].tolist()
            final_df['强势股'] = final_df['股票代码'].apply(lambda x: '是' if x in strong_codes else '否')
        else:
            final_df['强势股'] = '否'

        # E. 连续上涨天数
        rise_df = processed_data['consecutive_rise_raw']
        if not rise_df.empty:
            rise_df = rise_df[['股票代码', '连涨天数']].drop_duplicates(subset=['股票代码'])
            final_df = pd.merge(final_df, rise_df, on='股票代码', how='left').fillna({'连涨天数': 0})
        else:
            final_df['连涨天数'] = 0

        final_df['连涨天数'] = final_df['连涨天数'].astype(int)

        # F. 量价齐升
        if not processed_data['ljqs_raw'].empty:
            ljqs_codes = processed_data['ljqs_raw']['股票代码'].tolist()
            final_df['量价齐升'] = final_df['股票代码'].apply(lambda x: '是' if x in ljqs_codes else '否')
        else:
            final_df['量价齐升'] = '否'

        # G. 持续放量天数
        cxfl_df = processed_data['cxfl_raw']
        if not cxfl_df.empty:
            cxfl_df = cxfl_df[['股票代码', '放量天数']].drop_duplicates(subset=['股票代码'])
            final_df = pd.merge(final_df, cxfl_df, on='股票代码', how='left').fillna({'放量天数': 0})
        else:
            final_df['放量天数'] = 0

        final_df['放量天数'] = final_df['放量天数'].astype(int)


        ta_dfs_to_merge = []

        # 1. MACD 12269 (标准周期)
        macd_df_standard = processed_data.get('MACD_12269', pd.DataFrame())
        if not macd_df_standard.empty:
            # 确保使用正确的信号列名
            ta_dfs_to_merge.append(macd_df_standard[['股票代码', 'MACD_12269_Signal']].rename(
                columns={'MACD_12269_Signal': 'MACD_12269'}))

        # 2. MACD 6135 (加速周期)
        macd_df_fast = processed_data.get('MACD_6135', pd.DataFrame())
        if not macd_df_fast.empty:
            # 确保使用正确的信号列名
            ta_dfs_to_merge.append(macd_df_fast[['股票代码', 'MACD_6135_Signal']].rename(
                columns={'MACD_6135_Signal': 'MACD_6135'}))

        # 3. KDJ
        kdj_df = processed_data.get('KDJ', pd.DataFrame())
        if not kdj_df.empty:
            ta_dfs_to_merge.append(kdj_df[['股票代码', 'KDJ_Signal']].rename(
                columns={'KDJ_Signal': 'KDJ_Signal'}))

        # 4. CCI
        cci_df = processed_data.get('CCI', pd.DataFrame())
        if not cci_df.empty:
            # 改进后的CCI信号已包含数值和状态，此处只保留信号字符串
            ta_dfs_to_merge.append(cci_df[['股票代码', 'CCI_Signal']].rename(
                columns={'CCI_Signal': 'CCI_Signal'}))

        # 5. RSI
        rsi_df = processed_data.get('RSI', pd.DataFrame())
        if not rsi_df.empty:
            # 只保留信号的文字描述，移除数值
            rsi_df['RSI_Signal'] = rsi_df['RSI_Signal'].astype(str).str.split(' ').str[0]
            ta_dfs_to_merge.append(rsi_df[['股票代码', 'RSI_Signal']].rename(
                columns={'RSI_Signal': 'RSI_Signal'}))

        # 6. BOLL
        boll_df = processed_data.get('BOLL', pd.DataFrame())
        if not boll_df.empty:
            ta_dfs_to_merge.append(boll_df[['股票代码', 'BOLL_Signal']].rename(
                columns={'BOLL_Signal': 'BOLL_Signal'}))

        # 将所有 TA 信号合并到 final_df
        for ta_df in ta_dfs_to_merge:
            final_df = pd.merge(final_df, ta_df.drop_duplicates(subset=['股票代码']), on='股票代码', how='left')

        # 合并 MACD 动能状态数据

        momentum_df = processed_data.get('MACD_DIF_MOMENTUM', pd.DataFrame())
        if not momentum_df.empty and '股票代码' in momentum_df.columns:
            # 动能数据本身不作为过滤条件，直接 Left Join
            final_df = pd.merge(final_df, momentum_df, on='股票代码', how='left')
            # 填充空值
            for col in ['MACD_12269_动能', 'MACD_6135_动能']:
                if col in final_df.columns:
                    final_df[col] = final_df[col].fillna('')

        # 对新的 TA 信号列填充空值
        for col in ['MACD_12269', 'MACD_6135', 'KDJ_Signal', 'CCI_Signal', 'RSI_Signal', 'BOLL_Signal']:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna('')
            else:
                # 如果某个信号源为空，则补充空列以保持一致性
                final_df[col] = ''
        top_ind_df = processed_data.get('top_industry_cons_df', pd.DataFrame())
        if not top_ind_df.empty:
            top_codes = set(top_ind_df['股票代码'].astype(str).unique())
            final_df['TOP10行业']= final_df['股票代码'].apply(lambda x:'是' if str(x) in top_codes else '否')
        else:
            final_df['TOP10行业'] = '否'

        industry_df = processed_data.get('individual_industry', pd.DataFrame())
        if not industry_df.empty:
            final_df = pd.merge(final_df, industry_df, on='股票代码', how='left')
            final_df['行业'] = final_df['行业'].fillna('N/A')

        # 4. 过滤掉没有任何主要信号的股票
        def has_any_signal(row):
            # 这里的研报买入次数列在 run() 方法中已过滤，且在 merge 填充 NaN 为 0，故此处检查 > 0 即可
            return (row['研报买入次数'] > 0 or
                    row['完全多头排列'] == '是' or
                    row['强势股'] == '是' or
                    row['量价齐升'] == '是' or
                    row.get('TOP10行业') == '是' or
                    row['MACD_12269'] != '' or
                    row['MACD_6135'] != '' or
                    row['KDJ_Signal'] != '' or
                    row['CCI_Signal'] != '' or
                    row['RSI_Signal'] != '' or
                    row['BOLL_Signal'] != ''
                    )

        final_df = final_df[final_df.apply(has_any_signal, axis=1)].copy()

        # 5. 整理输出并排序 (按研报买入次数、连涨天数、放量天数降序排序)
        final_df.sort_values(by=['研报买入次数', '连涨天数', '放量天数'], ascending=[False, False, False], inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        final_df.insert(0, '序号', range(1, len(final_df) + 1))

        final_df['完整股票代码'] = final_df['股票代码'].apply(format_stock_code)
        final_df['股票链接'] = "https://hybrid.gelonghui.com/stock-check/" + final_df['完整股票代码']

        # 3. 移除临时列
        final_df.drop(columns=['完整股票代码'], inplace=True, errors='ignore')

        if '当前价格' in final_df.columns and '最新价' in final_df.columns:
            final_df.drop(columns=['当前价格'], inplace=True, errors='ignore')

        # 最终列顺序 (手动调整，确保最重要的信息在前)
        base_cols = ['序号', '股票代码', '股票简称', '行业','最新价']

        signal_cols = [
            '强势股', '量价齐升', '连涨天数', '放量天数','TOP10行业',
            'MACD_12269', 'MACD_12269_动能', 'MACD_12269_DIF',  # <== 新增
            'MACD_6135', 'MACD_6135_动能', 'MACD_6135_DIF',  # <== 新增
            'KDJ_Signal', 'CCI_Signal', 'RSI_Signal', 'BOLL_Signal',
        ]
        report_cols = [
            '研报买入次数', '完全多头排列', '10日均线最新价', '30日均线最新价', '60日均线最新价',
            '5日资金流入', '10日资金流入', '20日资金流入'
        ]

        # 将新列 '股票链接' 添加到列列表的末尾
        final_cols = base_cols + signal_cols + report_cols + ['股票链接']
        final_df = final_df[[col for col in final_cols if col in final_df.columns]]

        return final_df

    def _generate_report(self, sheets_data: Dict[str, pd.DataFrame]):
        """生成 Excel 报告。"""
        print(f"\n>>> 正在生成 Excel 报告...")
        report_path = os.path.join(self.config.SAVE_DIRECTORY, f"股票筛选报告_{self.today_str}.xlsx")

        try:
            writer = pd.ExcelWriter(report_path, engine='xlsxwriter')
            workbook = writer.book

            # 定义格式
            header_format = workbook.add_format(
                {'bold': True, 'text_wrap': True, 'valign': 'top', 'fg_color': '#D7E4BC', 'border': 1})
            currency_format = workbook.add_format({'num_format': '#,##0.00'})
            code_format = workbook.add_format({'num_format': '@'})  # 文本格式

            for sheet_name, df in sheets_data.items():
                if df is None or df.empty:
                    print(f"  - 警告：工作表 '{sheet_name}' 数据为空，跳过创建。")
                    continue

                df.to_excel(writer, sheet_name=sheet_name, startrow=1, header=False, index=False)
                worksheet = writer.sheets[sheet_name]

                # 写入表头
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)

                # 设置列宽和格式
                for i, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).str.len().max(), len(col))
                    col_width = min(max_len + 2, 30)

                    # 修复：确保 '最新价' 列被正确识别为货币格式
                    if col == '最新价' or '价格' in col or '价' in col or '线' in col or '均线' in col:
                        worksheet.set_column(i, i, col_width, currency_format)
                    elif '代码' in col:
                        worksheet.set_column(i, i, 10, code_format)
                    else:
                        worksheet.set_column(i, i, col_width)

            writer.close()
            print(f"  - 报告已成功生成并保存到: {report_path}")

        except Exception as e:
            print(f"[FATAL] 致命错误：生成 Excel 报告失败。原因: {e}")
            raise

    def run(self):
        """主运行方法。"""
        print(f"股票分析程序启动 (Today: {self.today_str})")

        try:
            # 1. 获取所有原始数据
            raw_data = self._get_all_raw_data()

            # 预处理主力研报数据
            processed_main_report = raw_data.get('main_report_raw', pd.DataFrame())
            spot_df = raw_data.get('spot_data_all', pd.DataFrame())

            report_col = '机构投资评级(近六个月)-买入'
            if not processed_main_report.empty and report_col in processed_main_report.columns:
                # 确保过滤列是数字类型
                processed_main_report[report_col] = pd.to_numeric(processed_main_report[report_col],
                                                                  errors='coerce').fillna(0)

                # 过滤条件：买入评级 >= 1
                processed_main_report = processed_main_report[processed_main_report[report_col] > 1].copy()

                if processed_main_report.empty:
                    print("[WARN] 主力研报数据过滤后为空，回退至使用【A股实时行情】中的股票作为分析基础。")
                else:
                    print(">>> 主力研报数据已过滤，仅保留 '买入' 评级 > 1 的股票。")

            if not processed_main_report.empty and '股票代码' in processed_main_report.columns:
                # 使用过滤后的列表作为分析基础
                all_codes = set(processed_main_report['股票代码'].unique())
            elif not spot_df.empty and '股票代码' in spot_df.columns:
                print("[WARN] 主力研报数据为空或无效，回退至使用【A股实时行情】中的股票作为分析基础。")

                all_codes = set(spot_df['股票代码'].unique())
            else:
                print("[FATAL] 无法获取有效的股票代码列表，流程终止。")
                return
            top_ind_df = raw_data.get('top_industry_raw', pd.DataFrame())
            if not top_ind_df.empty:
                all_codes.update(top_ind_df['股票代码'].astype(str).unique())

            filtered_codes_list = list(all_codes)

            if not filtered_codes_list:
                print("未找到任何有效的股票代码，流程终止。")
                return

            # 2. 历史数据获取和技术指标计算
            hist_df_all = self._fetch_hist_data_parallel(filtered_codes_list, days=90)

            # 3. 技术指标信号提取
            ta_signals = self._process_ta_signals(filtered_codes_list, hist_df_all, spot_df)
            print(">>> 股票历史数据和技术指标分析完成。")

            industry_info_df = self._fetch_individual_industry_parallel(filtered_codes_list)

            # 4. 保存技术指标信号到本地 TXT 文件
            self._save_ta_signals_to_txt(ta_signals)

            universe_codes_set = set(filtered_codes_list)

            def filter_df_by_universe(df, universe_set):
                if df is None or df.empty or '股票代码' not in df.columns:
                    return pd.DataFrame()
                df['股票代码'] = df['股票代码'].astype(str)
                return df[df['股票代码'].isin(universe_set)].copy()

            # A. 均线突破数据处理
            processed_xstp_df = self._process_xstp_and_filter(raw_data, spot_df)
            processed_xstp_df = filter_df_by_universe(processed_xstp_df, universe_codes_set)

            # B.
            # 过滤其他每日排名数据
            raw_data['market_fund_flow_raw'] = filter_df_by_universe(raw_data['market_fund_flow_raw'],
                                                                     universe_codes_set)
            raw_data['strong_stocks_raw'] = filter_df_by_universe(raw_data['strong_stocks_raw'], universe_codes_set)
            raw_data['consecutive_rise_raw'] = filter_df_by_universe(raw_data['consecutive_rise_raw'],
                                                                     universe_codes_set)
            raw_data['ljqs_raw'] = filter_df_by_universe(raw_data['ljqs_raw'], universe_codes_set)
            raw_data['cxfl_raw'] = filter_df_by_universe(raw_data['cxfl_raw'], universe_codes_set)

            # 5. 合并所有数据源和信号
            processed_data = {
                **raw_data,
                **ta_signals,  # 技术指标信号
                'processed_xstp_df': processed_xstp_df,
                'processed_main_report': processed_main_report,
                'individual_industry': industry_info_df
            }

            consolidated_report = self._consolidate_data(processed_data)

            # 6. 准备报告数据
            sheets_data = {
                '数据汇总': consolidated_report,  # 替换为数据汇总表
                '主力研报筛选': processed_data['processed_main_report'],
                '均线多头排列': processed_xstp_df,  # 使用处理后且过滤后的数据
                '实时行情': spot_df,  # 使用清洗后的数据
                '5日市场资金流向': raw_data['market_fund_flow_raw'],
                '10日市场资金流向': raw_data['market_fund_flow_raw_10'],
                '20日市场资金流向': raw_data['market_fund_flow_raw_20'],
                '强势股池': raw_data['strong_stocks_raw'],
                '连续上涨': raw_data['consecutive_rise_raw'],
                '量价齐升': raw_data['ljqs_raw'],
                '持续放量': raw_data['cxfl_raw'],
                'MACD_12269金叉': ta_signals.get('MACD_12269', pd.DataFrame()),
                'MACD_6135金叉': ta_signals.get('MACD_6135', pd.DataFrame()),
                'MACD_DIF_动能状态': ta_signals.get('MACD_DIF_MOMENTUM', pd.DataFrame()),  # <== 新增这一行
                'KDJ超卖金叉': ta_signals.get('KDJ', pd.DataFrame()),
                'CCI专业状态': ta_signals.get('CCI', pd.DataFrame()),  # **改进点：使用专业状态**
                'RSI超卖': ta_signals.get('RSI', pd.DataFrame()),
                'BOLL低波': ta_signals.get('BOLL', pd.DataFrame()),
                '前十板块成分股': raw_data['top_industry_cons_df'],
            }

            # 7. 生成报告
            self._generate_report(sheets_data)

        except Exception as e:
            print(f"\n[FATAL] 致命错误：数据分析流程意外终止。原因: {e}")
            raise

        finally:
            end_time = time.time()
            print(f"\n>>> 流程结束。总耗时: {timedelta(seconds=end_time - self.start_time)}")


if __name__ == "__main__":
    analyzer = StockAnalyzer()
    analyzer.run()
