import os
from datetime import datetime
import pandas as pd
import akshare as ak
import time
import numpy as np

class IndustryFlowAnalyzer:

    def __init__(self, config):
        self.config = config
        self.today_str = datetime.now().strftime('%Y%m%d')
        self.cache_filename = f"行业权重趋势_{self.today_str}.txt"
        self.cache_path = os.path.join(self.config.TEMP_DATA_DIRECTORY, self.cache_filename)

    def _normalize_amount(self, val):
        if pd.isna(val): return 0.0
        s = str(val).strip()
        try:
            if '亿' in s: return float(s.replace('亿', ''))
            elif '万' in s: return float(s.replace('万', '')) / 10000.0
            else: return float(s)
        except: return 0.0

    def _clean_pct_string(self, val):
        if pd.isna(val): return 0.0
        if isinstance(val, (int, float)): return float(val)
        try: return float(str(val).replace('%', '').strip())
        except: return 0.0

    def _fetch_and_clean(self, period_name):
        """抓取同花顺行业资金流接口"""
        try:
            df = ak.stock_fund_flow_industry(symbol=period_name)
            if df is None or df.empty: return pd.DataFrame()
            df = df.rename(columns={'行业': '行业名称'})
            # 清洗百分比和金额
            pct_cols = ['行业-涨跌幅', '阶段涨跌幅', '领涨股-涨跌幅']
            for col in pct_cols:
                if col in df.columns:
                    df[col] = df[col].apply(self._clean_pct_string)
            money_cols = ['流入资金', '流出资金', '净额']
            for col in money_cols:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].apply(self._normalize_amount)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            return df
        except Exception as e:
            print(f"[WARN] 周期 {period_name} 数据抓取失败: {e}")
            return pd.DataFrame()

    def _fetch_market_turnover(self):
        """抓取东方财富行业接口以补全【换手率】"""
        try:
            df = ak.stock_board_industry_name_em()
            # 统一列名以备 merge
            df = df[['板块名称', '换手率']]
            df.columns = ['行业名称', '换手率']
            return df
        except Exception as e:
            print(f"[WARN] 补全换手率数据失败: {e}")
            return pd.DataFrame()

    def _fetch_big_deal_logic(self):
        """抓取大单追踪以增强潜入识别"""
        try:
            df = ak.stock_fund_flow_big_deal()
            if df.empty: return set()
            # 只统计买入的大单
            buy_stocks = set(df[df['大单性质'].str.contains('买入', na=False)]['股票简称'].tolist())
            return buy_stocks
        except:
            return set()

    def run_analysis(self) -> pd.DataFrame:
        if os.path.exists(self.cache_path):
            print(f">>> 发现本地缓存：{self.cache_filename}，正在加载...")
            try:
                return pd.read_csv(self.cache_path, sep='\t', encoding='utf-8')
            except Exception as e:
                print(f"[WARN] 缓存加载失败: {e}")

        print(f"\n>>> 开始从接口获取行业趋势及筹码分布...")
        period_map = {"即时": "now", "3日排行": "3d", "5日排行": "5d", "10日排行": "10d", "20日排行": "20d"}
        dfs = {}
        for p_name, p_key in period_map.items():
            df = self._fetch_and_clean(p_name)
            if not df.empty: dfs[p_key] = df
            time.sleep(1.0)

        if "now" not in dfs: return pd.DataFrame()

        # 1. 基础资金数据对齐
        main = dfs['now'][['行业名称', '行业指数', '行业-涨跌幅', '净额', '流入资金', '领涨股', '领涨股-涨跌幅']].copy()
        main.rename(columns={'净额': '净额_now', '行业-涨跌幅': '涨幅_now'}, inplace=True)

        for p in ['3d', '5d', '10d', '20d']:
            if p in dfs:
                tmp = dfs[p][['行业名称', '净额', '阶段涨跌幅']]
                tmp.columns = ['行业名称', f'净额_{p}', f'涨幅_{p}']
                main = pd.merge(main, tmp, on='行业名称', how='left')

        # 2. 补全缺失的【换手率】和【大单】维度
        turnover_df = self._fetch_market_turnover()
        if not turnover_df.empty:
            main = pd.merge(main, turnover_df, on='行业名称', how='left')
        else:
            main['换手率'] = 0.0 # 兜底逻辑

        big_deal_stocks = self._fetch_big_deal_logic()
        main['大单印证'] = main['领涨股'].apply(lambda x: '确认' if x in big_deal_stocks else '无')

        # 3. 核心计算 (处理缺失值)
        money_rank_cols = ['净额_3d', '净额_5d', '净额_10d', '净额_20d']
        for col in money_rank_cols: main[col] = main[col].fillna(0.0)
        
        main['资金分'] = (main['净额_3d'] * 0.4 + main['净额_5d'] * 0.3 + main['净额_10d'] * 0.2 + main['净额_20d'] * 0.1).rank(pct=True) * 100
        main['价格分'] = (main['涨幅_now'].rank(pct=True) * 100) # 以即时强度为主
        main['换手分'] = main['换手率'].rank(pct=True, ascending=False) * 100 # 换手越低（缩量）得分越高
        main['趋势得分'] = (main['资金分'] * 0.5 + main['价格分'] * 0.5).round(2)

        # 4. 潜入识别信号逻辑
        # 黄金坑：钱在进（资金分>75），价没起（价格分<50），散户没动（换手分>60, 即低换手率）
        is_submerged = (main['资金分'] > 75) & (main['价格分'] < 50) & (main['换手分'] > 60)
        # 异动点：价格开始抬头，且有大单背书
        is_shaking = (main['趋势得分'].between(50, 80)) & (main['大单印证'] == '确认')

        conds = [
            (main['趋势得分'] > 85),
            (main['趋势得分'] < 25),
            (is_submerged),
            (is_shaking)
        ]
        main['行业信号'] = np.select(conds, ['资金主攻', '退潮预警', '黄金坑潜入', '低位强异动'], default='观望区间')

        result = main.sort_values('趋势得分', ascending=False)

        try:
            if not os.path.exists(self.config.TEMP_DATA_DIRECTORY): os.makedirs(self.config.TEMP_DATA_DIRECTORY)
            result.to_csv(self.cache_path, sep='\t', index=False, encoding='utf-8')
            print(f">>> 深度分析完成，结果存至: {self.cache_filename}")
        except Exception as e:
            print(f"[WARN] 保存失败: {e}")

        return result
