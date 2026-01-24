from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Any
import time
import pd


def _normalize_fund_data(df):
    """
    标准化资金数据：将所有资金相关的字符串列（包含'亿'、'万'单位）统一转换为数值型（单位：万元）。
    """
    if df is None or df.empty:
        return df

    # 定义关键词，自动识别需要转换单位的列
    zijin_keywords = ['资金', '流向', '净流入', '净额', '成交额']
    target_cols = [col for col in df.columns if any(k in col for k in zijin_keywords)]

    for col in target_cols:
        if df[col].dtype == object:  # 仅处理字符串类型
            def convert_to_wan(val):
                if val is None or str(val).strip() in ['', '-']:
                    return 0.0
                val_str = str(val).strip()
                try:
                    if '亿' in val_str:
                        return float(val_str.replace('亿', '')) * 10000
                    elif '万' in val_str:
                        return float(val_str.replace('万', ''))
                    else:
                        return float(val_str)
                except:
                    return 0.0
            df[col] = df[col].apply(convert_to_wan)
    return df


def run_with_thread_pool(
        items: Iterable[Any],
        worker_func: Callable[[Any], Any],
        max_workers: int = 10,
        desc: str = "任务"
) -> List[Any]:
    """
    通用的多线程执行器。

    :param items: 需要处理的数据列表 (如股票代码列表)
    :param worker_func: 处理单个数据的函数 (输入一个item，返回结果)
    :param max_workers: 最大线程数
    :param desc: 任务描述，用于日志打印
    :return: 包含所有成功结果的列表 (过滤掉 None)
    """
    results = []
    total = len(list(items))  # 注意：如果items是生成器，这里会消耗掉，建议传list

    print(f"\n>>> 开始并发执行: {desc} (数量: {total}, 线程: {max_workers})...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_item = {executor.submit(worker_func, item): item for item in items}

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                res = future.result()
                if res is not None:
                    # 如果结果是 DataFrame 且为空，视具体情况决定是否添加
                    # 这里只要不是 None 就添加，由调用方后续处理 (如 concat)
                    results.append(res)
            except Exception as e:
                print(f"[ERROR] 处理 {item} 时发生异常: {e}")

    print(f">>> {desc} 执行完毕，成功获取 {len(results)} 条结果。")
    return results
