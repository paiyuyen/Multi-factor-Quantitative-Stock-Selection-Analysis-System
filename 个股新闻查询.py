import akshare as ak
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta  # 导入日期时间模块

def get_period_selection() -> int:
    """
    提示用户选择新闻查询的周期 (30 天或 60 天)。
    """
    while True:
        print("\n请选择您要查询的新闻周期：")
        print("  输入 1: 最近 30 天")
        print("  输入 2: 最近 60 天")
        choice = input("请输入您的选择 (1 或 2): ").strip()

        if choice == '1':
            return 30
        elif choice == '2':
            return 60
        else:
            print(f"输入 '{choice}' 无效。请重新输入 1 或 2。")

def query_stock_news():
    """
    查询指定股票代码的最新新闻资讯，并显示关键信息。
    根据用户选择的周期进行过滤和排序，并将结果保存到 Excel 文件。
    （新闻内容不截断，完整保存）
    """
    print("--- 东方财富个股新闻查询工具 ---")

    # 循环直到用户输入有效的股票代码
    while True:
        stock_code = input("请输入您要查询的股票代码 (例如: 603777)，或输入 'exit' 退出: ").strip()

        if stock_code.lower() == 'exit':
            print("程序已退出。")
            return

        # 简单的验证，确保输入是数字且长度为 6 位
        if stock_code.isdigit() and len(stock_code) == 6:
            break
        else:
            print(f"输入 '{stock_code}' 无效。股票代码通常是 6 位数字。请重新输入。")

    # 第二个交互问题：选择查询周期
    days_to_query = get_period_selection()
    
    print(f"\n正在查询股票代码 {stock_code} 在过去 {days_to_query} 天内的最新新闻资讯...")

    try:
        # 调用 akshare 接口获取个股新闻
        # 接口: ak.stock_news_em(symbol="股票代码")
        news_df: Optional[pd.DataFrame] = ak.stock_news_em(symbol=stock_code)

        if news_df is None or news_df.empty:
            print(f"\n[提示] 未能获取到股票 {stock_code} 的新闻数据，可能是今日无相关新闻或接口查询失败。")
            return

        # =======================================================
        # 逻辑：只保留近 N 天的新闻 & 倒序排序
        # =======================================================
        if '发布时间' in news_df.columns:
            try:
                # 1. 计算 N 天前的日期
                cutoff_date = datetime.now() - timedelta(days=days_to_query)
                
                # 2. 将 '发布时间' 转换为 datetime 对象
                news_df['发布时间_dt'] = pd.to_datetime(news_df['发布时间'], errors='coerce')
                
                # 3. 过滤数据：只保留发布时间在 cutoff 日期之后的新闻，并移除 NaT
                news_df = news_df[
                    (news_df['发布时间_dt'].notna()) & 
                    (news_df['发布时间_dt'] >= cutoff_date)
                ].copy()

                # 4. 倒序排序：根据发布时间的 datetime 对象进行排序 (最新的在前)
                news_df = news_df.sort_values(by='发布时间_dt', ascending=False)
                
                # 5. 删除临时 datetime 列
                news_df = news_df.drop(columns=['发布时间_dt'])
                
            except Exception as e:
                # 如果日期处理出现任何问题，打印警告并继续使用全部数据（未排序）
                print(f"[警告] 对 '发布时间' 进行 {days_to_query} 天过滤和排序时失败，将显示所有新闻且可能未排序。错误详情: {e}")
        
        # 检查过滤后是否还有数据
        if news_df.empty:
            print(f"\n[提示] 股票 {stock_code} 在过去 {days_to_query} 天内没有新的新闻资讯。")
            return
        
        # =======================================================
        # 筛选和格式化要展示的列
        # =======================================================

        # 检查并标准化列名
        required_cols = ["关键词", "新闻标题", "新闻内容", "发布时间"]
        
        # 确保 DataFrame 包含所有必需的列
        # 仅保留存在的且需要的列
        display_df = news_df[[col for col in required_cols if col in news_df.columns]].copy()
        
        # 格式化发布时间
        if '发布时间' in display_df.columns:
            try:
                # 尝试将时间列转换为 datetime 对象，然后格式化为统一的字符串格式
                display_df['发布时间'] = pd.to_datetime(display_df['发布时间']).dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass # 保持原样如果转换失败
        
        # ** 新闻内容不截断，完整保存到 Excel **
        
        # =======================================================
        # 逻辑：保存到 Excel 文件
        # =======================================================
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"个股新闻_{stock_code}_{timestamp}.xlsx"

        try:
            # 使用 Pandas 的 to_excel 方法保存到 Excel
            # index=False 避免将 DataFrame 索引写入 Excel
            display_df.to_excel(filename, index=False, sheet_name=stock_code)
            
            # 更新提示信息，反映用户选择的周期
            print(f"\n成功获取 {stock_code} 在过去 {days_to_query} 天内的 {len(display_df)} 条新闻。")
            print(f"结果已保存到 Excel 文件: {filename}")
            print("您可以在 Excel 中打开此文件，查看完整的新闻内容。")

        except Exception as e:
            print(f"\n[错误] 保存 Excel 文件失败: {e}")
            # 文件保存失败，退回到控制台输出作为提示（但内容仍然完整，可能显示较乱）
            print("\n[警告] 文件保存失败，以下是完整数据控制台输出（可能格式错乱，建议解决文件保存问题后重试）:")
            print(display_df.to_string(index=False)) 

        print("\n查询完成。")

    except Exception as e:
        print(f"\n[错误] 查询过程中发生异常，请检查股票代码是否正确或网络连接: {e}")

if __name__ == "__main__":
    query_stock_news()
