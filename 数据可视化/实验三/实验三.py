import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel('某公司销售数据.xlsx')

print("数据概览:")
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print("\n前5行数据:")
print(df.head())

# 数据预处理
df['订单日期'] = pd.to_datetime(df['订单日期'])
df['运送日期'] = pd.to_datetime(df['运送日期'])
df['送货天数'] = (df['运送日期'] - df['订单日期']).dt.days


# 1. 甘特图观察订单送货时间
def create_gantt_chart():
    print("\n1. 创建甘特图观察订单送货时间...")

    # 选择前30个不同的订单作为示例，按订单日期排序
    sample_df = df.drop_duplicates('订单号').head(30).copy()
    sample_df = sample_df.sort_values('订单日期')

    # 创建甘特图
    fig = go.Figure()

    # 为不同订单等级设置颜色
    colors = {'低级': '#FF6B6B', '中级': '#4ECDC4', '高级': '#45B7D1', '其它': '#96CEB4'}

    y_positions = []
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        task_name = f"订单{row['订单号']}-{row['顾客姓名'][:3]}-{row['产品类别']}"
        y_positions.append(task_name)

                # 添加甘特条
        fig.add_trace(go.Bar(
            x=[row['送货天数']],
            y=[task_name],
            orientation='h',
            base=row['订单日期'],
            name=row['订单等级'],
            marker_color=colors.get(row['订单等级'], '#95A5A6'),
            text=f"{row['送货天数']}天",
            textposition='inside',  # 修改为有效值
            showlegend=False if i > 0 and row['订单等级'] in [r['订单等级'] for _, r in
                                                               sample_df.iloc[:i].iterrows()] else True,
            hovertemplate=f"<b>{task_name}</b><br>" +
                         f"订单日期: {row['订单日期'].strftime('%Y-%m-%d')}<br>" +
                         f"运送日期: {row['运送日期'].strftime('%Y-%m-%d')}<br>" +
                         f"送货天数: {row['送货天数']}天<br>" +
                         f"订单等级: {row['订单等级']}<br>" +
                         f"销售额: ¥{row['销售额']:,.2f}<extra></extra>"
        ))

    fig.update_layout(
        title={
            'text': '订单送货时间甘特图',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='时间',
        yaxis_title='订单信息',
        height=800,
        font=dict(size=10),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # 设置x轴为时间格式
    fig.update_xaxes(type='date')

    fig.show()
    fig.write_html('甘特图_订单送货时间.html')


# 2. 标靶图绘制实际销售和对应计划
def create_bullet_chart():
    print("\n2. 创建标靶图分析销售情况...")

    # 按产品类别汇总销售数据
    category_sales = df.groupby('产品类别')['销售额'].sum().reset_index()

    # 假设计划销售额为实际销售额的1.2倍（可根据实际情况调整）
    category_sales['计划销售额'] = category_sales['销售额'] * 1.2
    category_sales['达成率'] = (category_sales['销售额'] / category_sales['计划销售额'] * 100).round(1)

    fig = go.Figure()

    for i, row in category_sales.iterrows():
        # 添加计划值（背景条）
        fig.add_trace(go.Bar(
            y=[row['产品类别']],
            x=[row['计划销售额']],
            orientation='h',
            marker=dict(color='lightgray'),
            name='计划销售额' if i == 0 else '',
            showlegend=True if i == 0 else False,
            text=f"计划: {row['计划销售额']:,.0f}",
            textposition='inside'
        ))

        # 添加实际值（前景条）
        fig.add_trace(go.Bar(
            y=[row['产品类别']],
            x=[row['销售额']],
            orientation='h',
            marker=dict(color='steelblue'),
            name='实际销售额' if i == 0 else '',
            showlegend=True if i == 0 else False,
            text=f"实际: {row['销售额']:,.0f} ({row['达成率']}%)",
            textposition='inside'
        ))

    fig.update_layout(
        title='各产品类别销售额标靶图',
        xaxis_title='销售额',
        yaxis_title='产品类别',
        barmode='overlay',
        height=400,
        font=dict(size=12)
    )

    fig.show()
    fig.write_html('标靶图_销售分析.html')


# 3. 瀑布图分析不同产品类别净利润
def create_waterfall_chart():
    print("\n3. 创建瀑布图分析产品类别净利润...")

    # 按产品类别计算净利润
    category_profit = df.groupby('产品类别')['利润额'].sum().reset_index()
    category_profit = category_profit.sort_values('利润额', ascending=False)

    # 计算累计值
    cumulative = [0]
    for profit in category_profit['利润额']:
        cumulative.append(cumulative[-1] + profit)

    fig = go.Figure()

    # 添加起始点
    fig.add_trace(go.Waterfall(
        name="净利润瀑布图",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(category_profit) + ["total"],
        x=["起始"] + category_profit['产品类别'].tolist() + ["总计"],
        y=[0] + category_profit['利润额'].tolist() + [category_profit['利润额'].sum()],
        text=[f"{val:,.0f}" for val in [0] + category_profit['利润额'].tolist() + [category_profit['利润额'].sum()]],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="不同产品类别净利润瀑布图",
        xaxis_title="产品类别",
        yaxis_title="利润额",
        height=500,
        font=dict(size=12)
    )

    fig.show()
    fig.write_html('瀑布图_产品利润分析.html')


# 4. 直方图分析订单的利润分布情况
def create_histogram():
    print("\n4. 创建直方图分析订单利润分布...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 整体利润分布
    axes[0, 0].hist(df['利润额'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('整体订单利润分布')
    axes[0, 0].set_xlabel('利润额')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].axvline(df['利润额'].mean(), color='red', linestyle='--', label=f'平均值: {df["利润额"].mean():.2f}')
    axes[0, 0].legend()

    # 按产品类别分组的利润分布
    for i, category in enumerate(df['产品类别'].unique()):
        if i < 3:  # 只显示前3个类别
            row = (i + 1) // 2
            col = (i + 1) % 2
            category_data = df[df['产品类别'] == category]['利润额']
            axes[row, col].hist(category_data, bins=30, alpha=0.7, edgecolor='black')
            axes[row, col].set_title(f'{category} 利润分布')
            axes[row, col].set_xlabel('利润额')
            axes[row, col].set_ylabel('频次')
            axes[row, col].axvline(category_data.mean(), color='red', linestyle='--',
                                   label=f'平均值: {category_data.mean():.2f}')
            axes[row, col].legend()

    plt.tight_layout()
    plt.savefig('直方图_利润分布分析.png', dpi=300, bbox_inches='tight')
    plt.show()


# 5. 气泡图观察产品销售额和利润额
def create_bubble_chart():
    print("\n5. 创建气泡图观察产品销售额和利润额...")

    # 按产品子类别汇总数据
    product_summary = df.groupby('产品子类别').agg({
        '销售额': 'sum',
        '利润额': 'sum',
        '订单数量': 'sum',
        '产品类别': 'first'
    }).reset_index()

    # 计算利润率
    product_summary['利润率'] = (product_summary['利润额'] / product_summary['销售额'] * 100).round(2)

    # 选择前20个销售额最高的产品子类别
    product_summary = product_summary.nlargest(20, '销售额')

    # 创建气泡图
    fig = px.scatter(product_summary,
                     x='销售额',
                     y='利润额',
                     size='订单数量',
                     color='利润率',
                     color_continuous_scale='Viridis',
                     hover_name='产品子类别',
                     title='产品销售额与利润额关系气泡图（Top 20）',
                     labels={
                         '销售额': '销售额 (¥)',
                         '利润额': '利润额 (¥)',
                         '利润率': '利润率 (%)',
                         '订单数量': '订单数量'
                     },
                     size_max=80)

    # 自定义悬停信息
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                      "销售额: ¥%{x:,.0f}<br>" +
                      "利润额: ¥%{y:,.0f}<br>" +
                      "订单数量: %{marker.size}<br>" +
                      "利润率: %{marker.color:.1f}%<br>" +
                      "<extra></extra>"
    )

    # 添加趋势线
    fig.add_trace(go.Scatter(
        x=product_summary['销售额'],
        y=np.poly1d(np.polyfit(product_summary['销售额'], product_summary['利润额'], 1))(product_summary['销售额']),
        mode='lines',
        name='趋势线',
        line=dict(color='red', dash='dash', width=2),
        hoverinfo='skip'
    ))

    fig.update_layout(
        title={
            'text': '产品销售额与利润额关系气泡图（Top 20）',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=700,
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title='销售额 (¥)',
            gridcolor='lightgray',
            tickformat=',.0f'
        ),
        yaxis=dict(
            title='利润额 (¥)',
            gridcolor='lightgray',
            tickformat=',.0f'
        ),
        coloraxis_colorbar=dict(
            title="利润率 (%)"
        )
    )

    fig.show()
    fig.write_html('气泡图_产品销售利润分析.html')


# 6. 树状图观察产品销售额和利润额
def create_treemap():
    print("\n6. 创建树状图观察产品销售额和利润额...")

    # 按产品类别和子类别汇总
    tree_data = df.groupby(['产品类别', '产品子类别']).agg({
        '销售额': 'sum',
        '利润额': 'sum',
        '订单号': 'count'
    }).reset_index()

    # 计算利润率
    tree_data['利润率'] = (tree_data['利润额'] / tree_data['销售额'] * 100).round(2)

    # 过滤掉利润额为负数或零的数据
    tree_data_positive = tree_data[tree_data['利润额'] > 0].copy()

        # 创建销售额树状图（使用淡色方案）
    fig1 = px.treemap(tree_data,
                      path=['产品类别', '产品子类别'],
                      values='销售额',
                      title='产品销售额树状图',
                      color='利润率',
                      color_continuous_scale='peach',  # 使用有效的淡色方案
                      hover_data=['订单号'])

    fig1.update_traces(
        textinfo="label+value+percent entry",
        hovertemplate="<b>%{label}</b><br>" +
                      "销售额: ¥%{value:,.0f}<br>" +
                      "利润率: %{color:.1f}%<br>" +
                      "订单数: %{customdata[0]}<br>" +
                      "<extra></extra>"
    )

    fig1.update_layout(
        title={
            'text': '产品销售额树状图',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=700,
        font=dict(size=11),
        coloraxis_colorbar=dict(title="利润率 (%)")
    )

    fig1.show()
    fig1.write_html('树状图_产品销售额.html')

        # 创建利润额树状图（修复显示问题）
    fig2 = px.treemap(tree_data_positive,
                      path=['产品类别', '产品子类别'],
                      values='利润额',
                      title='产品利润额树状图',
                      color='销售额',
                      color_continuous_scale='blues',  # 使用蓝色系（小写）
                      hover_data=['订单号'])

    fig2.update_traces(
        textinfo="label+value+percent entry",
        hovertemplate="<b>%{label}</b><br>" +
                      "利润额: ¥%{value:,.0f}<br>" +
                      "销售额: ¥%{color:,.0f}<br>" +
                      "订单数: %{customdata[0]}<br>" +
                      "<extra></extra>"
    )

    fig2.update_layout(
        title={
            'text': '产品利润额树状图',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=700,
        font=dict(size=11),
        coloraxis_colorbar=dict(title="销售额 (¥)")
    )

    fig2.show()
    fig2.write_html('树状图_产品利润额.html')

    # 输出统计信息
    print(f"销售额树状图: 包含 {len(tree_data)} 个产品子类别")
    print(f"利润额树状图: 包含 {len(tree_data_positive)} 个盈利产品子类别")
    print(f"负利润产品子类别数量: {len(tree_data) - len(tree_data_positive)}")


# 数据统计摘要
def print_summary():
    print("\n=== 数据分析摘要 ===")
    print(f"总订单数: {len(df):,}")
    print(f"总销售额: {df['销售额'].sum():,.2f}")
    print(f"总利润额: {df['利润额'].sum():,.2f}")
    print(f"平均利润率: {(df['利润额'].sum() / df['销售额'].sum() * 100):.2f}%")
    print(f"平均送货天数: {df['送货天数'].mean():.1f} 天")

    print("\n按产品类别统计:")
    category_stats = df.groupby('产品类别').agg({
        '销售额': 'sum',
        '利润额': 'sum',
        '订单号': 'count'
    }).round(2)
    category_stats['利润率%'] = (category_stats['利润额'] / category_stats['销售额'] * 100).round(2)
    print(category_stats)

    print("\n按订单等级统计:")
    level_stats = df.groupby('订单等级').agg({
        '销售额': 'sum',
        '利润额': 'sum',
        '订单号': 'count',
        '送货天数': 'mean'
    }).round(2)
    print(level_stats)


# 主函数
def main():
    print("开始数据分析...")
    print("=" * 50)

    # 打印数据摘要
    print_summary()

    # 创建各种图表
    create_gantt_chart()
    create_bullet_chart()
    create_waterfall_chart()
    create_histogram()
    create_bubble_chart()
    create_treemap()

    print("\n所有图表已生成完成！")
    print("生成的文件:")
    print("- 甘特图_订单送货时间.html")
    print("- 标靶图_销售分析.html")
    print("- 瀑布图_产品利润分析.html")
    print("- 直方图_利润分布分析.png")
    print("- 气泡图_产品销售利润分析.html")
    print("- 树状图_产品销售额.html")
    print("- 树状图_产品利润额.html")


if __name__ == "__main__":
    main()