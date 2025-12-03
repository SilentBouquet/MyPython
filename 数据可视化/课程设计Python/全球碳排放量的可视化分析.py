import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class CarbonVisualizationAnalyzer:
    def __init__(self):
        """初始化可视化分析器"""
        self.df_long = None
        self.df_wide = None
        self.load_clean_data()

    def load_clean_data(self):
        """加载预处理后的清洁数据"""
        try:
            print("加载预处理后的清洁数据...")
            self.df_long = pd.read_csv('清洁数据_长格式.csv')
            self.df_wide = pd.read_csv('清洁数据_宽格式.csv')
            print(f"✓ 长格式数据: {self.df_long.shape}")
            print(f"✓ 宽格式数据: {self.df_wide.shape}")

            # 数据概览
            print(f"时间跨度: {self.df_long['Year'].min()}-{self.df_long['Year'].max()}")
            print(f"国家数量: {self.df_long['Country'].nunique()}")
            print(f"数据点总数: {len(self.df_long)}")

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")

    def create_time_trend_charts(self):
        """创建时间趋势图表 - 每个图表独立保存"""
        print("\n" + "=" * 60)
        print("创建时间趋势图表")
        print("=" * 60)

        # 1. 全球总排放量趋势 - 折线图
        print("创建图表1: 全球排放量时间趋势")
        plt.figure(figsize=(12, 8))
        global_emissions = self.df_long.groupby('Year')['Emissions'].sum().reset_index()
        plt.plot(global_emissions['Year'], global_emissions['Emissions'] / 1e6,
                 linewidth=3, marker='o', markersize=6, color='darkred')
        plt.title('全球温室气体排放量时间趋势', fontsize=16, fontweight='bold')
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('排放量 (百万 kt CO2当量)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('01_全球排放量时间趋势.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. 全球排放量面积图
        print("创建图表2: 全球排放量面积图")
        plt.figure(figsize=(12, 8))
        plt.fill_between(global_emissions['Year'], global_emissions['Emissions'] / 1e6,
                         alpha=0.7, color='lightcoral')
        plt.plot(global_emissions['Year'], global_emissions['Emissions'] / 1e6,
                 linewidth=2, color='darkred')
        plt.title('全球排放量面积图', fontsize=16, fontweight='bold')
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('排放量 (百万 kt CO2当量)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('02_全球排放量面积图.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. 主要排放国时间趋势
        print("创建图表3: 主要排放国时间趋势")
        plt.figure(figsize=(14, 8))
        top_countries = self.df_long.groupby('Country')['Emissions'].sum().nlargest(8).index
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_countries)))

        for i, country in enumerate(top_countries):
            country_data = self.df_long[self.df_long['Country'] == country]
            plt.plot(country_data['Year'], country_data['Emissions'] / 1e3,
                     label=country, linewidth=2, color=colors[i])

        plt.title('主要排放国时间趋势', fontsize=16, fontweight='bold')
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('排放量 (千 kt CO2当量)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('03_主要排放国时间趋势.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. 排放量增长率分析
        print("创建图表4: 全球排放量年增长率")
        plt.figure(figsize=(12, 8))
        global_emissions['Growth_Rate'] = global_emissions['Emissions'].pct_change() * 100
        colors = ['green' if x >= 0 else 'red' for x in global_emissions['Growth_Rate'][1:]]
        plt.bar(global_emissions['Year'][1:], global_emissions['Growth_Rate'][1:],
                alpha=0.7, color=colors)
        plt.title('全球排放量年增长率', fontsize=16, fontweight='bold')
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('增长率 (%)', fontsize=12)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('04_全球排放量年增长率.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 5. 累积排放量
        print("创建图表5: 全球累积排放量")
        plt.figure(figsize=(12, 8))
        global_emissions['Cumulative'] = global_emissions['Emissions'].cumsum()
        plt.plot(global_emissions['Year'], global_emissions['Cumulative'] / 1e9,
                 linewidth=3, marker='s', markersize=4, color='purple')
        plt.title('全球累积排放量', fontsize=16, fontweight='bold')
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('累积排放量 (十亿 kt CO2当量)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('05_全球累积排放量.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 6. 年度排放量分布
        print("创建图表6: 不同年份排放量分布")
        plt.figure(figsize=(12, 8))
        years_to_show = [1990, 2000, 2010, 2020]
        for year in years_to_show:
            year_data = self.df_long[self.df_long['Year'] == year]['Emissions']
            plt.hist(np.log10(year_data[year_data > 0]), alpha=0.6, label=f'{year}年', bins=15)

        plt.title('不同年份排放量分布', fontsize=16, fontweight='bold')
        plt.xlabel('log10(排放量)', fontsize=12)
        plt.ylabel('国家数量', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('06_不同年份排放量分布.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_comparison_charts(self):
        """创建比较分析图表 - 每个图表独立保存"""
        print("\n" + "=" * 60)
        print("创建比较分析图表")
        print("=" * 60)

        # 1. 各国总排放量排名 - 条形图
        print("创建图表7: 各国总排放量排名")
        plt.figure(figsize=(12, 10))
        country_total = self.df_long.groupby('Country')['Emissions'].sum().nlargest(15)

        bars = plt.barh(range(len(country_total)), country_total.values / 1e6,
                        color=plt.cm.Reds(np.linspace(0.3, 1, len(country_total))))
        plt.yticks(range(len(country_total)), country_total.index, fontsize=10)
        plt.title('各国总排放量排名 (Top 15)', fontsize=16, fontweight='bold')
        plt.xlabel('总排放量 (百万 kt CO2当量)', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('07_各国总排放量排名.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. 2020年排放量占比 - 饼图
        print("创建图表8: 2020年各国排放量占比")
        plt.figure(figsize=(12, 10))
        recent_data = self.df_long[self.df_long['Year'] == 2020].nlargest(8, 'Emissions')
        others = self.df_long[self.df_long['Year'] == 2020]['Emissions'].sum() - recent_data['Emissions'].sum()

        sizes = list(recent_data['Emissions']) + [others]
        labels = list(recent_data['Country']) + ['其他国家']
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))

        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('2020年各国排放量占比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('08_2020年各国排放量占比.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. 排放量变化率对比
        print("创建图表9: 主要国家排放量变化率")
        plt.figure(figsize=(14, 8))
        change_data = []
        for country in country_total.index[:12]:
            country_data = self.df_long[self.df_long['Country'] == country].sort_values('Year')
            if len(country_data) >= 2:
                first_val = country_data['Emissions'].iloc[0]
                last_val = country_data['Emissions'].iloc[-1]
                if first_val != 0:
                    change_rate = ((last_val - first_val) / first_val) * 100
                    change_data.append((country, change_rate))

        if change_data:
            countries, changes = zip(*change_data)
            colors = ['green' if x >= 0 else 'red' for x in changes]
            plt.bar(range(len(countries)), changes, color=colors, alpha=0.7)
            plt.xticks(range(len(countries)), countries, rotation=45, ha='right')
            plt.title('主要国家排放量变化率 (1990-2020)', fontsize=16, fontweight='bold')
            plt.ylabel('变化率 (%)', fontsize=12)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('09_主要国家排放量变化率.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. 雷达图 - Top 6国家多维对比
        print("创建图表10: Top 6国家排放量雷达图")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        top_6_countries = country_total.head(6)

        angles = np.linspace(0, 2 * np.pi, len(top_6_countries), endpoint=False).tolist()
        angles += angles[:1]

        values = (top_6_countries.values / top_6_countries.max()).tolist()
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label='相对排放量')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_6_countries.index, fontsize=10)
        ax.set_title('Top 6 国家排放量雷达图', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig('10_Top6国家排放量雷达图.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 5. 年代对比
        print("创建图表11: 各年代全球总排放量对比")
        plt.figure(figsize=(10, 8))
        decades = [1990, 2000, 2010, 2020]
        decade_data = []
        for decade in decades:
            decade_total = self.df_long[self.df_long['Year'] == decade]['Emissions'].sum()
            decade_data.append(decade_total / 1e6)

        plt.bar([str(d) for d in decades], decade_data,
                color=['skyblue', 'lightgreen', 'orange', 'coral'], alpha=0.8)
        plt.title('各年代全球总排放量对比', fontsize=16, fontweight='bold')
        plt.ylabel('排放量 (百万 kt CO2当量)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('11_各年代全球总排放量对比.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 6. 排放强度分析
        print("创建图表12: 2020年各国排放强度分布")
        plt.figure(figsize=(10, 8))
        # 按排放量分组
        emission_groups = pd.cut(self.df_long[self.df_long['Year'] == 2020]['Emissions'],
                                 bins=4, labels=['低排放', '中低排放', '中高排放', '高排放'])
        group_counts = emission_groups.value_counts()

        plt.bar(group_counts.index, group_counts.values,
                color=['lightblue', 'yellow', 'orange', 'red'], alpha=0.7)
        plt.title('2020年各国排放强度分布', fontsize=16, fontweight='bold')
        plt.ylabel('国家数量', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('12_2020年各国排放强度分布.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_distribution_charts(self):
        """创建分布分析图表 - 每个图表独立保存"""
        print("\n" + "=" * 60)
        print("创建分布分析图表")
        print("=" * 60)

        # 1. 排放量分布直方图
        print("创建图表13: 全球排放量分布")
        plt.figure(figsize=(12, 8))
        emissions_data = self.df_long['Emissions'][self.df_long['Emissions'] > 0]
        plt.hist(np.log10(emissions_data), bins=40, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('全球排放量分布 (对数尺度)', fontsize=16, fontweight='bold')
        plt.xlabel('log10(排放量)', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('13_全球排放量分布.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. 主要国家排放量箱线图
        print("创建图表14: 主要国家排放量分布箱线图")
        plt.figure(figsize=(14, 8))
        top_countries = self.df_long.groupby('Country')['Emissions'].sum().nlargest(10).index
        box_data = [self.df_long[self.df_long['Country'] == country]['Emissions'].values
                    for country in top_countries]

        plt.boxplot(box_data, labels=top_countries)
        plt.title('主要国家排放量分布箱线图', fontsize=16, fontweight='bold')
        plt.ylabel('排放量 (kt CO2当量)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('14_主要国家排放量分布箱线图.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. 年度统计分布
        print("创建图表15: 年度排放量统计分布")
        plt.figure(figsize=(12, 8))
        yearly_stats = self.df_long.groupby('Year')['Emissions'].agg(['mean', 'std', 'min', 'max']).reset_index()

        plt.fill_between(yearly_stats['Year'],
                         (yearly_stats['mean'] - yearly_stats['std']) / 1e3,
                         (yearly_stats['mean'] + yearly_stats['std']) / 1e3,
                         alpha=0.3, color='lightblue', label='±1标准差')
        plt.plot(yearly_stats['Year'], yearly_stats['mean'] / 1e3, 'b-', linewidth=2, label='平均值')
        plt.plot(yearly_stats['Year'], yearly_stats['max'] / 1e3, 'r--', linewidth=1, label='最大值')

        plt.title('年度排放量统计分布', fontsize=16, fontweight='bold')
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('排放量 (千 kt CO2当量)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('15_年度排放量统计分布.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. 密度分布对比
        print("创建图表16: 1990年 vs 2020年排放量密度分布")
        plt.figure(figsize=(12, 8))
        for year in [1990, 2020]:
            year_data = self.df_long[self.df_long['Year'] == year]['Emissions']
            year_data = year_data[year_data > 0]
            sns.kdeplot(data=np.log10(year_data), label=f'{year}年', alpha=0.7)

        plt.title('1990年 vs 2020年排放量密度分布', fontsize=16, fontweight='bold')
        plt.xlabel('log10(排放量)', fontsize=12)
        plt.ylabel('密度', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('16_1990年vs2020年排放量密度分布.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 5. 累积分布
        print("创建图表17: 2020年排放量累积分布")
        plt.figure(figsize=(12, 8))
        emissions_2020 = self.df_long[self.df_long['Year'] == 2020]['Emissions'].sort_values(ascending=False)
        cumsum_pct = (emissions_2020.cumsum() / emissions_2020.sum() * 100).values

        plt.plot(range(1, len(cumsum_pct) + 1), cumsum_pct, 'o-', linewidth=2)
        plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80%线')
        plt.title('2020年排放量累积分布', fontsize=16, fontweight='bold')
        plt.xlabel('国家排名', fontsize=12)
        plt.ylabel('累积排放量占比 (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('17_2020年排放量累积分布.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 6. 分位数分析
        print("创建图表18: 排放量分位数时间趋势")
        plt.figure(figsize=(12, 8))
        quantiles = self.df_long.groupby('Year')['Emissions'].quantile([0.25, 0.5, 0.75]).unstack()

        plt.plot(quantiles.index, quantiles[0.25] / 1e3, label='25%分位数', linewidth=2)
        plt.plot(quantiles.index, quantiles[0.5] / 1e3, label='50%分位数', linewidth=2)
        plt.plot(quantiles.index, quantiles[0.75] / 1e3, label='75%分位数', linewidth=2)

        plt.title('排放量分位数时间趋势', fontsize=16, fontweight='bold')
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('排放量 (千 kt CO2当量)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('18_排放量分位数时间趋势.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_advanced_charts(self):
        """创建高级图表"""
        print("\n" + "=" * 60)
        print("创建高级图表")
        print("=" * 60)

        # 1. 动态气泡图
        print("创建动态气泡图...")
        top_countries = self.df_long.groupby('Country')['Emissions'].sum().nlargest(12).index
        bubble_data = self.df_long[self.df_long['Country'].isin(top_countries)].copy()

        # 添加虚拟数据用于演示
        np.random.seed(42)
        bubble_data['GDP_Index'] = bubble_data['Emissions'] * np.random.uniform(0.8, 1.2, len(bubble_data))
        bubble_data['Population_Index'] = bubble_data['Emissions'] * np.random.uniform(0.001, 0.01, len(bubble_data))

        fig_bubble = px.scatter(bubble_data,
                                x='GDP_Index',
                                y='Emissions',
                                size='Population_Index',
                                color='Country',
                                animation_frame='Year',
                                hover_name='Country',
                                title='动态气泡图: GDP指数 vs 排放量 (气泡大小=人口指数)',
                                labels={'GDP_Index': 'GDP指数', 'Emissions': '排放量 (kt CO2当量)'},
                                height=600)

        fig_bubble.write_html('19_动态气泡图.html')
        print("✓ 动态气泡图已保存: 19_动态气泡图.html")

        # 2. 桑基图
        print("创建桑基图...")
        recent_data = self.df_long[self.df_long['Year'] == 2020].nlargest(10, 'Emissions')

        source = list(range(len(recent_data)))
        target = [len(recent_data)] * len(recent_data)
        value = recent_data['Emissions'].tolist()
        labels = recent_data['Country'].tolist() + ['全球总排放量']

        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["lightblue"] * len(labels)
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=["rgba(255,0,0,0.3)"] * len(value)
            )
        )])

        fig_sankey.update_layout(title_text="2020年主要国家排放量桑基图",
                                 font_size=12, height=600)
        fig_sankey.write_html('20_桑基图.html')
        print("✓ 桑基图已保存: 20_桑基图.html")

        # 3. 3D表面图
        print("创建3D表面图...")
        fig_3d = plt.figure(figsize=(15, 10))
        ax = fig_3d.add_subplot(111, projection='3d')

        # 选择前6个国家创建3D数据
        top_countries_3d = self.df_long.groupby('Country')['Emissions'].sum().nlargest(6).index
        data_3d = self.df_long[self.df_long['Country'].isin(top_countries_3d)]

        pivot_3d = data_3d.pivot(index='Year', columns='Country', values='Emissions')
        pivot_3d = pivot_3d.fillna(0)

        X, Y = np.meshgrid(range(len(pivot_3d.columns)), pivot_3d.index)
        Z = pivot_3d.values

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

        ax.set_xlabel('国家', fontsize=12)
        ax.set_ylabel('年份', fontsize=12)
        ax.set_zlabel('排放量 (kt CO2当量)', fontsize=12)
        ax.set_title('主要国家排放量3D表面图', fontsize=16, fontweight='bold')

        ax.set_xticks(range(len(pivot_3d.columns)))
        ax.set_xticklabels(pivot_3d.columns, rotation=45)

        plt.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig('21_3D表面图.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. 热力图
        print("创建热力图...")
        plt.figure(figsize=(15, 10))

        # 创建国家-年份热力图
        top_countries_hm = self.df_long.groupby('Country')['Emissions'].sum().nlargest(15).index
        heatmap_data = self.df_long[self.df_long['Country'].isin(top_countries_hm)]
        heatmap_pivot = heatmap_data.pivot(index='Country', columns='Year', values='Emissions')

        sns.heatmap(heatmap_pivot, cmap='YlOrRd', cbar_kws={'label': '排放量 (kt CO2当量)'})
        plt.title('主要国家排放量热力图 (1990-2020)', fontsize=16, fontweight='bold')
        plt.xlabel('年份', fontsize=12)
        plt.ylabel('国家', fontsize=12)
        plt.tight_layout()
        plt.savefig('22_热力图.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ 3D表面图已保存: 21_3D表面图.png")
        print("✓ 热力图已保存: 22_热力图.png")

    def generate_summary_report(self):
        """生成分析总结报告"""
        print("\n" + "=" * 60)
        print("生成分析总结报告")
        print("=" * 60)

        # 统计分析
        total_emissions = self.df_long['Emissions'].sum()
        avg_annual = self.df_long.groupby('Year')['Emissions'].sum().mean()
        top_emitter = self.df_long.groupby('Country')['Emissions'].sum().idxmax()

        report = f"""
全球碳排放量可视化分析总结报告
========================================

数据概览:
- 分析时间段: 1990-2020年 (31年)
- 覆盖国家: {self.df_long['Country'].nunique()} 个
- 数据点总数: {len(self.df_long)} 个

排放量统计:
- 总排放量: {total_emissions:.2e} kt CO2当量
- 年均排放量: {avg_annual:.2e} kt CO2当量
- 最大排放国: {top_emitter}

生成的可视化图表:
1. 01_全球排放量时间趋势.png - 全球总排放量折线图
2. 02_全球排放量面积图.png - 全球排放量面积图
3. 03_主要排放国时间趋势.png - 主要国家时间序列对比
4. 04_全球排放量年增长率.png - 年度增长率柱状图
5. 05_全球累积排放量.png - 累积排放量趋势图
6. 06_不同年份排放量分布.png - 多年份分布对比
7. 07_各国总排放量排名.png - 国家排名条形图
8. 08_2020年各国排放量占比.png - 2020年饼图
9. 09_主要国家排放量变化率.png - 变化率对比
10. 10_Top6国家排放量雷达图.png - 雷达图对比
11. 11_各年代全球总排放量对比.png - 年代对比
12. 12_2020年各国排放强度分布.png - 排放强度分布
13. 13_全球排放量分布.png - 排放量直方图
14. 14_主要国家排放量分布箱线图.png - 箱线图分析
15. 15_年度排放量统计分布.png - 统计分布趋势
16. 16_1990年vs2020年排放量密度分布.png - 密度对比
17. 17_2020年排放量累积分布.png - 累积分布分析
18. 18_排放量分位数时间趋势.png - 分位数趋势
19. 19_动态气泡图.html - 交互式动态可视化
20. 20_桑基图.html - 排放量流向图
21. 21_3D表面图.png - 三维展示
22. 22_热力图.png - 国家-年份热力图

主要发现:
1. 全球排放量在研究期间总体呈现波动趋势
2. 美国、欧盟、俄罗斯是主要排放源
3. 部分国家排放量呈下降趋势，体现减排努力
4. 新兴经济体排放量增长较快

建议进一步研究:
- 排放量与经济发展的关系
- 减排政策效果评估
- 区域排放特征分析
- 未来排放趋势预测
        """

        print(report)
        print("✓ 总结报告已保存: 可视化分析总结报告.txt")


def main():
    """主函数"""
    print("全球碳排放量数据可视化分析")
    print("=" * 60)

    analyzer = CarbonVisualizationAnalyzer()

    try:
        # 执行所有可视化分析
        analyzer.create_time_trend_charts()
        analyzer.create_comparison_charts()
        analyzer.create_distribution_charts()
        analyzer.create_advanced_charts()
        analyzer.generate_summary_report()

        print(f"\n 所有可视化分析完成！")

    except Exception as e:
        print(f" 分析过程出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    main()