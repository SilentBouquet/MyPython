import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings('ignore')


class EChartsOptimizedDataPreparator:
    def __init__(self):
        """初始化数据准备器"""
        self.df_long = None
        self.df_wide = None
        self.load_clean_data()

    def load_clean_data(self):
        """加载预处理后的清洁数据"""
        try:
            print(" 加载清洁数据...")
            self.df_long = pd.read_csv('清洁数据_长格式.csv')
            self.df_wide = pd.read_csv('清洁数据_宽格式.csv')
            print(f" 数据加载成功: {self.df_long.shape[0]} 行, {len(self.df_long['Country'].unique())} 个国家")
        except Exception as e:
            print(f" 数据加载失败: {e}")
            return

    def prepare_sankey_data(self):
        print("\n🌊 准备桑基图数据...")

        # 使用2020年数据创建区域到国家的数据流
        sankey_data = self.df_long[self.df_long['Year'] == 2020].copy()

        # 区域映射
        region_mapping = {
            'United States of America': 'North America',
            'Canada': 'North America',
            'European Union (Convention)': 'Europe',
            'Germany': 'Europe',
            'United Kingdom of Great Britain and Northern Ireland': 'Europe',
            'France': 'Europe',
            'Italy': 'Europe',
            'Spain': 'Europe',
            'Netherlands': 'Europe',
            'Poland': 'Europe',
            'Russian Federation': 'Asia',
            'Japan': 'Asia',
            'Kazakhstan': 'Asia',
            'Türkiye': 'Asia',
            'Australia': 'Oceania',
            'New Zealand': 'Oceania'
        }

        sankey_data['Region'] = sankey_data['Country'].map(region_mapping).fillna('Other')

        # 选择前12个排放大国
        top_countries = sankey_data.nlargest(12, 'Emissions')

        # 构建桑基图数据结构
        nodes = []
        links = []

        # 添加区域节点
        regions = top_countries['Region'].unique()
        for region in regions:
            nodes.append({'name': str(region), 'category': 'region'})  # 转换为字符串

        # 添加国家节点
        for _, row in top_countries.iterrows():
            nodes.append({'name': str(row['Country']), 'category': 'country'})  # 转换为字符串

        # 添加全球总节点
        nodes.append({'name': '全球排放总量', 'category': 'global'})

        # 创建连接：区域到国家
        for _, row in top_countries.iterrows():
            links.append({
                'source': str(row['Region']),  # 转换为字符串
                'target': str(row['Country']),  # 转换为字符串
                'value': float(row['Emissions'])  # 转换为Python float
            })

        # 创建连接：国家到全球
        for _, row in top_countries.iterrows():
            links.append({
                'source': str(row['Country']),  # 转换为字符串
                'target': '全球排放总量',
                'value': float(row['Emissions'])  # 转换为Python float
            })

        sankey_result = {
            'nodes': nodes,
            'links': links,
            'title': '2020年全球碳排放流向图'
        }

        # 保存数据
        with open('ECharts_桑基图数据.json', 'w', encoding='utf-8') as f:
            json.dump(sankey_result, f, ensure_ascii=False, indent=2)

        print(" 桑基图数据已保存: ECharts_桑基图数据.json")
        return sankey_result

    def prepare_radar_data(self):
        print("\n 准备雷达图数据...")

        # 选择前6个排放大国进行多维度比较
        top_countries = self.df_long.groupby('Country')['Emissions'].sum().nlargest(6).index

        # 计算各个维度的指标
        radar_data = {}

        for country in top_countries:
            country_data = self.df_long[self.df_long['Country'] == country]

            # 计算各维度指标
            total_emissions = country_data['Emissions'].sum()
            avg_emissions = country_data['Emissions'].mean()
            max_emissions = country_data['Emissions'].max()

            # 计算增长率
            first_year = country_data[country_data['Year'] == country_data['Year'].min()]['Emissions'].iloc[0]
            last_year = country_data[country_data['Year'] == country_data['Year'].max()]['Emissions'].iloc[0]
            growth_rate = ((last_year - first_year) / first_year * 100) if first_year > 0 else 0

            # 计算波动性（标准差）
            volatility = country_data['Emissions'].std()

            # 计算近期趋势（最近10年平均）
            recent_data = country_data[country_data['Year'] >= 2010]
            recent_avg = recent_data['Emissions'].mean()

            radar_data[country] = {
                'total_emissions': total_emissions,
                'avg_emissions': avg_emissions,
                'max_emissions': max_emissions,
                'growth_rate': growth_rate,
                'volatility': volatility,
                'recent_avg': recent_avg
            }

        # 设置固定的最大值范围，确保雷达图有区分度
        all_values = list(radar_data.values())

        # 计算合适的最大值，增加区分度
        max_total = max([d['total_emissions'] for d in all_values])
        max_avg = max([d['avg_emissions'] for d in all_values])
        max_peak = max([d['max_emissions'] for d in all_values])
        max_growth = max([abs(d['growth_rate']) for d in all_values]) + 50  # 增加一些范围
        max_volatility = max([d['volatility'] for d in all_values])
        max_recent = max([d['recent_avg'] for d in all_values])

        indicators = [
            {'name': '总排放量', 'max': float(max_total * 1.2)},  # 增加20%余量
            {'name': '平均排放量', 'max': float(max_avg * 1.2)},
            {'name': '峰值排放量', 'max': float(max_peak * 1.2)},
            {'name': '增长率', 'max': 100.0},  # 固定最大值
            {'name': '波动性', 'max': float(max_volatility * 1.5)},  # 增加更多余量
            {'name': '近期平均', 'max': float(max_recent * 1.2)}
        ]

        # 构建雷达图数据 - 使用原始数值而非百分比
        radar_series = []
        for country, values in radar_data.items():
            radar_values = [
                float(values['total_emissions']),
                float(values['avg_emissions']),
                float(values['max_emissions']),
                float(abs(values['growth_rate'])),
                float(values['volatility']),
                float(values['recent_avg'])
            ]

            radar_series.append({
                'name': str(country),
                'value': radar_values
            })

        radar_result = {
            'indicators': indicators,
            'series': radar_series,
            'title': '主要排放国多维度雷达图'
        }

        # 保存数据
        with open('ECharts_雷达图数据.json', 'w', encoding='utf-8') as f:
            json.dump(radar_result, f, ensure_ascii=False, indent=2)

        print(" 雷达图数据已保存: ECharts_雷达图数据.json")
        return radar_result

    def prepare_animated_line_data(self):
        print("\n 准备动画时间序列数据...")

        # 选择前8个排放大国
        top_countries = self.df_long.groupby('Country')['Emissions'].sum().nlargest(8).index
        line_data = self.df_long[self.df_long['Country'].isin(top_countries)].copy()

        # 按年份和国家组织数据
        years = sorted(line_data['Year'].unique())

        # 构建ECharts需要的时间序列数据格式
        series_data = []

        for country in top_countries:
            country_data = line_data[line_data['Country'] == country].sort_values('Year')
            country_series = {
                'name': str(country),
                'type': 'line',
                'data': []
            }

            for year in years:
                year_data = country_data[country_data['Year'] == year]
                if not year_data.empty:
                    country_series['data'].append(float(year_data['Emissions'].iloc[0]))
                else:
                    country_series['data'].append(0)

            series_data.append(country_series)

        # 构建时间序列数据
        timeseries_data = {
            'years': [int(year) for year in years],  # x轴数据
            'series': series_data,  # 系列数据
            'title': '主要排放国时间趋势动画'
        }

        # 保存数据
        with open('ECharts_动画时间序列数据.json', 'w', encoding='utf-8') as f:
            json.dump(timeseries_data, f, ensure_ascii=False, indent=2)

        print(" 动画时间序列数据已保存: ECharts_动画时间序列数据.json")
        return timeseries_data

    def prepare_3d_surface_data(self):
        print("\n 准备3D表面图数据...")

        # 选择前10个国家创建3D数据
        top_countries = self.df_long.groupby('Country')['Emissions'].sum().nlargest(10).index
        surface_data = self.df_long[self.df_long['Country'].isin(top_countries)].copy()

        # 创建国家-年份矩阵
        pivot_data = surface_data.pivot(index='Country', columns='Year', values='Emissions')
        pivot_data = pivot_data.fillna(0)

        # 构建3D表面数据
        countries_list = list(pivot_data.index)
        years_list = list(pivot_data.columns)

        # 准备数据点
        data_points = []
        for i, country in enumerate(countries_list):
            for j, year in enumerate(years_list):
                value = pivot_data.loc[country, year]
                data_points.append([i, j, float(value)])  # 转换为Python float

        # 构建坐标轴数据
        surface_result = {
            'xAxis': {
                'name': '国家',
                'data': [str(country) for country in countries_list]  # 确保为字符串
            },
            'yAxis': {
                'name': '年份',
                'data': [int(year) for year in years_list]  # 转换为Python int
            },
            'zAxis': {
                'name': '排放量 (kt CO2当量)'
            },
            'data': data_points,
            'title': '主要国家排放量3D表面图'
        }

        # 添加额外的3D散点数据用于对比
        scatter_3d_data = []
        for _, row in surface_data.iterrows():
            country_index = countries_list.index(row['Country'])
            year_index = years_list.index(row['Year'])
            scatter_3d_data.append({
                'country': str(row['Country']),  # 转换为字符串
                'year': int(row['Year']),  # 转换为Python int
                'emissions': float(row['Emissions']),  # 转换为Python float
                'coordinates': [country_index, year_index, float(row['Emissions'])]  # 转换为Python float
            })

        surface_result['scatter_data'] = scatter_3d_data

        # 保存数据
        with open('ECharts_3D表面图数据.json', 'w', encoding='utf-8') as f:
            json.dump(surface_result, f, ensure_ascii=False, indent=2)

        print(" 3D表面图数据已保存: ECharts_3D表面图数据.json")
        return surface_result


def main():
    """主函数"""
    print(" ECharts优化数据准备")
    print("=" * 60)

    preparator = EChartsOptimizedDataPreparator()

    if preparator.df_long is None:
        return

    try:
        # 准备4个最适合ECharts的图表数据
        preparator.prepare_sankey_data()
        preparator.prepare_radar_data()
        preparator.prepare_animated_line_data()
        preparator.prepare_3d_surface_data()

        print(f"\n ECharts数据准备完成！")
        print(" 生成的文件:")
        print("    ECharts_桑基图数据.json")
        print("    ECharts_雷达图数据.json")
        print("    ECharts_动画时间序列数据.json")
        print("    ECharts_3D表面图数据.json")
        print("    ECharts数据说明.md")
        print("\n 这4个图表类型充分发挥了ECharts的优势！")

    except Exception as e:
        print(f" 数据准备过程出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()