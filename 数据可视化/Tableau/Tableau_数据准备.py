import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class TableauOptimizedDataPreparator:
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

    def prepare_heatmap_data(self):
        print("\n 准备热力图数据...")

        # 基础热力图数据
        heatmap_data = self.df_long.copy()

        # 添加地理区域分组
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
            'Belgium': 'Europe',
            'Austria': 'Europe',
            'Czechia': 'Europe',
            'Greece': 'Europe',
            'Portugal': 'Europe',
            'Hungary': 'Europe',
            'Sweden': 'Europe',
            'Denmark': 'Europe',
            'Finland': 'Europe',
            'Slovakia': 'Europe',
            'Ireland': 'Europe',
            'Lithuania': 'Europe',
            'Slovenia': 'Europe',
            'Latvia': 'Europe',
            'Estonia': 'Europe',
            'Croatia': 'Europe',
            'Luxembourg': 'Europe',
            'Malta': 'Europe',
            'Cyprus': 'Europe',
            'Iceland': 'Europe',
            'Liechtenstein': 'Europe',
            'Monaco': 'Europe',
            'Switzerland': 'Europe',
            'Norway': 'Europe',
            'Russian Federation': 'Asia',
            'Japan': 'Asia',
            'Kazakhstan': 'Asia',
            'Türkiye': 'Asia',
            'Ukraine': 'Europe',
            'Belarus': 'Europe',
            'Bulgaria': 'Europe',
            'Romania': 'Europe',
            'Australia': 'Oceania',
            'New Zealand': 'Oceania'
        }

        heatmap_data['Region'] = heatmap_data['Country'].map(region_mapping)

        # 添加排放等级
        def get_emission_level(emission):
            if emission < 50000:
                return 'Low'
            elif emission < 200000:
                return 'Medium'
            elif emission < 500000:
                return 'High'
            else:
                return 'Very High'

        heatmap_data['Emission_Level'] = heatmap_data['Emissions'].apply(get_emission_level)

        # 添加时期分组
        def get_decade(year):
            if year < 2000:
                return '1990s'
            elif year < 2010:
                return '2000s'
            elif year < 2020:
                return '2010s'
            else:
                return '2020s'

        heatmap_data['Decade'] = heatmap_data['Year'].apply(get_decade)

        # 计算相对排放量（便于比较）
        heatmap_data['Relative_Emissions'] = heatmap_data.groupby('Year')['Emissions'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
        )

        # 保存数据
        heatmap_data.to_excel('Tableau_热力图数据.xlsx', index=False)
        print(" 热力图数据已保存: Tableau_热力图数据.xlsx")
        return heatmap_data

    def prepare_bubble_chart_data(self):
        print("\n 准备动态气泡图数据...")

        bubble_data = self.df_long.copy()

        # 计算多个指标用于气泡图的不同维度
        np.random.seed(42)
        bubble_data['Per_Capita_Index'] = bubble_data['Emissions'] / np.random.uniform(10, 100, len(bubble_data))

        bubble_data['Economic_Index'] = bubble_data['Emissions'] * np.random.uniform(0.001, 0.01, len(bubble_data))

        # 大小：绝对排放量
        bubble_data['Size_Metric'] = bubble_data['Emissions']

        # 颜色：区域
        bubble_data['Region'] = bubble_data['Country'].map({
            'United States of America': 'North America',
            'Canada': 'North America',
            'European Union (Convention)': 'Europe',
            'Germany': 'Europe',
            'United Kingdom of Great Britain and Northern Ireland': 'Europe',
            'France': 'Europe',
            'Italy': 'Europe',
            'Russian Federation': 'Asia',
            'Japan': 'Asia',
            'Australia': 'Oceania',
            'China': 'Asia',
            'India': 'Asia'
        }).fillna('Other')

        # 添加发展状态
        developed_countries = [
            'United States of America', 'Germany', 'Japan', 'United Kingdom of Great Britain and Northern Ireland',
            'France', 'Italy', 'Canada', 'Australia', 'Netherlands', 'Spain',
            'Belgium', 'Austria', 'Sweden', 'Denmark', 'Finland', 'Norway', 'Switzerland'
        ]

        bubble_data['Development_Status'] = bubble_data['Country'].apply(
            lambda x: 'Developed' if x in developed_countries else 'Developing'
        )

        # 计算年度变化率
        bubble_data = bubble_data.sort_values(['Country', 'Year'])
        bubble_data['Growth_Rate'] = bubble_data.groupby('Country')['Emissions'].pct_change() * 100
        bubble_data['Growth_Rate'] = bubble_data['Growth_Rate'].fillna(0)

        # 保存数据
        bubble_data.to_excel('Tableau_动态气泡图数据.xlsx', index=False)
        print(" 动态气泡图数据已保存: Tableau_动态气泡图数据.xlsx")
        return bubble_data

    def prepare_ranking_bar_data(self):
        print("\n 准备排名条形图数据...")

        # 计算各种排名数据
        ranking_data_list = []

        # 1. 总排放量排名
        total_emissions = self.df_long.groupby('Country')['Emissions'].sum().reset_index()
        total_emissions['Ranking_Type'] = 'Total_Emissions'
        total_emissions['Rank'] = total_emissions['Emissions'].rank(method='dense', ascending=False)
        total_emissions['Metric_Name'] = 'Total Emissions (1990-2020)'
        ranking_data_list.append(total_emissions[['Country', 'Emissions', 'Rank', 'Ranking_Type', 'Metric_Name']])

        # 2. 2020年排放量排名
        emissions_2020 = self.df_long[self.df_long['Year'] == 2020].copy()
        emissions_2020['Ranking_Type'] = 'Emissions_2020'
        emissions_2020['Rank'] = emissions_2020['Emissions'].rank(method='dense', ascending=False)
        emissions_2020['Metric_Name'] = '2020 Emissions'
        ranking_data_list.append(emissions_2020[['Country', 'Emissions', 'Rank', 'Ranking_Type', 'Metric_Name']])

        # 3. 1990年排放量排名
        emissions_1990 = self.df_long[self.df_long['Year'] == 1990].copy()
        emissions_1990['Ranking_Type'] = 'Emissions_1990'
        emissions_1990['Rank'] = emissions_1990['Emissions'].rank(method='dense', ascending=False)
        emissions_1990['Metric_Name'] = '1990 Emissions'
        ranking_data_list.append(emissions_1990[['Country', 'Emissions', 'Rank', 'Ranking_Type', 'Metric_Name']])

        # 4. 平均增长率排名
        growth_rates = []
        for country in self.df_long['Country'].unique():
            country_data = self.df_long[self.df_long['Country'] == country].sort_values('Year')
            if len(country_data) >= 2:
                first_year = country_data.iloc[0]['Emissions']
                last_year = country_data.iloc[-1]['Emissions']
                if first_year > 0:
                    growth_rate = ((last_year - first_year) / first_year) * 100
                    growth_rates.append({
                        'Country': country,
                        'Emissions': growth_rate,
                        'Ranking_Type': 'Growth_Rate',
                        'Metric_Name': 'Growth Rate 1990-2020 (%)'
                    })

        growth_df = pd.DataFrame(growth_rates)
        if not growth_df.empty:
            growth_df['Rank'] = growth_df['Emissions'].rank(method='dense', ascending=False)
            ranking_data_list.append(growth_df[['Country', 'Emissions', 'Rank', 'Ranking_Type', 'Metric_Name']])

        # 合并所有排名数据
        ranking_data = pd.concat(ranking_data_list, ignore_index=True)

        # 添加分组信息
        region_mapping = {
            'United States of America': 'North America',
            'Canada': 'North America',
            'European Union (Convention)': 'Europe',
            'Germany': 'Europe',
            'United Kingdom of Great Britain and Northern Ireland': 'Europe',
            'France': 'Europe',
            'Italy': 'Europe',
            'Russian Federation': 'Asia',
            'Japan': 'Asia',
            'Australia': 'Oceania'
        }
        ranking_data['Region'] = ranking_data['Country'].map(region_mapping).fillna('Other')

        # 添加排放级别
        def get_emission_category(value, ranking_type):
            if ranking_type == 'Growth_Rate':
                if value < -20:
                    return 'Large Decrease'
                elif value < 0:
                    return 'Decrease'
                elif value < 20:
                    return 'Small Increase'
                else:
                    return 'Large Increase'
            else:
                if value < 100000:
                    return 'Low'
                elif value < 500000:
                    return 'Medium'
                elif value < 1000000:
                    return 'High'
                else:
                    return 'Very High'

        ranking_data['Category'] = ranking_data.apply(
            lambda row: get_emission_category(row['Emissions'], row['Ranking_Type']), axis=1
        )

        # 保存数据
        ranking_data.to_excel('Tableau_排名条形图数据.xlsx', index=False)
        print(" 排名条形图数据已保存: Tableau_排名条形图数据.xlsx")
        return ranking_data

    def prepare_hierarchical_pie_data(self):
        print("\n 准备分层饼图数据...")

        # 使用2020年数据作为基础
        pie_data = self.df_long[self.df_long['Year'] == 2020].copy()

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
            'Belgium': 'Europe',
            'Austria': 'Europe',
            'Czechia': 'Europe',
            'Greece': 'Europe',
            'Portugal': 'Europe',
            'Hungary': 'Europe',
            'Sweden': 'Europe',
            'Denmark': 'Europe',
            'Finland': 'Europe',
            'Slovakia': 'Europe',
            'Ireland': 'Europe',
            'Lithuania': 'Europe',
            'Slovenia': 'Europe',
            'Latvia': 'Europe',
            'Estonia': 'Europe',
            'Croatia': 'Europe',
            'Luxembourg': 'Europe',
            'Malta': 'Europe',
            'Cyprus': 'Europe',
            'Iceland': 'Europe',
            'Liechtenstein': 'Europe',
            'Monaco': 'Europe',
            'Switzerland': 'Europe',
            'Norway': 'Europe',
            'Russian Federation': 'Asia',
            'Japan': 'Asia',
            'Kazakhstan': 'Asia',
            'Türkiye': 'Asia',
            'Ukraine': 'Europe',
            'Belarus': 'Europe',
            'Bulgaria': 'Europe',
            'Romania': 'Europe',
            'Australia': 'Oceania',
            'New Zealand': 'Oceania'
        }

        pie_data['Region'] = pie_data['Country'].map(region_mapping)

        # 发展状态分类
        developed_countries = [
            'United States of America', 'Germany', 'Japan', 'United Kingdom of Great Britain and Northern Ireland',
            'France', 'Italy', 'Canada', 'Australia', 'Netherlands', 'Spain',
            'Belgium', 'Austria', 'Sweden', 'Denmark', 'Finland', 'Norway', 'Switzerland'
        ]

        pie_data['Development_Status'] = pie_data['Country'].apply(
            lambda x: 'Developed' if x in developed_countries else 'Developing'
        )

        # 排放等级分类
        def get_emission_tier(emission):
            if emission >= 1000000:
                return 'Tier 1 (>1M)'
            elif emission >= 500000:
                return 'Tier 2 (500K-1M)'
            elif emission >= 100000:
                return 'Tier 3 (100K-500K)'
            else:
                return 'Tier 4 (<100K)'

        pie_data['Emission_Tier'] = pie_data['Emissions'].apply(get_emission_tier)

        # 计算百分比
        total_emissions = pie_data['Emissions'].sum()
        pie_data['Percentage'] = (pie_data['Emissions'] / total_emissions) * 100

        # 创建层级数据
        # 第一层：区域
        region_summary = pie_data.groupby('Region').agg({
            'Emissions': 'sum',
            'Percentage': 'sum'
        }).reset_index()
        region_summary['Level'] = 'Region'
        region_summary['Parent'] = 'Global'
        region_summary['Country'] = region_summary['Region']
        # 为区域层级添加缺失的列
        region_summary['Development_Status'] = ''
        region_summary['Emission_Tier'] = ''

        # 第二层：国家
        country_data = pie_data.copy()
        country_data['Level'] = 'Country'
        country_data['Parent'] = country_data['Region']

        # 合并层级数据
        hierarchical_data = pd.concat([
            region_summary[
                ['Country', 'Region', 'Development_Status', 'Emission_Tier', 'Emissions', 'Percentage', 'Level',
                 'Parent']],
            country_data[
                ['Country', 'Region', 'Development_Status', 'Emission_Tier', 'Emissions', 'Percentage', 'Level',
                 'Parent']]
        ], ignore_index=True)

        # 添加排名
        hierarchical_data['Rank'] = hierarchical_data.groupby('Level')['Emissions'].rank(method='dense',
                                                                                         ascending=False)

        # 保存数据
        hierarchical_data.to_excel('Tableau_分层饼图数据.xlsx', index=False)
        print(" 分层饼图数据已保存: Tableau_分层饼图数据.xlsx")
        return hierarchical_data


def main():
    """主函数"""
    print(" Tableau优化数据准备")
    print("=" * 60)

    preparator = TableauOptimizedDataPreparator()

    if preparator.df_long is None:
        return

    try:
        # 准备4个最适合Tableau的图表数据
        preparator.prepare_heatmap_data()
        preparator.prepare_bubble_chart_data()
        preparator.prepare_ranking_bar_data()
        preparator.prepare_hierarchical_pie_data()

        print(f"\n Tableau数据准备完成！")
        print(" 生成的文件:")
        print("    Tableau_热力图数据.xlsx")
        print("    Tableau_动态气泡图数据.xlsx")
        print("    Tableau_排名条形图数据.xlsx")
        print("    Tableau_分层饼图数据.xlsx")

    except Exception as e:
        print(f" 数据准备过程出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()