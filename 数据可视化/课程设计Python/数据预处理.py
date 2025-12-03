import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def preprocess_emission_data(file_path):
    """预处理碳排放数据"""
    print("=" * 60)
    print("开始数据预处理")
    print("=" * 60)

    try:
        # 1. 读取原始数据
        print("1. 读取原始Excel文件...")
        df_raw = pd.read_excel(file_path)
        print(f"   原始数据形状: {df_raw.shape}")

        # 2. 分析数据结构
        print("\n2. 分析数据结构...")
        print("   前几行预览:")
        print(df_raw.head(3))

        # 3. 提取年份信息（从第2行）
        print("\n3. 提取年份信息...")
        year_row = df_raw.iloc[1]  # 第2行包含年份信息
        years = []
        year_columns = []

        # 从第3列开始查找年份
        for i, val in enumerate(year_row[2:], start=2):
            try:
                year = int(float(val))
                if 1990 <= year <= 2030:
                    years.append(year)
                    year_columns.append(i)
            except (ValueError, TypeError):
                continue

        print(f"   识别到的年份: {years}")
        print(f"   年份列索引: {year_columns}")

        # 4. 提取国家数据（从第3行开始）
        print("\n4. 提取国家数据...")
        # 找到实际的国家数据行（跳过标题行）
        country_start_row = 2  # 从第3行开始

        # 提取国家名称列（第1列）
        countries = []
        country_rows = []

        for i in range(country_start_row, len(df_raw)):
            country_name = df_raw.iloc[i, 0]  # 第1列
            if pd.notna(country_name) and str(country_name).strip() not in ['', 'nan', '派对']:
                countries.append(str(country_name).strip())
                country_rows.append(i)

        print(f"   识别到的国家数量: {len(countries)}")
        print(f"   前10个国家: {countries[:10]}")

        # 5. 构建清洁的数据集
        print("\n5. 构建清洁数据集...")

        # 创建新的DataFrame
        clean_data = []

        for country_idx, country in enumerate(countries):
            row_idx = country_rows[country_idx]
            country_data = {'Country': country}

            # 提取该国家对应年份的排放量数据
            for year_idx, year in enumerate(years):
                col_idx = year_columns[year_idx]
                try:
                    value = df_raw.iloc[row_idx, col_idx]
                    # 转换为数值，处理缺失值
                    if pd.notna(value):
                        country_data[str(year)] = float(value)
                    else:
                        country_data[str(year)] = np.nan
                except (ValueError, IndexError):
                    country_data[str(year)] = np.nan

            clean_data.append(country_data)

        # 转换为DataFrame
        df_clean = pd.DataFrame(clean_data)

        print(f"   清洁数据形状: {df_clean.shape}")
        print(f"   列名: {list(df_clean.columns)}")

        # 6. 数据质量检查
        print("\n6. 数据质量检查...")

        # 检查缺失值
        missing_by_country = df_clean.isnull().sum(axis=1)
        missing_by_year = df_clean.isnull().sum(axis=0)

        print(f"   每个国家的缺失值统计:")
        countries_with_missing = missing_by_country[missing_by_country > 0]
        if len(countries_with_missing) > 0:
            print(f"   有缺失值的国家数量: {len(countries_with_missing)}")
            for idx, missing_count in countries_with_missing.head(10).items():
                print(f"     {df_clean.iloc[idx]['Country']}: {missing_count} 个缺失值")
        else:
            print("   ✓ 所有国家数据完整")

        print(f"   每年的缺失值统计:")
        years_with_missing = missing_by_year[missing_by_year > 0]
        if len(years_with_missing) > 0:
            for year, missing_count in years_with_missing.items():
                if year != 'Country':
                    print(f"     {year}年: {missing_count} 个国家缺失数据")

        # 7. 创建长格式数据
        print("\n7. 创建长格式数据...")
        year_cols = [col for col in df_clean.columns if col != 'Country']

        df_long = df_clean.melt(
            id_vars=['Country'],
            value_vars=year_cols,
            var_name='Year',
            value_name='Emissions'
        )

        # 转换数据类型
        df_long['Year'] = pd.to_numeric(df_long['Year'])
        df_long['Emissions'] = pd.to_numeric(df_long['Emissions'], errors='coerce')

        # 移除缺失值行
        df_long_clean = df_long.dropna()

        print(f"   长格式数据形状: {df_long.shape}")
        print(f"   清理后长格式数据形状: {df_long_clean.shape}")

        # 8. 基本统计信息
        print("\n8. 基本统计信息...")
        print(f"   时间跨度: {df_long_clean['Year'].min():.0f} - {df_long_clean['Year'].max():.0f}")
        print(f"   国家数量: {df_long_clean['Country'].nunique()}")
        print(f"   数据点总数: {len(df_long_clean)}")
        print(f"   排放量统计:")
        print(f"     最小值: {df_long_clean['Emissions'].min():.2f} kt CO₂当量")
        print(f"     最大值: {df_long_clean['Emissions'].max():.2f} kt CO₂当量")
        print(f"     平均值: {df_long_clean['Emissions'].mean():.2f} kt CO₂当量")
        print(f"     中位数: {df_long_clean['Emissions'].median():.2f} kt CO₂当量")

        # 9. 保存处理后的数据
        print("\n9. 保存处理后的数据...")

        # 保存宽格式数据
        df_clean.to_csv('清洁数据_宽格式.csv', encoding='utf-8-sig', index=False)
        print("   ✓ 宽格式数据已保存: 清洁数据_宽格式.csv")

        # 保存长格式数据
        df_long_clean.to_csv('清洁数据_长格式.csv', encoding='utf-8-sig', index=False)
        print("   ✓ 长格式数据已保存: 清洁数据_长格式.csv")

        # 保存数据样本预览
        print("\n10. 数据预览:")
        print("    宽格式数据前5行:")
        print(df_clean.head())
        print("\n    长格式数据前10行:")
        print(df_long_clean.head(10))

        # 生成预处理报告
        print("\n11. 生成预处理报告...")

        report = f"""
数据预处理报告
==================

原始数据:
- 原始文件: {file_path}
- 原始维度: {df_raw.shape[0]} 行 × {df_raw.shape[1]} 列

处理后数据:
- 宽格式: {df_clean.shape[0]} 行 × {df_clean.shape[1]} 列
- 长格式: {df_long_clean.shape[0]} 行 × {df_long_clean.shape[1]} 列

时间信息:
- 年份范围: {df_long_clean['Year'].min():.0f} - {df_long_clean['Year'].max():.0f}
- 年份数量: {len(years)} 年

地理信息:
- 国家/地区数量: {len(countries)}
- 有效数据的国家数量: {df_long_clean['Country'].nunique()}

排放量统计:
- 数据点总数: {len(df_long_clean)}
- 最小排放量: {df_long_clean['Emissions'].min():.2f} kt CO₂当量
- 最大排放量: {df_long_clean['Emissions'].max():.2f} kt CO₂当量
- 平均排放量: {df_long_clean['Emissions'].mean():.2f} kt CO₂当量
- 总排放量: {df_long_clean['Emissions'].sum():.2f} kt CO₂当量

排放量最高的前5个国家:
{df_long_clean.groupby('Country')['Emissions'].sum().nlargest(5).to_string()}

数据文件:
- 清洁数据_宽格式.csv: 用于表格分析
- 清洁数据_长格式.csv: 用于可视化分析

数据已准备就绪，可以开始可视化分析！
"""

        print(report)
        print("   ✓ 预处理报告已保存: 数据预处理报告.txt")

        return df_clean, df_long_clean

    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """主函数"""
    file_path = "Time Series - 含土地利用、土地利用变化和林业的温室气体总量, in kt CO₂ equivalent.xlsx"

    print("开始碳排放数据预处理...")
    df_wide, df_long = preprocess_emission_data(file_path)

    if df_wide is not None and df_long is not None:
        print(f"\n 数据预处理成功完成！")
    else:
        print(f"\n 数据预处理失败，请检查原始数据格式。")


if __name__ == "__main__":
    main()