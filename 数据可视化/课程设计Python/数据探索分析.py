import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def explore_excel_data(file_path):
    """探索Excel数据文件"""
    print("=" * 80)
    print("全球碳排放量数据探索分析")
    print("=" * 80)

    try:
        # 1. 读取Excel文件
        print("\n1. 读取Excel文件...")
        df = pd.read_excel(file_path)
        print(f"✓ 文件读取成功！")

        # 2. 基本信息
        print(f"\n2. 数据基本信息:")
        print(f"   - 数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"   - 内存使用: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        # 3. 列名分析
        print(f"\n3. 列名分析:")
        print(f"   - 总列数: {len(df.columns)}")
        print(f"   - 前10个列名:")
        for i, col in enumerate(df.columns[:10]):
            print(f"     {i + 1:2d}. '{col}' (类型: {df[col].dtype})")

        if len(df.columns) > 10:
            print(f"   - ... 还有 {len(df.columns) - 10} 列")
            print(f"   - 最后5个列名:")
            for i, col in enumerate(df.columns[-5:], len(df.columns) - 4):
                print(f"     {i:2d}. '{col}' (类型: {df[col].dtype})")

        # 4. 年份列识别
        print(f"\n4. 年份列识别:")
        year_columns = []
        other_columns = []

        for col in df.columns:
            col_str = str(col).strip()
            try:
                year = int(col_str)
                if 1900 <= year <= 2030:
                    year_columns.append(col)
                else:
                    other_columns.append(col)
            except (ValueError, TypeError):
                other_columns.append(col)

        print(f"   - 识别到年份列: {len(year_columns)} 个")
        if year_columns:
            print(f"   - 年份范围: {min(year_columns)} - {max(year_columns)}")
            print(f"   - 年份列样例: {year_columns[:5]}...")

        print(f"   - 非年份列: {len(other_columns)} 个")
        for col in other_columns:
            print(f"     - '{col}'")

        # 5. 数据预览
        print(f"\n5. 数据预览:")
        print("   前5行数据:")
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 120)
        print(df.head())

        # 6. 第一列分析（通常是国家/地区）
        first_col = df.columns[0]
        print(f"\n6. 第一列分析 ('{first_col}'):")
        print(f"   - 数据类型: {df[first_col].dtype}")
        print(f"   - 唯一值数量: {df[first_col].nunique()}")
        print(f"   - 缺失值数量: {df[first_col].isnull().sum()}")
        print(f"   - 前10个值:")
        for i, val in enumerate(df[first_col].head(10)):
            print(f"     {i + 1:2d}. {val}")

        # 7. 数值列分析
        print(f"\n7. 数值列分析:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"   - 数值列数量: {len(numeric_cols)}")

        if len(numeric_cols) > 0:
            print(f"   - 数值列统计:")
            stats = df[numeric_cols].describe()
            print(stats)

        # 8. 缺失值分析
        print(f"\n8. 缺失值分析:")
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100

        missing_summary = pd.DataFrame({
            '列名': missing_data.index,
            '缺失数量': missing_data.values,
            '缺失百分比': missing_percent.values
        })
        missing_summary = missing_summary[missing_summary['缺失数量'] > 0].sort_values('缺失数量', ascending=False)

        if len(missing_summary) > 0:
            print(f"   - 有缺失值的列数: {len(missing_summary)}")
            print(missing_summary.head(10))
        else:
            print("   ✓ 没有发现缺失值")

        # 9. 数据质量检查
        print(f"\n9. 数据质量检查:")

        # 检查重复行
        duplicate_rows = df.duplicated().sum()
        print(f"   - 重复行数量: {duplicate_rows}")

        # 检查年份列的数据范围
        if year_columns:
            print(f"   - 年份列数据范围检查:")
            for col in year_columns[:5]:  # 检查前5个年份列
                col_data = pd.to_numeric(df[col], errors='coerce')
                valid_count = col_data.notna().sum()
                min_val = col_data.min()
                max_val = col_data.max()
                print(f"     {col}: 有效值 {valid_count}, 范围 [{min_val:.0f}, {max_val:.0f}]")

        # 10. 保存数据样本
        print(f"\n10. 保存数据样本:")
        sample_file = "数据样本.csv"
        df.head(20).to_csv(sample_file, encoding='utf-8-sig', index=False)
        print(f"   ✓ 前20行数据已保存到: {sample_file}")

        # 11. 生成数据结构报告
        print(f"\n11. 生成数据结构报告:")

        report = f"""
数据结构分析报告
===================

文件信息:
- 文件名: {file_path}
- 数据维度: {df.shape[0]} 行 × {df.shape[1]} 列
- 内存使用: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB

列结构:
- 总列数: {len(df.columns)}
- 年份列: {len(year_columns)} 个 (范围: {min(year_columns) if year_columns else 'N/A'} - {max(year_columns) if year_columns else 'N/A'})
- 非年份列: {len(other_columns)} 个

数据类型分布:
- 数值型列: {len(df.select_dtypes(include=[np.number]).columns)} 个
- 文本型列: {len(df.select_dtypes(include=['object']).columns)} 个
- 其他类型列: {len(df.columns) - len(df.select_dtypes(include=[np.number, 'object']).columns)} 个

数据质量:
- 重复行: {duplicate_rows} 行
- 有缺失值的列: {len(missing_summary)} 个
- 数据完整性: {((len(df) * len(df.columns) - df.isnull().sum().sum()) / (len(df) * len(df.columns)) * 100):.1f}%

第一列信息 ('{first_col}'):
- 数据类型: {df[first_col].dtype}
- 唯一值数量: {df[first_col].nunique()}
- 可能是: 国家/地区标识

建议的下一步:
1. 确认第一列是否为国家/地区名称
2. 验证年份列的数据质量
3. 处理缺失值（如有）
4. 开始可视化分析
"""

        # 保存报告
        with open('数据结构报告.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        print(report)
        print(f"   ✓ 完整报告已保存到: 数据结构报告.txt")

        return df, year_columns, other_columns

    except Exception as e:
        print(f"❌ 数据探索失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    """主函数"""
    file_path = "Time Series - 含土地利用、土地利用变化和林业的温室气体总量, in kt CO₂ equivalent.xlsx"

    print("开始数据探索...")
    df, year_columns, other_columns = explore_excel_data(file_path)

    if df is not None:
        print(f"\n 数据探索完成！")
    else:
        print(f"\n 数据探索失败，请检查文件路径和格式。")


if __name__ == "__main__":
    main()