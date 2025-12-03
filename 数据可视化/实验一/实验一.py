import jieba
import pandas as pd
from collections import Counter
import re  # 导入正则表达式库用于去除标点符号


# 读取文本文件并进行分词
def read_and_segment(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # 去除换行符
    text = text.replace('\n', '')

    # 使用正则表达式去除标点符号
    text = re.sub(r'[^\w\s]', '', text)

    # 使用jieba进行分词
    words = jieba.lcut(text)

    while ' ' in words:
        words.remove(' ')

    return words


# 统计词频并保存为Excel文件
def save_to_excel(words, excel_file_path):
    # 统计词频
    word_counts = Counter(words)

    # 转换为DataFrame
    df = pd.DataFrame(word_counts.items(), columns=['关键词', '词频'])

    # 按词频降序排序
    df = df.sort_values(by='词频', ascending=False)

    # 保存为Excel文件
    df.to_excel(excel_file_path, index=False)
    print(f"词频统计结果已保存到 {excel_file_path}")


# 主函数
if __name__ == "__main__":
    input_txt_file = "i have a dream.txt"  # 替换为你的输入txt文件路径
    output_excel_file = "i have a dream.xlsx"  # 替换为你希望输出的Excel文件路径

    # 读取文本文件并分词
    segmented_words = read_and_segment(input_txt_file)

    # 保存分词结果及其词频到Excel
    save_to_excel(segmented_words, output_excel_file)