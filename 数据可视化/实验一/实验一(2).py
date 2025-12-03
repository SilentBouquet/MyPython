# 导入所需的库
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# 读取文本文件并进行预处理
def preprocess_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # 去除噪声：换行符和标点符号
    text = text.replace('\n', '')
    text = re.sub(r'[^\w\s]', '', text)
    return text


# 分词函数
def segment_text(text):
    # 使用jieba进行分词
    words = jieba.lcut(text)

    return words


# 生成词云图函数
def generate_wordcloud(words, output_image_path):
    # 统计词频
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    # 创建词云对象
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        font_path='simhei.ttf'  # 指定中文字体路径，确保正确显示中文
    )
    # 生成词云
    wordcloud.generate_from_frequencies(word_freq)
    # 显示词云图
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    # 保存词云图
    wordcloud.to_file(output_image_path)
    print(f"词云图已保存到 {output_image_path}")


# 主函数
if __name__ == "__main__":
    import re  # 导入正则表达式库用于去除标点符号

    input_txt_file = "i have a dream.txt"
    output_image_file = "i have a dream.png"

    # 读取并预处理文本
    processed_text = preprocess_text(input_txt_file)

    # 分词
    segmented_words = segment_text(processed_text)

    # 生成词云图
    generate_wordcloud(segmented_words, output_image_file)