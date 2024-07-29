import re
import jieba
from matplotlib import pyplot as plt
from wordcloud import WordCloud

text = " "
with open("../操作文件/小说.txt", mode="r", encoding="utf-8") as f:
    text += f.read()

text = text.replace("    ", "")
new_text = re.sub('[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)

stopwords = set()
word = [line.strip() for line in open("../操作文件/中文停用词表.txt", 'r', encoding='utf-8')]
stopwords.update(word)

cut = jieba.cut(new_text)
string = " ".join(cut)
wc = WordCloud(font_path=r"C:\Windows\Fonts\Microsoft YaHei UI\msyh.ttc",
               width=500, height=400, mode="RGBA", background_color=None,
               max_words=40, stopwords=stopwords).generate(string)

plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()