import re
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud

text = ' '
with open("../python学习/我用什么把你留住.txt", 'r', encoding='utf-8') as f:
    text += f.read()
new_text = re.sub('[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)

cut = jieba.cut(new_text)
string = " ".join(cut)
wc = WordCloud(font_path=r"C:\Windows\Fonts\Microsoft YaHei UI\msyh.ttc",
               width=500, height=400, mode="RGBA", background_color=None).generate(string)

plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()