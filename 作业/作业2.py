import re
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.charts import Pie
from wordcloud import WordCloud

# 数据导入
city_name = ["神农架林区", "恩施州", "十堰市", "宜昌市", "襄阳市", "咸宁市", "荆门市", "潜江市",
             "黄冈市", "黄石市", "随州市", "鄂州市", "荆州市", "仙桃市", "天门市", "孝感市", "武汉市"]

CW = [2.6016, 3.0729, 3.4300, 3.4555, 4.0543, 4.1145, 4.1777, 4.2574,
      4.4093, 4.4914, 4.6480, 4.8873, 4.9619, 5.0019, 5.0204, 5.0245, 5.3657]

# 绘制直方图
bar = Bar()
bar.add_xaxis(city_name)
bar.add_yaxis("CWQI", CW)
bar.set_global_opts(title_opts=opts.TitleOpts(title="2023年9月湖北省地表水环境质量月报"))
bar.render("2023年9月湖北省地表水环境质量月报直方图.html")

# 绘制饼状图
pie = Pie()
pie.add("2023年9月湖北省地表水环境质量月报", [list(z) for z in zip(city_name, CW)])
pie.render("2023年9月湖北省地表水环境质量月报饼状图.html")

# 提取词云
url = "https://sthjt.hubei.gov.cn/fbjd/xxgkml/gysyjs/sthj/hjzl/dbsh/202310/t20231013_4888019.shtml"
resp = requests.get(url)
resp.encoding = resp.apparent_encoding
essay = resp.text
content = BeautifulSoup(resp.text, "html.parser")
page = content.find("div", class_="view TRS_UEDITOR trs_paper_default trs_web trs_word")
alist = page.find_all("span")

with open('essay.text', 'w', encoding='utf-8') as f:
    for i in alist:
        f.write(str(i.text.strip()) + '\n')

text = ' '
with open('essay.text', 'r', encoding='utf-8') as f:
    text += f.read()
new_text = re.sub('[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)

wc = WordCloud(font_path=r"C:\Windows\Fonts\Microsoft YaHei UI\msyh.ttc",
               width=500, height=400, mode="RGBA", background_color=None).generate(new_text)

# 显示词云图
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()