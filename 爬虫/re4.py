import csv
import re
import requests

page = []
for i in range(0, 10):
    page.append(25*i)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 '
                  'Safari/537.36 Edg/124.0.0.0'
}
f = open("../操作文件/data.csv", mode="w")
csv_writer = csv.writer(f)
for i in page:
    url = f"https://movie.douban.com/top250?start={i}&filter="
    resp = requests.get(url, headers=headers)
    page_content = resp.text
    obj = re.compile(r'<div class="hd">.*?<span class="title">(?P<name>.*?)'
                     r'</span>.*?<p class="">.*?<br>(?P<time>.*?)&nbsp.*?'
                     r'<span class="rating_num" property="v:average">(?P<score>.*?)</span>'
                     r'.*?<span>(?P<num>.*?)</span>', re.S)
    result = obj.finditer(page_content)
    for item in result:
        dic = item.groupdict()
        dic['time'] = dic['time'].strip()
        csv_writer.writerow(dic.values())
    resp.close()
f.close()
print("over!")