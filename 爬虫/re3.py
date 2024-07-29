import requests
import re

url = 'https://movie.douban.com/chart'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 '
                  'Safari/537.36 Edg/124.0.0.0'
}
resp = requests.get(url, headers=headers)
page_content = resp.text

obj = re.compile(r'<a class="nbg" href=".*?" {2}title="(?P<title>.*?)">', re.S)
obj1 = re.compile(r'<p class="pl">(?P<time>.*?)/', re.S)
result = obj.finditer(page_content)
result1 = obj1.finditer(page_content)
lst = []
lst1 = []
for i in result:
    lst.append(i.group('title'))
for item in result1:
    lst1.append(item.group('time'))
zipped = zip(lst, lst1)
for i in zipped:
    print(list(i))
resp.close()