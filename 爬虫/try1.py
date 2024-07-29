from urllib.request import urlopen

url = "http://www.baidu.com"
resp = urlopen(url)
with open('../操作文件/mybaidu.html', mode='w', encoding='utf-8') as f:
    f.write(resp.read().decode("utf-8"))        # 读取到的是网页的页面源代码
print('over!')