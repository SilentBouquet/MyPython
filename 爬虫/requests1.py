import requests

query = input("请输入一个你喜欢的明星：")
url = f'https://www.baidu.com/s?wd={query}'
dic = {
       'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                     'Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0'
}
resp = requests.get(url, headers=dic)       # 处理一个小小的反爬
print(resp)
print(resp.text)            # 拿到页面源代码
resp.close()