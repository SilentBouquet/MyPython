# 代理：通过第三方的一个机器去发送请求
import requests

proxies = {
    "https": "http://120.24.50.164:3128"
}

url = "https://www.baidu.com"
resp = requests.get(url, proxies=proxies)
resp.encoding = 'utf-8'
print(resp.text)