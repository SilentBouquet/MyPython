import requests
import re

domain = "https://dytt89.com/"
resp = requests.get(domain, verify=False)  # verify=False 去掉安全验证
resp.encoding = 'gb2312'  # 指定字符集

obj1 = re.compile(r'2024必看热片.*?<ul>(?P<ul>.*?)</ul>', re.S)
obj2 = re.compile(r"<a href='(?P<href>.*?)'", re.S)
obj3 = re.compile(r'◎片　　名　(?P<name>.*?)<br />.*?<td style='
                  r'"WORD-WRAP: break-word" bgcolor="#fdfddf"><a href="(?P<download>.*?)">', re.S)
result1 = obj1.finditer(resp.text)
child_href_list = []
for it in result1:
    ul = it.group('ul').strip()
    result2 = obj2.finditer(ul)
    for item in result2:
        child_href = domain + item.group('href').strip("/")
        child_href_list.append(child_href)

for href in child_href_list:
    child_resp = requests.get(href, verify=False)
    child_resp.encoding = 'gb2312'
    result3 = obj3.search(child_resp.text)
    print(result3.group('name'))
    print(result3.group('download'))
resp.close()