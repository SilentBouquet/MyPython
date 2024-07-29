import requests
from bs4 import BeautifulSoup

# session可以认为是一连串的请求，这个过程中的cookies不会丢失
# 会话
session = requests.session()
data = {
    'action': 'login',
    'username': '17387461002',
    'password': 'yy040806'
}
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 '
                  'Safari/537.36 Edg/124.0.0.0'
}

URL = 'https://www.bqguu.cc'
url = "https://www.bqguu.cc/user/action.html"
session.post(url, data=data, headers=headers)
url2 = "https://www.bqguu.cc/user/action.html?action=bookcase&t=1715153097464"
resp = session.get(url2)
page = resp.json()[0]
novel_url = URL + page.get("url_list")
s = ''
for i in range(1, 3):
    novel_URL = novel_url + f'{i}.html'
    novel_resp = session.get(novel_URL)
    content = BeautifulSoup(novel_resp.text, "html.parser")
    novel = content.find("div", class_="Readarea ReadAjax_content").text
    s += novel
    novel_resp.close()
str1 = s.replace("　　记住网址．com", "")
str2 = str1.replace("　　请收藏本站：https://www.bqguu.cc。笔趣阁手机版：https://m.bqguu.cc ", "")
str3 = str2.replace("『点此报错』『加入书签』", "")
str4 = str3.replace("　　", '\n'+'\n')
print("\n"+str4.strip())
resp.close()