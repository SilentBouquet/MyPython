import requests
from bs4 import BeautifulSoup
import time

URL = 'https://www.netbian.com'
for s in range(1, 5):
    if s == 1:
        url = URL
    else:
        url = f'https://www.netbian.com/index_{s}.htm'
    resp = requests.get(url)
    resp.encoding = 'gbk'
    # print(resp.text)
    main_page = BeautifulSoup(resp.text, "html.parser")
    alist = main_page.find("div", class_="list").find_all("a")
    # print(alist)
    for i in alist:
        if i.get('href') == "https://pic.netbian.com/":
            continue
        href = URL + i.get('href')
        child_resp = requests.get(href)
        child_resp.encoding = 'gbk'
        child_page_text = child_resp.text
        child_page = BeautifulSoup(child_page_text, "html.parser")
        div = child_page.find("div", class_="pic")
        img = div.find("img")
        src = img.get('src')
        child_resp.close()
        img_resp = requests.get(src)
        img_name = src.split("/")[-1]
        with open("../壁纸图片/"+img_name, mode="wb") as f:
            f.write(img_resp.content)
        img_resp.close()
        print("over", img_name)
        time.sleep(1)
    resp.close()
print("all over")
