import requests
from lxml import etree

for i in range(1, 5):
    url = f"https://www.zbj.com/fw/?k=saas&p={i}&osStr=ad-0,np-0,rf-0,sr-60,mb-0"
    resp = requests.get(url)

    html = etree.HTML(resp.text)
    divs = html.xpath("/html/body/div[1]/div/div/div[3]/div/div[4]/div/div[2]/div[1]/div")

    for div in divs:
        lst = []
        price = div.xpath("./div/div[3]/div[1]/span/text()")[0].strip("¥")
        title = "saas".join(div.xpath("./div/div[3]/div[2]/a/text()"))
        com_name = div.xpath("./div/a/div[2]/div[1]/div/text()")[0]
        lst.append(title)
        lst.append(com_name)
        lst.append(price + '元')
        print(lst)

    resp.close()