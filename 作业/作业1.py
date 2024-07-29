from tqdm import tqdm
import requests
from lxml import etree

url = 'https://www.scuec.edu.cn/xww/info/1002/12141.htm'
for i in tqdm(range(12141, 3090, -1)):
    next_url = f'https://www.scuec.edu.cn/xww/info/1002/{str(i)}.htm'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0'
    }
    resp = requests.get(next_url, headers=headers)

    if resp.status_code == 200:
        resp.encoding = 'UTF-8'
        html = etree.HTML(resp.text)

        # 查找标题
        title = html.xpath('//title/text()')[0]
        print(f"标题: {title}")

        # 查找文章的作者、来源以及责编、审核、上传者和发布时间
        writer = ''.join(html.xpath('/html/body/div[5]/div/div[1]/div[2]/form/div/p[1]/text()')).strip()
        time = ''.join(html.xpath('/html/body/div[5]/div/div[1]/div[2]/form/div/p[2]/text()')).strip()
        print(writer)
        print(time)

        # 查找正文
        content = ''.join(html.xpath('/html/body/div[5]/div/div[1]/div[2]/form/div/div[1]/div/div/p/text()')).strip()
        print(f"正文:{content}")