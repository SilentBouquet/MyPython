# 防盗链：溯源，本次请求的上一级是谁
import requests

url = 'https://www.pearvideo.com/video_1794130'
contID = url.split("_")[1]

videoStatusUrl = f"https://www.pearvideo.com/videoStatus.jsp?contId={contID}&mrd=0.67114665156251"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 '
                  'Safari/537.36 Edg/124.0.0.0',
    'Referer': url
}
resp = requests.get(videoStatusUrl, headers=headers)
dic = resp.json()
srcUrl = dic['videoInfo']['videos']['srcUrl']
systemTime = dic['systemTime']
True_srcUrl = srcUrl.replace(systemTime, f"cont-{contID}")
resp.close()

with open("../操作文件/video1.mp4", 'wb') as f:
    f.write(requests.get(True_srcUrl).content)

print("over!")