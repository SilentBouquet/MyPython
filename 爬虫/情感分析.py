import json
import requests
import matplotlib.pyplot as plt

# text = ' '
# with open("../python学习/我用什么把你留住.txt", 'r', encoding='utf-8') as f:
# text += f.read().replace("\n", "")

text = "我从未拥有过你一秒钟，心里却失去过你千万次。"
content = json.dumps({
    "text": text
})

url = ("https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=MjVVQmu5M089ZN7cbHuFRb6T"
       "&client_secret=z0z1ti60E05Y25JRMnpdxafZJo8kWu8m")

headers = {
    'Content-Type': 'application/json',
}

response = requests.request("POST", url, headers=headers, data=content)
# print(response.json())

mytoken = response.json()["access_token"]
url1 = f"https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token={mytoken}"

result = requests.post(url=url1, headers=headers, data=content).json()
print(result)

plt.figure()
x = ['正面情感词频', '负面情感词频']
y = [result["items"][0]["positive_prob"], result["items"][0]["negative_prob"]]
plt.bar(x, y, color=['pink', 'gray'])
plt.rcParams["font.sans-serif"] = 'SimHei'
for x, y in zip(x, y):
    plt.text(x, y + 0.05, '%.6f' % y, ha='center', va='bottom')
plt.title('情感词频统计')
plt.xlabel('情感类型')
plt.ylabel('词频')
plt.ylim(0, 1.2)
plt.yticks(())
plt.show()