import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By

# 创建浏览器对象
driver = webdriver.Chrome()

# 打开目标网页
driver.get("https://www.nstl.gov.cn/resources_search.html?t=DegreePaper")

# 等待页面加载
time.sleep(2)

# 获取动态生成的参数
# 假设参数在某个元素的属性中
element = driver.find_element(By.XPATH, '//*[@id="pl_type_content"]/div[1]')

print(element.text)

# 关闭浏览器
driver.quit()


'''
# 使用获取的参数发送请求
import requests
url = "https://www.nstl.gov.cn/api/service/nstl/web/execute"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Cookie": "libcode=CN000000; nstl_token=3238ccce-be41-bee7-871f-e106537105d4",
    "Referer": "https://www.nstl.gov.cn/paper_detail.html?id=e1457b9afe83268f2ffe9bdd5f0bd5f1"
}
data = {
    "target": "nstl4.search4",
    "function": "paper/pc/detail",
    "param": param_value
}
response = requests.post(url, headers=headers, data=data)
print(response.json())
'''