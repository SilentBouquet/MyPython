import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL = "https://kyfw.12306.cn/otn/resources/login.html"


def login():
    # 登录12306网站
    driver.get(URL)

    '''
    # 输入电话号码和密码
    username = driver.find_element(By.ID, "J-userName")
    password = driver.find_element(By.ID, "J-password")
    input_username = "17387461002"
    input_password = "yy040806"
    username.send_keys(input_username)
    password.send_keys(input_password)

    # 点击登录
    driver.find_element(By.ID, "J-login").click()

    # 输入身份证号后四位以及验证码
    id_card = driver.find_element(By.ID, "id_card")
    input_id_card = "2715"
    id_card.send_keys(input_id_card)
    driver.find_element(By.ID, "verification_code").click()
    code = driver.find_element(By.ID, "code")
    input_code = input("请输入验证码：")
    code.send_keys(input_code)
    driver.find_element(By.ID, "sureClick").click()
    '''
    # 二维码登录
    driver.find_element(By.XPATH, '//*[@id="toolbar_Div"]/div[2]/div[2]/ul/li[2]/a').click()

    # 检查是否登录成功
    while True:
        time.sleep(1)
        url = driver.current_url
        if url != URL:
            break

    # 定位到单程票的页面
    time.sleep(3)
    driver.find_element(By.XPATH, '//*[@id="J-chepiao"]/a').click()
    time.sleep(1)
    driver.find_element(By.XPATH, '//*[@id="megamenu-3"]/div[1]/ul/li[1]/a').click()
    return driver


def query_train_tickets():
    # 填入车票信息
    driver.find_element(By.ID, "fromStationText").click()
    start = driver.find_element(By.XPATH, '//*[@id="fromStationText"]')
    input_start = "曲靖北"
    start.send_keys(input_start)
    driver.find_element(By.XPATH, '//*[@id="citem_0"]').click()
    driver.find_element(By.ID, "toStationText").click()
    end = driver.find_element(By.XPATH, '//*[@id="toStationText"]')
    input_end = "武汉"
    end.send_keys(input_end)
    driver.find_element(By.XPATH, '//*[@id="citem_0"]').click()
    driver.find_element(By.XPATH, '//*[@id="train_date"]').clear()
    data = driver.find_element(By.XPATH, '//*[@id="train_date"]')
    input_date = "2024-09-03"
    data.send_keys(input_date)
    driver.find_element(By.ID, "sf2").click()
    driver.find_element(By.ID, "query_ticket").click()
    return driver


def IsSure():
    # 等待页面加载完成，设置最大等待时间为10秒
    wait = WebDriverWait(driver, 10)

    try:
        # 等待查询结果元素出现
        result_element = wait.until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="ticket_80000G154202_02_14"]/td[13]/a'))
        )

        if result_element.text:
            result_element.click()

            try:
                palce_order()

            except Exception as e:
                print("")
                return

    except Exception as e:
        print(f"暂时无票，继续监控")


def palce_order():
    print("检查到有票，执行下单操作中···")
    time.sleep(2)

    # 找到乘车人信息
    checkbox_for_order = driver.find_element(By.XPATH, '//*[@id="normalPassenger_0"]')

    # 判断复选框是否已被勾选，如果未勾选，则点击勾选
    if not checkbox_for_order.is_selected():
        checkbox_for_order.click()

    # 检查是否需要确认学生购票，有则确认
    checkbox_for_student = driver.find_element(By.XPATH, '//*[@id="dialog_xsertcj_ok"]')
    if checkbox_for_student:
        checkbox_for_student.click()

    # 找到并点击提交订单按钮
    submit_order = driver.find_element(By.XPATH, '//*[@id="submitOrder_id"]')
    submit_order.click()
    time.sleep(2)

    isF = driver.find_element(By.ID, '1F')
    isF.click()
    isSure = driver.find_element(By.XPATH, '//*[@id="qr_submit_id"]')
    print("请点击确认")
    ret = input("是否成功？")


def job():
    # 定时任务，执行查询车票的操作
    print("执行定时任务···")
    query_train_tickets()
    IsSure()


def main():
    global driver
    driver = webdriver.Edge()
    login()
    query_train_tickets()
    try:
        IsSure()
    except Exception as e:
        driver.close()
        main()


main()