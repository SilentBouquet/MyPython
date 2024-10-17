import logging
import os

if os.path.exists("../操作文件/基础日志.txt"):
    os.remove("../操作文件/基础日志.txt")
if os.path.exists("../操作文件/l1.txt"):
    os.remove("../操作文件/l1.txt")
if os.path.exists("../操作文件/l2.txt"):
    os.remove("../操作文件/l2.txt")

# filename：文件名
# format：数据的格式化输出，最终在日志文件中的样子
# 时间-名称-级别-模块：错误信息
# datefmt：时间的格式
# level：错误的级别权重，当错误的级别权重大于等于level时才会写入日志
logging.basicConfig(filename="../操作文件/基础日志.txt",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s：%(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S',
                    level=10, encoding='utf-8')
# 当前配置表示10分以上的分数会被写入日志

# logging日志记录
logging.critical("今天系统炸了，请程序员来测试")          # 最高级别的日志信息：50
logging.error("一般指的是普通的程序错误，俗称bug")         # 40
logging.warning("我只是一个警告信息")        # 30
logging.info("我只是一个普通的消息")          # 20
logging.debug("默认最低等级的消息")          # 10

# 如果想要把日志记录在不同的文件中：
# 创建一个操作日志的对象logger（依赖FileHandler）
file_handler = logging.FileHandler("../操作文件/l1.txt", "a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s：%(message)s"))

# 创建日志对象
logger1 = logging.Logger("财务系统", level=logging.ERROR)
# 给日志对象设置文件信息
logger1.addHandler(file_handler)

# 再创建一个操作日志的对象logger（依赖FileHandler）
file_handler2 = logging.FileHandler("../操作文件/l2.txt", "a", encoding="utf-8")
file_handler2.setFormatter(logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s：%(message)s"))

# 创建日志对象
logger2 = logging.Logger("会计系统", level=logging.ERROR)
# 给日志对象设置文件信息
logger2.addHandler(file_handler2)

# 项目1：财务系统出错了
logger1.error("财务系统出错了，请程序员来修复")
# 项目2：会计系统出错了
logger2.error("会计系统出错了，领导出来溜达溜达")