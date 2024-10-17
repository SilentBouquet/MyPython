import traceback
import logging
import os

if os.path.exists("../操作文件/traceback日志.txt"):
    os.remove("../操作文件/traceback日志.txt")

logging.basicConfig(filename="../操作文件/traceback日志.txt",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s：%(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S',
                    level=10, encoding='utf-8')

try:
    print(1/0)
except ZeroDivisionError as z:
    print("出错了")
    logging.error(traceback.format_exc())