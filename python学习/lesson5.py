# 文件的修改
import os
import time
with open('../操作文件/人名单.txt', mode='r', encoding='utf-8 ') as f1, \
        open('../操作文件/人名单_副本.txt', mode='w', encoding='utf-8') as f2:
    for line in f1:
        line = line.strip()
        if line.startswith('周'):
            line = line.replace('周', '张')

        f2.write(line)
        f2.write('\n')

time.sleep(3)
# 删除源文件
os.remove('../操作文件/人名单.txt')
time.sleep(3)
# 把副本文件重命名成源文件
os.rename('../操作文件/人名单_副本.txt', '../操作文件/人名单.txt')