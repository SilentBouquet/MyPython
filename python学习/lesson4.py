# 文件的复制
# 从源文件中读取内容，写入到新路径去
with open('头像.jpeg', mode='rb') as f1, \
        open('../操作文件/少女.jpeg', mode='wb') as f2:
    for line in f1:
        f2.write(line)