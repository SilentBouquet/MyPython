lst = ['马尔克斯', '博尔赫斯', '富恩特斯', '聂鲁达']
f = open('../操作文件/拉美文学.txt', mode='w', encoding='utf-8')
# 大多数情况下要把open写在循环外面
for i in lst:
    f.write(i)
    f.write('\n')
f.close()

with open('../操作文件/拉美文学.txt', mode='a', encoding='utf-8') as e:
    e.write('鲁尔福')
    e.close()

with open('头像.jpeg', mode='rb') as d:
    for line in d:
        print(line)