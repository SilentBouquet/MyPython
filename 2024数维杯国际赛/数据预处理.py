# 存储变量名
Variable = ['F1_target_variable', 'F1_collaborative_variable1', 'F1_collaborative_variable2',
            'F1_collaborative_variable3', 'F1_collaborative_variable4']

# 存储数据的文件地址
Path = [r"C:\Users\21336\Documents\WXWork\1688858313404405\Cache\File\2024-11\F1_target_variable.txt",
        r"C:\Users\21336\Documents\WXWork\1688858313404405\Cache\File\2024-11\F1_collaborative_variable1.txt",
        r"C:\Users\21336\Documents\WXWork\1688858313404405\Cache\File\2024-11\F1_collaborative_variable2.txt",
        r"C:\Users\21336\Documents\WXWork\1688858313404405\Cache\File\2024-11\F1_collaborative_variable3.txt",
        r"C:\Users\21336\Documents\WXWork\1688858313404405\Cache\File\2024-11\F1_collaborative_variable4.txt"]

# 读取数据
F1_ = []
for path in Path:
    with (open(path, "r")) as f:
        cnt = 0
        L = []
        while True:
            content = f.readline().strip('\n')
            if cnt >= 6:
                if not content:
                    break
                if path != Path[1]:
                    lst = content.split('         ')
                else:
                    lst = content.split('      ')
                for i in range(1, len(lst)):
                    L.append(eval(lst[i]))
            cnt += 1
    f.close()
    F1_.append(L)

# 将五个变量的数据一一对应，返回一个列表
F1 = []
for i in range(0, len(F1_[0])):
    F = []
    for j in range(0, len(F1_)):
        F.append(F1_[j][i])
    F1.append(F)

# 将数据写入文件
with open('F1.txt', 'w') as f:
    f.write(','.join(Variable) + '\n')
    for i in range(0, len(F1)):
        content = ','.join(map(str, F1[i])) + '\n'
        f.write(content)
        print('第{}次over!'.format(i+1))
    f.close()