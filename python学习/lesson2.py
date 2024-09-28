f = open('../操作文件/我用什么把你留住.txt', mode='r', encoding='utf-8');
print(f.readable())

line = f.readline().strip()     # 去掉字符串两端的空白、空格、换行和制表符
print(line)

line = f.readline()
print(line)

content = f.read()
print(content)

F = open('../操作文件/我用什么把你留住.txt', mode='r', encoding='utf-8')
note = F.readlines()
print(note)
print()

e = open('../操作文件/街道.txt', mode='r', encoding='utf-8')
for i in e:
    print(i.strip())