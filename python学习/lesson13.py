# 推导式
lst = [i for i in range(10)]
print(lst)

lst1 = [i for i in range(1, 10, 2)]
lst2 = [i for i in range(1, 10) if i % 2 == 1]
print(lst1)
print(lst2)

lst3 = ['allen', 'tony', 'kevin', 'jory']
lst4 = [item.upper() for item in lst3]
print(lst4)

s = {i for i in range(10)}
print(s)

lst5 = ['康德', '费希特', '谢林', '黑格尔', '荷尔德林']
dic = {i+1: lst5[i] for i in range(len(lst5))}
print(dic)

gen = (i**2 for i in range(10))
lst6 = list(gen)
print(lst6)