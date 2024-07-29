import math
import matplotlib.pyplot as plt
import numpy as np

khmd = []
khzd = []

with open("业务表.csv", "r", encoding="utf-8") as file:
    lines = file.readlines()
    for i in range(1, len(lines)):
        content = lines[i].strip().split(",")
        if content[1].split("-")[0] != "2023":
            continue
        if content[4] not in khmd:
            khmd.append(content[4])
            khzd.append(eval(content[11]))
        else:
            i = khmd.index(content[4])
            khzd[i] = (khzd[i] + eval(content[11]))

kh = []
for i in range(0, len(khzd)):
    value = khzd[i]
    khzd[i] = round(value, 2)
    dic = {
        "khmc": khmd[i],
        "khhk": khzd[i]
    }
    kh.append(dic)

for k in range(0, len(kh)):
    for j in range(0, len(kh) - k - 1):
        if kh[j]["khhk"] < kh[j+1]["khhk"]:
            temp = kh[j+1]
            kh[j+1] = kh[j]
            kh[j] = temp

s = 0
for i in range(0, len(kh)):
    t = float(kh[i]["khhk"])
    s += t
j = s / len(kh)
print(round(j, 2))


labels = []
value = []
impt = math.ceil(len(kh) * 0.1)
for i in range(0, impt):
    a = kh[i]
    labels.append(a["khmc"])
    value.append(float(a["khhk"]))
print(labels)
print(value)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
for i in range(0, len(labels)):
    plt.bar(labels[i], value[i], color="#A1CAF1")
plt.xticks(rotation=90)
plt.title("客户需求量分析")
plt.xlabel('客户名称')
plt.ylabel('元')
plt.savefig('04.png')
plt.show()