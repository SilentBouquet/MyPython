import math
import matplotlib.pyplot as plt

xsmd = []
xszd = []

with open("业务表.csv", "r", encoding="utf-8") as file:
    lines = file.readlines()
    for i in range(1, len(lines)):
        content = lines[i].strip().split(",")
        if content[1].split("-")[0] != "2023":
            continue
        if content[2] not in xsmd:
            xsmd.append(content[2])
            xszd.append(eval(content[11]))
        else:
            i = xsmd.index(content[2])
            xszd[i] = (xszd[i] + eval(content[11]))

xs = []
for i in range(0, len(xszd)):
    value = xszd[i]
    xszd[i] = round(value, 2)
    dic = {
        "xsmc": xsmd[i],
        "xshk": xszd[i]
    }
    xs.append(dic)

for k in range(0, len(xs)):
    for j in range(0, len(xs) - k - 1):
        if xs[j]["xshk"] < xs[j+1]["xshk"]:
            temp = xs[j+1]
            xs[j+1] = xs[j]
            xs[j] = temp

print(xs)
s = 0
for i in range(0, len(xs)):
    t = float(xs[i]["xshk"])
    s += t
j = s / len(xs)
print(round(j, 2))
labels = []
value = []
for i in range(0, len(xs)):
    a = xs[i]
    labels.append(a["xsmc"])
    value.append(float(a["xshk"]))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
for i in range(0, len(labels)):
    plt.bar(labels[i], value[i], color="#ADD8E6")
plt.xticks(rotation=-30)
plt.title("销售人员业绩分析")
plt.xlabel('销售人员')
plt.ylabel('元')
plt.savefig('05.png')
plt.show()