import matplotlib.pyplot as plt


def count(x):
    yw2022 = []
    yw2023 = []
    yw2024 = []
    month22 = []
    month23 = []
    month24 = []
    with open("业务表.csv", "r", encoding="utf-8") as file:
        lines = file.readlines()
        for i in range(1, len(lines)):
            content = lines[i].strip().split(",")
            if content[5] != x:
                continue
            year = content[1].split("-")[0]
            if year == "2022":
                m = content[1].split("-")[1]
                if m == "07":
                    continue
                if m not in month22:
                    month22.append(m)
                    yw2022.append(eval(content[11]))
                else:
                    i = month22.index(m)
                    yw2022[i] = (yw2022[i] + eval(content[11]))
            elif year == "2023":
                m = content[1].split("-")[1]
                if m not in month23:
                    month23.append(m)
                    yw2023.append(eval(content[11]))
                else:
                    i = month23.index(m)
                    yw2023[i] = (yw2023[i] + eval(content[11]))
            else:
                m = content[1].split("-")[1]
                if m == "06":
                    break
                if m not in month24:
                    month24.append(m)
                    yw2024.append(eval(content[11]))
                else:
                    i = month24.index(m)
                    yw2024[i] = (yw2024[i] + eval(content[11]))
    yw = []
    for i in range(0, len(yw2022)):
        yw.append(yw2022[i])
    for i in range(0, len(yw2023)):
        yw.append(yw2023[i])
    for i in range(0, len(yw2024)):
        yw.append(yw2024[i])
    for i in range(0, len(yw)):
        value = yw[i]
        yw[i] = round(value, 2)
    return yw


yw1 = count("0")
yw2 = count("1")
yw1.append(0)
for i in range(1, 4):
    k = len(yw1)
    temp = yw1[k - i]
    yw1[k - i] = yw1[k - i - 1]
    yw1[k - i - 1] = temp
print(yw1)
print(yw2)

X = []
for i in range(1, len(yw1) + 1):
    X.append(i)
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.plot(X, yw1, label='水泥')
plt.plot(X, yw2, label='矿粉')
plt.title("月销售量")
plt.xlabel('月份数')
plt.ylabel('元')
plt.legend()
plt.savefig('06.png')
plt.show()