import matplotlib.pyplot as plt

sum1 = 0
sum2 = 0
weight1 = 0
weight2 = 0

with open("业务表.csv", "r", encoding="utf-8") as file:
    lines = file.readlines()
    for i in range(1, len(lines)):
        content = lines[i].strip().split(",")
        if content[1].split("-")[0] != "2023":
            continue
        if content[5] == "0":
            sum1 += eval(content[11])
            weight1 += eval(content[8])
        elif content[5] == "1":
            sum2 += eval(content[11])
            weight2 += eval(content[8])
        else:
            continue

dj1 = sum1 / weight1
dj2 = sum2 / weight2

print(f"去年水泥的总销售额是{sum1:.2f}元，矿粉的总销售额是{sum2:.2f}元")
print(f"去年水泥的总销售量是{weight1:.2f}吨，矿粉的总销售量是{weight2:.2f}吨")
print(f"去年水泥的平均单价是{dj1:.2f}元/吨，矿粉的平均单价是{dj2:.2f}元/吨")

X = ["水泥", "矿粉"]
sum3 = [round(sum1, 2), round(sum2, 2)]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.bar(X[0], sum3[0], color="#A1CAF1")
plt.bar(X[1], sum3[1], color="#ADD8E6")
plt.title("总销售额")
plt.xlabel('货品类型')
plt.ylabel('元')
plt.savefig('01.png')
plt.show()

sum3 = [round(weight1, 2), round(weight2, 2)]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.bar(X[0], sum3[0], color="#A1CAF1")
plt.bar(X[1], sum3[1], color="#ADD8E6")
plt.title("总销售额量")
plt.xlabel('货品类型')
plt.ylabel('吨')
plt.savefig('02.png')
plt.show()

sum3 = [round(dj1, 2), round(dj2, 2)]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.bar(X[0], sum3[0], color="#A1CAF1")
plt.bar(X[1], sum3[1], color="#ADD8E6")
plt.title("平均单价")
plt.xlabel('货品类型')
plt.ylabel('元/吨')
plt.savefig('03.png')
plt.show()

s = sum1 + sum2
w = weight1 + weight2
s1 = round(sum1 / s, 2)
s2 = round(sum2 / s, 2)
w1 = round(weight1 / w, 2)
w2 = round(weight2 / w, 2)
print(f"其中，水泥的占比为{s1*100}%，矿粉的占比为{s2*100}%")
print(f"其中，水泥的占比为{w1*100}%，矿粉的占比为{w2*100}%")