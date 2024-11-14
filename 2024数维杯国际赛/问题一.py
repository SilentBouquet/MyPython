import pykrige
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

# 读取目标变量的值
F1_target_variable_lst = []
with open('F1.txt', 'r') as f:
    while True:
        line = f.readline().strip('\n')
        if not line:
            break
        content = line.split(',')
        f1_target_variable = content[0]
        if f1_target_variable == 'F1_target_variable':
            continue
        F1_target_variable_lst.append(eval(f1_target_variable))
    f.close()

# 原始F1_target_variable矩阵
F1_target_variable = np.array(F1_target_variable_lst)
F1_target_variable = F1_target_variable.reshape(266, 266)
# 原始方格矩阵
Row_spans = np.arange(78750.0, 92000.0 + 50, 13250.0 / 265)
Column_spans = np.arange(51250.0, 64500.0 + 50, 13250.0 / 265)


def cocular_krige(p):
    # 设置随机数种子
    # random.seed(1)
    # 对序号进行随机抽样
    sample_X = random.sample(list(range(0, len(Column_spans))), int(len(Column_spans) * p))
    sample_Y = random.sample(list(range(0, len(Row_spans))), int(len(Row_spans) * p))
    # 抽取对应的方格矩阵元素和目标变量矩阵元素
    data_X = [Column_spans[i] for i in sample_X]
    data_Y = [Row_spans[i] for i in sample_Y]
    data_F1 = []
    for i in range(len(sample_X)):
        x = sample_X[i]
        y = sample_Y[i]
        data_F1.append(F1_target_variable[y][x])

    # 利用克里金插值法进行预测，结果保存到k3d1
    ok3d = pykrige.ok.OrdinaryKriging(data_X, data_Y, data_F1, variogram_model="spherical")
    k3d1, ss3d = ok3d.execute("grid", Column_spans, Row_spans)

    # 计算均方误差
    diff = F1_target_variable - k3d1
    square_diff = np.square(diff)
    mse = np.mean(square_diff)
    # 计算决定系数
    r2 = r2_score(F1_target_variable.ravel(), k3d1.ravel())
    return k3d1, mse


k3d1, mse_07 = cocular_krige(0.7)

# 绘制等高线图
plt.figure(figsize=(12, 6))
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.subplot(1, 2, 1)
xx11, xx21 = np.meshgrid(Column_spans, Row_spans)
ct1 = plt.contourf(xx11, xx21, F1_target_variable, alpha=0.5)
cb1 = plt.colorbar(ct1)
cb1.ax.tick_params(labelsize='small')
cb_yticks1 = cb1.ax.get_yticks()
cb_yticks1 = ['%.2f' % i for i in cb_yticks1]
cb1.ax.set_yticklabels(cb_yticks1, fontproperties='Times New Roman')
plt.xticks(fontproperties='Times New Roman', fontsize='small')
plt.yticks(fontproperties='Times New Roman', fontsize='small')
plt.xlabel(r'$Column\ spans$')
plt.ylabel(r'$Row\ spans$')
plt.title(r'$F1\ target\ variable$')
plt.subplot(1, 2, 2)
xx12, xx22 = np.meshgrid(Column_spans, Row_spans)
ct2 = plt.contourf(xx12, xx22, k3d1, alpha=0.5)
cb2 = plt.colorbar(ct2)
cb2.ax.tick_params(labelsize='small')
cb_yticks2 = cb2.ax.get_yticks()
cb_yticks2 = ['%.2f' % i for i in cb_yticks2]
cb2.ax.set_yticklabels(cb_yticks2, fontproperties='Times New Roman')
plt.xticks(fontproperties='Times New Roman', fontsize='small')
plt.yticks(fontproperties='Times New Roman', fontsize='small')
plt.xlabel(r'$Column\ spans$')
plt.ylabel(r'$Row\ spans$')
plt.title(r'$Contour\ map\ of\ predictions\ by\ Kriging(p=0.7)$')
plt.show()

MSE = []
P = [0.4, 0.5, 0.6, 0.7, 0.8]
for i in P:
    min_Mse = 1
    for j in range(0, 10):
        k, mse = cocular_krige(i)
        if mse < min_Mse:
            min_Mse = mse
    MSE.append(min_Mse)

print(MSE)
plt.figure(figsize=(8, 6))
plt.plot(P, MSE, 'o-', c='steelblue')
plt.xlabel(r'$Sample\ Ratio$')
plt.ylabel('$MSE$')
plt.yticks(fontproperties='Times New Roman', fontsize='small')
plt.xticks(P, fontproperties='Times New Roman', fontsize='small')
plt.show()