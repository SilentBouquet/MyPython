import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# 读取目标变量和协同变量的值
F1_target_variable_lst = []
F1_collaborative_variable3_lst = []
F1_collaborative_variable4_lst = []
with open('F1.txt', 'r') as f:
    while True:
        line = f.readline().strip('\n')
        if not line:
            break
        content = line.split(',')
        f1_target_variable = content[0]
        f1_collaborative_variable3 = content[3]
        f1_collaborative_variable4 = content[4]
        if f1_target_variable == 'F1_target_variable':
            continue
        F1_target_variable_lst.append(eval(f1_target_variable))
        F1_collaborative_variable3_lst.append(eval(f1_collaborative_variable3))
        F1_collaborative_variable4_lst.append(eval(f1_collaborative_variable4))
    f.close()

# 原始F1_target_variable数组
F1_target_variable_ = np.array(F1_target_variable_lst)
F1_target_variable = F1_target_variable_.reshape(266, 266)
L = len(F1_target_variable_)
# 原始F1_collaborative_variable3数组
F1_collaborative_variable3 = np.array(F1_collaborative_variable3_lst)
# 原始F1_collaborative_variable4数组
F1_collaborative_variable4 = np.array(F1_collaborative_variable4_lst)
# 原始方格矩阵
Row_spans = np.arange(78750.0, 92000.0 + 50, 13250.0 / 265)
Column_spans = np.arange(51250.0, 64500.0 + 50, 13250.0 / 265)
X = np.array([[F1_collaborative_variable3[i], F1_collaborative_variable4[i]]
              for i in range(len(F1_collaborative_variable3))])


def RandomForest(p):
    # 对序号进行随机抽样
    sample_x = random.sample(list(range(0, L)), int(L * p))
    x = []
    y = []
    for i in sample_x:
        x.append([F1_collaborative_variable3[i], F1_collaborative_variable4[i]])
        y.append(F1_target_variable_[i])
    x = np.array(x)
    y = np.array(y)

    # 随机森林回归模型
    forest = RandomForestRegressor(
        n_estimators=500,
        criterion='squared_error',
        random_state=None,
        n_jobs=-1
    )
    forest.fit(x, y)

    # 预测目标变量的值
    y_pred = forest.predict(X)
    y_pred = y_pred.reshape(266, 266)
    # print("F1_target_variable_pred:", y_pred)

    # 计算均方误差
    diff = F1_target_variable - y_pred
    square_diff = np.square(diff)
    mse = np.mean(square_diff)
    # 计算决定系数
    r2 = r2_score(F1_target_variable.ravel(), y_pred.ravel())
    return y_pred, mse, r2


y_pred_07, mse_07, r2_07 = RandomForest(0.7)
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
ct2 = plt.contourf(xx12, xx22, y_pred_07, alpha=0.5)
cb2 = plt.colorbar(ct2)
cb2.ax.tick_params(labelsize='small')
cb_yticks2 = cb2.ax.get_yticks()
cb_yticks2 = ['%.2f' % i for i in cb_yticks2]
cb2.ax.set_yticklabels(cb_yticks2, fontproperties='Times New Roman')
plt.xticks(fontproperties='Times New Roman', fontsize='small')
plt.yticks(fontproperties='Times New Roman', fontsize='small')
plt.xlabel(r'$Column\ spans$')
plt.ylabel(r'$Row\ spans$')
plt.title(r'$Contour\ map\ of\ predictions\ by\ RandomForestRegressor(p=0.7)$')
plt.show()

MSE = []
P = [0.4, 0.5, 0.6, 0.7, 0.8]
for i in P:
    k, mse, r2 = RandomForest(i)
    MSE.append(mse)

print(MSE)
plt.figure(figsize=(8, 6))
plt.plot(P, MSE, 'o-', c='steelblue')
plt.xlabel(r'$Sample\ Ratio$')
plt.ylabel('$MSE$')
plt.yticks(fontproperties='Times New Roman', fontsize='small')
plt.xticks(P, fontproperties='Times New Roman', fontsize='small')
plt.show()