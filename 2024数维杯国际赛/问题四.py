import pykrige
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 存储变量名
Variable = ['F2_target_variable', 'F2_collaborative_variable1', 'F2_collaborative_variable2',
            'F2_collaborative_variable3', 'F2_collaborative_variable4']

# 存储数据的文件地址
Path_F2_target_variable = r"C:\Users\21336\Documents\WXWork\1688858313404405\Cache\File\2024-11\F2_target_variable.xlsx"
Path = [r"C:\Users\21336\Documents\WXWork\1688858313404405\Cache\File\2024-11\F2_collaborative_variable1.txt",
        r"C:\Users\21336\Documents\WXWork\1688858313404405\Cache\File\2024-11\F2_collaborative_variable2.txt",
        r"C:\Users\21336\Documents\WXWork\1688858313404405\Cache\File\2024-11\F2_collaborative_variable3.txt",
        r"C:\Users\21336\Documents\WXWork\1688858313404405\Cache\File\2024-11\F2_collaborative_variable4.txt"]

# 读取数据
F2_ = []
for path in Path:
    with (open(path, "r")) as f:
        cnt = 0
        L = []
        while True:
            content = f.readline().strip('\n')
            if cnt >= 6:
                if not content:
                    break
                if path != Path[0]:
                    lst = content.split('         ')
                else:
                    lst = content.split('       ')
                    if len(lst) != 6:
                        lst = content.split('      ')
                for i in range(1, len(lst)):
                    L.append(eval(lst[i]))
            cnt += 1
    f.close()
    F2_.append(L)

# 将四个变量的数据一一对应，返回一个列表
F2 = []
for i in range(0, len(F2_[0])):
    F = []
    for j in range(0, len(F2_)):
        F.append(F2_[j][i])
    F2.append(F)

# 原始方格矩阵
Row_spans = np.arange(78750.0, 92000.0 + 50, 13250.0 / 265)
Column_spans = np.arange(51250.0, 64500.0 + 50, 13250.0 / 265)
df = pd.read_excel(Path_F2_target_variable, header=None).values
Row_index = df[1:, 0]
Col_index = df[1:, 1]
Target = df[1:, 4]
F2 = np.array(F2)
F2 = F2.reshape(266, 266, 4)

X = []
data_X = []
data_Y = []
for i in range(0, len(Row_index)):
    c = Col_index[i]
    r = Row_index[i]
    data_X.append(Column_spans[c])
    data_Y.append(Row_spans[r])
    X.append(list(F2[r][c]))
y = Target

# 随机森林回归模型
forest = RandomForestRegressor(
    n_estimators=1000,
    criterion='squared_error',
    random_state=None,
    n_jobs=-1
)
forest.fit(X, y)

# 利用随机森林回归模型进行预测，结果保存到y_pred1
F2 = F2.reshape(266 * 266, 4)
y_pred1 = forest.predict(F2)
y_pred1 = y_pred1.reshape(266, 266)
print("y_pred_by_randomforestregressor:\n", y_pred1)

# 利用克里金插值法进行预测，结果保存到y_pred2
ok3d = pykrige.ok.OrdinaryKriging(data_X, data_Y, y, variogram_model="spherical")
y_pred2, ss3d = ok3d.execute("grid", Column_spans, Row_spans)
print('y_pred_by_kriging:\n', y_pred2)

# 绘制等高线图
plt.figure(figsize=(12, 6))
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.subplot(1, 2, 1)
xx11, xx21 = np.meshgrid(Column_spans, Row_spans)
ct1 = plt.contourf(xx11, xx21, y_pred1, alpha=0.5)
cb1 = plt.colorbar(ct1)
cb1.ax.tick_params(labelsize='small')
cb_yticks1 = cb1.ax.get_yticks()
cb_yticks1 = ['%.2f' % i for i in cb_yticks1]
cb1.ax.set_yticklabels(cb_yticks1, fontproperties='Times New Roman')
plt.xticks(fontproperties='Times New Roman', fontsize='small')
plt.yticks(fontproperties='Times New Roman', fontsize='small')
plt.xlabel(r'$Column\ spans$')
plt.ylabel(r'$Row\ spans$')
plt.title(r'$Contour\ map\ of\ predictions\ by\ RandomForestRegressor$')
plt.subplot(1, 2, 2)
xx12, xx22 = np.meshgrid(Column_spans, Row_spans)
ct2 = plt.contourf(xx12, xx22, y_pred2, alpha=0.5)
cb2 = plt.colorbar(ct2)
cb2.ax.tick_params(labelsize='small')
cb_yticks2 = cb2.ax.get_yticks()
cb_yticks2 = ['%.2f' % i for i in cb_yticks2]
cb2.ax.set_yticklabels(cb_yticks2, fontproperties='Times New Roman')
plt.xticks(fontproperties='Times New Roman', fontsize='small')
plt.yticks(fontproperties='Times New Roman', fontsize='small')
plt.xlabel(r'$Column\ spans$')
plt.ylabel(r'$Row\ spans$')
plt.title(r'$Contour\ map\ of\ predictions\ by\ Kriging$')
plt.show()