import sklearn
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions

if __name__ == '__main__':
    # 设置随机数生成器种子
    np.random.seed(1)
    # 返回一个200行2列的矩阵，具有标准正态分布
    X_xor = np.random.randn(200, 2)
    # 对每个元素进行 异或 运算，返回 bool 值组成的数组
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor > 0, 1, -1)
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1],
                c='tomato', marker='^', label=r'$Class\ 0$')
    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
                c='royalblue', marker='s', label=r'$Class\ 1$')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel(r'$Feature\ 1$')
    plt.ylabel(r'$Feature\ 2$')
    plt.title(r'$XOR$数据集')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # gamma参数为高斯球面的截止参数，增大则会增加训练样本的影响力，导致更严格的决策边界
    svm = SVC(kernel='rbf', C=10.0, random_state=1, gamma=0.15)
    svm.fit(X_xor, y_xor)
    plot_decision_regions(X_xor, y_xor, clf=svm)
    plt.title('使用$XOR$数据核支持向量机的决策边界')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    # 将数据拆分为30%的测试数据和70%的训练数据
    # stratify=y表示内置分层支持
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)
    # 特征标准化
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    # 在竖直方向上堆叠
    X_combined_std = np.vstack((X_test_std, X_train_std))
    # 在水平方向上平铺
    y_combined = np.hstack((y_train, y_test))
    svm = SVC(kernel='rbf', C=1, random_state=1, gamma=0.2)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined, clf=svm)
    plt.xlabel(r'$Petal\ length\ [standardized]$')
    plt.ylabel(r'$Petal\ width\ [standardized]$')
    plt.title('高斯核支持向量机模型在鸢尾花数据集上的决策边界')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()