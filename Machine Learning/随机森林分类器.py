import sklearn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    # 将数据拆分为30%的测试数据和70%的训练数据
    # stratify=y表示内置分层支持
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)
    # 在竖直方向上堆叠
    X_combined = np.vstack((X_test, X_train))
    # 在水平方向上平铺
    y_combined = np.hstack((y_train, y_test))

    # n_estimators表示组成随机森林的决策树的数量，默认情况下使用基尼指数作为分类标准
    # n_jobs表示允许多少个计算机的内核并行训练模型
    forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
    forest.fit(X_train, y_train)
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    plot_decision_regions(X_combined, y_combined, clf=forest)
    plt.xlabel(r'$Petal\ length\ [standardized]$')
    plt.ylabel(r'$Petal\ width\ [standardized]$')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()