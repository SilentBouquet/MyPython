import sklearn
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions

if __name__ == '__main__':
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

    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree.fit(X_train_std, y_train)
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    plot_decision_regions(X_combined_std, y_combined, clf=tree)
    plt.xlabel(r'$Petal\ length\ [standardized]$')
    plt.ylabel(r'$Petal\ width\ [standardized]$')
    plt.title('使用鸢尾花数据构建的决策树决策边界')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()