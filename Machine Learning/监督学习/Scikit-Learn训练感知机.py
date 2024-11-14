import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    # 绘制等势图
    plt.contourf(xx1, xx2, lab, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx],
                    label=rf"$Class\ {cl}$",
                    edgecolors='black')

    # 标记出测试数据
    # if test_idx:
    #     X_test, y_test = X[test_idx], y[test_idx]
    #     plt.scatter(X_test[:, 0], X_test[:, 1], c='none',
    #                 alpha=1.0, edgecolors='black',
    #                 linewidth=1, marker='o', s=100, label='Test set')


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

    # 训练感知机
    ppn = Perceptron(eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)

    # 预测
    y_pred = ppn.predict(X_test_std)
    print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))

    # 在竖直方向上堆叠
    X_combined_std = np.vstack((X_test_std, X_train_std))
    # 在水平方向上平铺
    y_combined = np.hstack((y_train, y_test))
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    plot_decision_regions(X_combined_std, y_combined,
                          classifier=ppn, test_idx=range(105, 150))
    plt.xlabel(r'$Petal\ length\ [standardized]$')
    plt.ylabel(r'$Petal\ width\ [standardized]$')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()