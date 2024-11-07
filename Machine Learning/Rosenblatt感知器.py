import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# 感知器对象
class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta      # 学习率
        self.n_iter = n_iter        # 学习次数
        self.random_state = random_state    # 随机种子

    def fit(self, X, y):
        # 随机数生成器
        rgen = np.random.RandomState(self.random_state)
        # 正态分布：其中，loc为均值，scale为标准差，size为输出矩阵的shape（默认为None）
        self.w_ = rgen.normal(loc=0, scale=self.eta, size=X.shape[1])       # 权重
        self.b_ = 0     # 偏置项
        self.errors_ = []       # 分类错误样本

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


def plot_decision_regions(X, y, classifier, resolution=0.02):
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
                    label=f"Class {cl}",
                    cmap=ListedColormap(colors[idx]))


if __name__ == '__main__':
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(URL, header=None, encoding='utf-8')

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)
    X = df.iloc[0:100, [0, 2]].values
    # 绘制散点图
    plt.scatter(X[:50, 0], X[:50, 1], color='green', marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='yellow', marker='s', label='Versicolor')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()

    ppn = Perceptron(eta=0.01, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o', label='Training Error')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show()

    plot_decision_regions(X, y, classifier=ppn, resolution=0.02)
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()