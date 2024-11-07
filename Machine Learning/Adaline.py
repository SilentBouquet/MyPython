import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


# 自适应线性神经元
class AdalineGD:
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
        self.losses_ = []       # 分类错误样本

        # 全批梯度下降法
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * errors.mean()     # mean函数用于求均值
            loss = (errors ** 2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


if __name__ == '__main__':
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(URL, header=None, encoding='utf-8')
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)
    X = df.iloc[0:100, [0, 2]].values

    # 特征缩放
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    fig = plt.figure(figsize=(10, 5))
    ada = AdalineGD(eta=0.5, n_iter=20).fit(X_std, y)
    plot_decision_regions(X_std, y, clf=ada)
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    plt.legend(loc='upper left')
    plt.title('Adaline - Gradient descent')
    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error')
    plt.tight_layout()
    plt.show()