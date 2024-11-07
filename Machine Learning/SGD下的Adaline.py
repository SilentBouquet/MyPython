import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


# 自适应线性神经元
class AdalineSGD:
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=1):
        self.eta = eta      # 学习率
        self.n_iter = n_iter        # 学习次数
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state    # 随机种子

    def fit(self, X, y):
        self.initialize_weights(X.shape[1])
        self.losses_ = []       # 分类错误样本

        # 随机梯度下降法（SGD）
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self.Shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self.upgrate_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self.initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self.upgrate_weights(xi, target)
        else:
            self.upgrate_weights(X, y)
        return self

    def Shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0, scale=0.01, size=m)
        self.b_ = 0
        self.w_initialized = True

    def upgrate_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * error * xi * 2.0
        self.b_ += self.eta * error * 2.0
        loss = error ** 2
        return loss

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
    ada = AdalineSGD(eta=0.01, n_iter=20).fit(X_std, y)
    plot_decision_regions(X_std, y, clf=ada)
    plt.xlabel('Sepal length [standardized]')
    plt.ylabel('Petal length [standardized]')
    plt.legend(loc='upper left')
    plt.title('Adaline - Stochastic gradient descent')
    plt.tight_layout()
    plt.show()

    plt.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.tight_layout()
    plt.show()