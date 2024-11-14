import sklearn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def plot_sigmoid():
    z = np.arange(-7.0, 7.0, 0.1)
    sigma_z = sigmoid(z)
    plt.plot(z, sigma_z)
    # 绘制垂直线，标记出y轴
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$sigma(z)$')
    plt.title(r'$logistic\ sigmoid$')
    # 设置ytick的间隔
    plt.yticks([0.0, 0.5, 1.0])
    # 获取当前所有坐标轴对象
    ax = plt.gca()
    # 显示水平网格线
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()


def loss_1(z):
    return -1.0 * np.log(sigmoid(z))


def loss_0(z):
    return -1.0 * np.log(1.0 - sigmoid(z))


def plot_loss_function():
    z = np.arange(-10, 10, 0.1)
    sigma_z = sigmoid(z)
    c1 = [loss_1(x) for x in z]
    plt.plot(sigma_z, c1, label=r'$L(w,\ b)\ if\ y=1$')
    c0 = [loss_0(x) for x in z]
    plt.plot(sigma_z, c0, linestyle='--', label=r'$L(w,\ b)\ if\ y=0$')
    plt.xlabel(r'$sigma(z)$')
    plt.ylabel(r'$L(w,\ b)$')
    plt.ylim(0.0, 5.1)
    plt.xlim(0, 1)
    plt.title('逻辑回归损失函数图')
    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.show()


# 逻辑回归(全批梯度下降)
class LogisticRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta  # 学习率
        self.n_iter = n_iter  # 学习次数
        self.random_state = random_state  # 随机种子

    def fit(self, X, y):
        # 随机数生成器
        rgen = np.random.RandomState(self.random_state)
        # 正态分布：其中，loc为均值，scale为标准差，size为输出矩阵的shape（默认为None）
        self.w_ = rgen.normal(loc=0, scale=self.eta, size=X.shape[1])  # 权重
        self.b_ = 0  # 偏置项
        self.losses_ = []  # 分类错误样本

        # 全批梯度下降法
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * errors.mean()  # mean函数用于求均值
            loss = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1.0 - output))) / X.shape[0])
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        # clip函数是对元素进行截取赋值，超过端点的都赋值为端点值
        return 1. / (1 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'cm'
    plot_sigmoid()
    plot_loss_function()

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

    X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    Irgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
    Irgd.fit(X_train_01_subset, y_train_01_subset)
    plot_decision_regions(X_train_01_subset, y_train_01_subset, clf=Irgd)
    plt.xlabel(r'$Petal\ length\ [standardized]$')
    plt.ylabel(r'$Petal\ width\ [standardized]$')
    plt.legend(loc='upper left')
    plt.title('逻辑回归模型的决策区域')
    plt.tight_layout()
    plt.show()