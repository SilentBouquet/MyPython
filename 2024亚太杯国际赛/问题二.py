import sklearn
import kaiwu as kw
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


kw.license.init(user_id="51648452576182274", sdk_code="AvuHse0ji1LEsAMWjI1Y3u7igbvxxF")


def Decision_Function_by_QUBO(label):
    # 定义类标记集合
    Label = [0, 1, 2]
    Label.remove(label)
    label1 = Label[0]
    label2 = Label[1]
    # 导入鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # 选取两种类标记
    X = np.array([X[i] for i in range(len(y)) if y[i] != label])
    y = np.array([y[i] for i in range(len(y)) if y[i] != label])
    # 将标签转换成1和-1
    y_ = []
    for i in range(len(y)):
        if y[i] == label2:
            y_.append(1)
        else:
            y_.append(-1)
    y_ = np.array(y_)
    # 将数据拆分为30%的测试数据和70%的训练数据
    # stratify=y_表示内置分层支持
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y_, test_size=0.3, random_state=1, stratify=y_)

    N = len(X_train)
    # 定义精确度
    k = 6
    # 定义拉格朗日乘子alpha的阈值
    alpha_min = 0
    alpha_max = 5
    # 计算alpha的间隔
    delta_alpha = (alpha_max - alpha_min) / (2 ** k - 1)
    coef = np.array([2 ** i for i in range(k)])
    # 定义每个数据点上的alpha表达式
    alpha_list = []
    for i in range(N):
        x_alpha = kw.qubo.ndarray(k, 'x_alpha_{}'.format(i + 1), kw.qubo.Binary)
        alpha = alpha_min + delta_alpha * (x_alpha @ coef)
        alpha_list.append(alpha)

    # 定义拉格朗日函数L
    L = 0
    for i in range(N):
        alpha_i = alpha_list[i]
        for j in range(N):
            alpha_j = alpha_list[j]
            s = 0.5 * (y_train[i] * y_train[j] * (X_train[i] @ X_train[j].T) * (alpha_i * alpha_j))
            L += s
        L -= alpha_i

    # 定义等式约束
    cons = 0
    for i in range(N):
        cons += alpha_list[i] * y_train[i]
    # 定义最小罚因子
    sigma = 15
    # 定义罚函数
    P = sigma * cons ** 2
    L_ = L + P

    # 求解QUBO模型
    q = kw.qubo.make(L_)
    ising = kw.qubo.qubo_model_to_ising_model(q)
    qm = kw.qubo.qubo_model_to_qubo_matrix(q)['qubo_matrix']
    print("Q矩阵为：\n", qm)
    im = kw.qubo.qubo_matrix_to_ising_matrix(qm)[0]
    # 得到解向量
    sm_min = 0
    delta_min = 10000
    for i in range(10):
        worker = kw.classical.TabuSearchOptimizer(100, size_limit=1)
        sm = np.array(worker.solve(im))[0]
        var = ising.get_variables()
        sol_dict = kw.qubo.get_sol_dict(sm, var)
        delta = kw.qubo.get_val(q, sol_dict)
        if delta < delta_min:
            delta_min = delta
            sm_min = sm

    # 将ising变量转变成qubo变量
    for i in range(len(sm_min)):
        if sm_min[i] < 0:
            sm_min[i] = 0

    print(len(sm_min))
    print("误差最小时的解向量：\n", sm_min)

    # 计算拉格朗日乘子项列表
    Alpha_list = []
    cnt = 0
    for i in range(N):
        alpha_list = [sm_min[j] for j in range(cnt, cnt + k)]
        alpha = alpha_min + delta_alpha * (np.array(alpha_list) @ coef)
        Alpha_list.append(alpha)
        cnt += k
    print('拉格朗日乘子项如下：\n', Alpha_list)

    # 计算超平面的斜率
    w = 0
    for i in range(N):
        w += Alpha_list[i] * y_train[i] * X_train[i]

    # 计算超平面的截距
    j = 0
    for i in range(N):
        if Alpha_list[i] > 0:
            j = i
        break
    b = y_train[j]
    for i in range(N):
        b -= Alpha_list[i] * y_train[i] * (X_train[i] @ X_train[j].T)

    print('分离超平面为w * x + b = 0，其中，w为{}，b为{}'.format(w, b))

    # 定义决策函数
    def sign(x):
        if np.array(w) @ x + b > 0:
            return label2
        else:
            return label1

    # 统计分类正确的样本
    cnt = 0
    for i in range(len(X)):
        predict = sign(X[i].T)
        if y[i] == predict:
            cnt += 1

    # 计算在原数据集上分类的准确率
    print('对于标签{}和{}的分类问题，qubo模型的分类准确率为：'.format(label1, label2), cnt / len(X))


def Decision_Function_by_SVM(label):
    # 定义类标记集合
    Label = [0, 1, 2]
    Label.remove(label)
    label1 = Label[0]
    label2 = Label[1]
    # 导入鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # 选取两种类标记
    X = np.array([X[i] for i in range(len(y)) if y[i] != label])
    y = np.array([y[i] for i in range(len(y)) if y[i] != label])
    # 将数据拆分为30%的测试数据和70%的训练数据
    # stratify=y_表示内置分层支持
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)
    svm = SVC(kernel='linear', C=1.0, random_state=1)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X)
    print('对于标签{}和{}的分类问题，SVM模型的分类准确率为：'.format(label1, label2), accuracy_score(y, y_pred))


Decision_Function_by_QUBO(0)
Decision_Function_by_QUBO(1)
Decision_Function_by_QUBO(2)
Decision_Function_by_SVM(0)
Decision_Function_by_SVM(1)
Decision_Function_by_SVM(2)