import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 加载MNIST数据集
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values
y = y.astype(int).values

# 数据归一化
X = (X / 255. - .5) * 2

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'

# 绘制不同手写数字的代表性图像
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
count = 0
for i in ax:
    if count == 0:
        i.set_xticks([])
        i.set_yticks([])
    img = X[y == count][0].reshape(28, 28)
    i.imshow(img, cmap="Greys")
    count += 1
plt.tight_layout()
plt.show()

# 绘制同一数字的多个手写样本
count = 0
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in ax:
    if count == 0:
        i.set_xticks([])
        i.set_yticks([])
    img = X[y == 7][count].reshape(28, 28)
    i.imshow(img, cmap="Greys")
    count += 1
plt.tight_layout()
plt.show()

# 将数据集划分为训练集、验证集和测试集
X_temp, X_test, y_temp, y_test = train_test_split(X, y,
                                                  test_size=10000, random_state=123, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp,
                                                      test_size=5000, random_state=123, stratify=y_temp)


# 定义激活函数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# 将整数类别标签转化为独热编码向量
def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary


# 实现多层感知机
class NeuralNetMLP:
    def __init__(self, num_features, num_hidden,
                 num_classes, random_seed=123):
        super().__init__()  # 调用父类的初始化方法
        self.num_classes = num_classes

        # 隐藏层
        rng = np.random.RandomState(random_seed)
        self.weight_h = rng.normal(
            loc=0.0, scale=1.0, size=(num_hidden, num_features)
        )
        self.bias_h = np.zeros(num_hidden)

        # 输出层
        self.weight_out = rng.normal(
            loc=0.0, scale=1.0, size=(num_classes, num_hidden)
        )
        self.bias_out = np.zeros(num_classes)

    # 前向传播法
    def forward(self, x):
        # 隐藏层
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # 输出层
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    # 反向传播法
    def backward(self, x, a_h, a_out, y):
        # 独热编码
        y_onehot = int_to_onehot(y, self.num_classes)

        # 输出层的权重梯度
        d_loss__d_a_out = 2. * (a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1. - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out
        d_z_out__dw_out = a_h
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        # 隐藏层的权重梯度
        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        d_a_h__d_z_h = a_h * (1. - a_h)
        d_z_h__d_w_h = x
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out,
                d_loss__d_w_h, d_loss__d_b_h)


model = NeuralNetMLP(num_features=28 * 28,
                     num_hidden=50,
                     num_classes=10)

# 定义数据加载器
num_epochs = 50
minibatch_size = 100


# 小批量生成器
def minibatch_generator(X, y, minibatch_size):
    # 该数组用于随机访问X中的样本
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        # 得到当前小批量的索引
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        # yield返回两个值：对应于batch_idx索引的特征数据X[batch_idx]和标签数据y[batch_idx]
        yield X[batch_idx], y[batch_idx]


"""
# 输出小批量数据的维度
for i in range(num_epochs):
    minibatch_gen = minibatch_generator(
        X_train, y_train, minibatch_size)
    for X_train_mini, y_train_mini in minibatch_gen:
        break
    break
print(X_train_mini.shape)
print(y_train_mini.shape)
"""


# 定义均方误差损失函数
def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas) ** 2)


# 定义准确率函数
def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)


_, probas = model.forward(X_valid)
mse = mse_loss(y_valid, probas)
# 沿行寻找最大值，并返回索引
predicted_labels = np.argmax(probas, axis=1)
acc = accuracy(y_valid, predicted_labels)

# 未训练前的均方误差和准确率
print(f'Initial validation MSE: {mse:.1f}')
print(f'Initial validation accuracy: {acc * 100:.1f}%')


# 迭代小批量数据逐步计算均方误差和准确率
def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)

    i = 0
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)

        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas) ** 2)
        correct_pred += np.array(predicted_labels == targets).sum()

        num_examples += targets.shape[0]
        mse += loss

    mse = mse / (i + 1)
    acc = correct_pred / num_examples
    return mse, acc


mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f'Initial valid MSE: {mse:.1f}')
print(f'Initial valid accuracy: {acc * 100:.1f}%')


def train(model, X_train, y_train, X_valid, y_valid, num_epochs,
          learning_rate=0.1):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):
        minibatch_gen = minibatch_generator(
            X_train, y_train, minibatch_size)

        for X_train_mini, y_train_mini in minibatch_gen:
            a_h, a_out = model.forward(X_train_mini)

            # 计算梯度
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = model.backward(X_train_mini, a_h, a_out,
                                                                                            y_train_mini)

            # 更新权重和偏置项
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out

        # 计算每一次迭代的均方误差和准确率
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc * 100, valid_acc * 100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e + 1:03d}/{num_epochs:03d} '
              f'| Train MSE: {train_mse:.2f} '
              f'| Train Acc: {train_acc:.2f}% '
              f'| Valid Acc: {valid_acc:.2f}%')

    return epoch_loss, epoch_train_acc, epoch_valid_acc


np.random.seed(123)
epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    model, X_train, y_train, X_valid, y_valid,
    num_epochs=50, learning_rate=0.1)

# 迭代过程可视化
plt.plot(range(len(epoch_loss)), epoch_loss)
plt.ylabel(r'$Mean\ squared\ error$')
plt.xlabel('$Epoch$')
plt.show()

plt.plot(range(len(epoch_train_acc)), epoch_train_acc,
         label='$Training$')
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc,
         label='$Validation$')
plt.ylabel('$Accuracy$')
plt.xlabel('$Epochs$')
plt.legend(loc='lower right')
plt.show()

# 计算模型在测试数据集上的预测准确率以评估模型的泛化性能
test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# 提取并绘制模型在测试数据集上出现错误的前25个样本
X_test_subset = X_test[:1000, :]
y_test_subset = y_test[:1000]

_, probas = model.forward(X_test_subset)
test_pred = np.argmax(probas, axis=1)

misclassified_images = X_test_subset[y_test_subset != test_pred][:25]
misclassified_labels = test_pred[y_test_subset != test_pred][:25]
correct_labels = y_test_subset[y_test_subset != test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5,
                       sharex=True, sharey=True, figsize=(8, 8))
ax = ax.flatten()
cnt = 0
for i in ax:
    if cnt == 0:
        i.set_xticks([])
        i.set_yticks([])
    img = misclassified_images[cnt].reshape(28, 28)
    i.imshow(img, cmap='Greys', interpolation='nearest')
    i.set_title(f'{cnt + 1}) '
                f'True: {correct_labels[cnt]}\n'
                f' Predicted: {misclassified_labels[cnt]}')
plt.tight_layout()
plt.show()