import numpy as np
import matplotlib.pyplot as plt


# 定义激活函数
def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


# 定义输入 x
x = np.linspace(-5, 5, 100)

# 计算激活函数的输出
relu_output = relu(x)
sigmoid_output = sigmoid(x)
tanh_output = tanh(x)

# 绘制图像
plt.figure(figsize=(14, 6))
plt.rcParams['font.sans-serif'] = ['Lucida Sans Unicode']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'

plt.subplot(1, 3, 1)
plt.plot(x, relu_output, label="ReLU", color="blue")
plt.title(r"$ReLU\ Activation\ Function$", fontsize=15)
plt.xlabel("$x$", fontsize=14)
plt.ylabel("$F(x)$", fontsize=14)
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, sigmoid_output, label="Sigmoid", color="red")
plt.title(r"$Sigmoid\ Activation\ Function$", fontsize=15)
plt.xlabel("$x$", fontsize=14)
plt.ylabel("$F(x)$", fontsize=14)
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, tanh_output, label="Tanh", color="green")
plt.title(r"$Tanh\ Activation\ Function$", fontsize=15)
plt.xlabel("$x$", fontsize=14)
plt.ylabel("$F(x)$", fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.show()