import torch
import torch.nn as nn
import numpy as np

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
y_values = [2 * i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

# 指定使用GPU进行训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

epochs = 1000
learning_rate = 0.01
# 随机梯度下降法的优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 损失函数
criterion = nn.MSELoss()

for epoch in range(epochs):
    epoch += 1
    # 格式转化成tensor
    inputs = torch.from_numpy(x_train).to(device)
    targets = torch.from_numpy(y_train).to(device)
    # 每一次迭代都要对梯度清零
    optimizer.zero_grad()
    # 前向传播
    outputs = model(inputs)
    # 计算损失
    loss = criterion(outputs, targets)
    # 反向传播
    loss.backward()
    # 更新权重参数
    optimizer.step()
    # 打印迭代过程
    if epoch % 50 == 0:
        print("epoch: {}/{}, loss: {}".format(epoch, epochs, loss.item()))

# 模型预测结果
predictions_gpu = model((torch.from_numpy(x_train)).to(device).requires_grad_())
predictions_cpu = predictions_gpu.cpu().data.numpy()
print("predictions: \n{}".format(predictions_cpu.reshape(-1)))

# 模型的保存与读取
# torch.save(model.state_dict(), 'model.pkl')
# model.load_state_dict(torch.load('model.pkl'))