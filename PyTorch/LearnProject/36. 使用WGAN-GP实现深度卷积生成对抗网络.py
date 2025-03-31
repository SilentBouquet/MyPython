import os
import matplotlib
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import grad as torch_grad


# 保存模型
def save_models(generator, discriminator):
    # 创建保存目录（如果不存在）
    save_path = '../My_Model/WGAN-GP'
    os.makedirs(save_path, exist_ok=True)

    # 保存生成器和判别器的模型参数
    torch.save(generator.state_dict(), f'{save_path}/generator_final.pth')
    torch.save(discriminator.state_dict(), f'{save_path}/discriminator_final.pth')
    print("Models saved")


# 生成随机向量
def creat_noise(batch_size, z_size, mode_z):
    if mode_z == 'normal':
        return torch.randn(batch_size, z_size, 1, 1)
    elif mode_z == 'uniform':
        return torch.rand(batch_size, z_size, 1, 1) * 2 - 1


# 生成器网络
def make_generator_network(input_size, n_filters):
    model = nn.Sequential(
        # 转置卷积层，用于上采样操作，将输入特征图的通道数从 input_size 转换为 n_filters * 4
        nn.ConvTranspose2d(input_size, n_filters * 4, 4, 1, 0, bias=False),
        # 层归一化层：在特征维度上进行归一化
        nn.InstanceNorm2d(n_filters * 4),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 3, 2, 1, bias=False),
        nn.InstanceNorm2d(n_filters * 2),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(n_filters),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters, 1, 4, 2, 1, bias=False),
        # 将输出值限制在 [-1, 1] 范围内，适用于生成图像的像素值
        nn.Tanh()
    )
    return model


# 判别器网络
class Discriminator(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.network = nn.Sequential(
            # 卷积层，将输入图像的通道数从 1 转换为 n_filters，进行下采样
            nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters * 2, n_filters * 4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters * 4, 1, 4, 1, 0, bias=False),
            # 将输出值限制在 [0, 1] 范围内，表示输入图像为真实图像的概率
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.network(x)
        return output.squeeze(0).view(-1, 1)


# 加载数据集
image_path = '../'
transform = transforms.Compose([
    # 将图像数据转换为张量，并将像素值从 [0, 255] 归一化到 [0, 1]
    transforms.ToTensor(),
    # 将数据分布对称地映射到 [-1, 1] 范围内
    transforms.Normalize(0.5, 0.5)]
)
mnist_dataset = torchvision.datasets.MNIST(
    root=image_path, train=True, transform=transform, download=False
)

# 构建深度卷积生成对抗网络模型
batch_size = 64
torch.manual_seed(1)
np.random.seed(1)
z_size = 100
image_size = (28, 28)
n_filters = 32
lambda_gp = 10.0
mode_z = 'uniform'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen_model = make_generator_network(z_size, n_filters).to(device)
disc_model = Discriminator(n_filters).to(device)
mnist_dl = torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print(gen_model)
print(disc_model)

# 定义优化器
gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.0002)
disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=0.0002)


# 定义一个计算梯度惩罚的函数
def gradient_penalty(real_data, generated_data):
    batch_size = real_data.size(0)
    # requires_grad=True表示这个张量在后续计算中需要计算梯度
    alpha = torch.rand(real_data.shape[0], 1, 1, 1, requires_grad=True, device=device)
    # 通过线性插值构造新的样本interpolated，这些样本位于真实数据和生成数据之间的某个位置。
    interpolated = alpha * real_data + ((1 - alpha) * generated_data)
    proba_interpolated = disc_model(interpolated)
    gradients = torch_grad(
        outputs=proba_interpolated,
        inputs=interpolated,
        # 指定梯度计算的权重
        grad_outputs=torch.ones(proba_interpolated.size()).to(device),
        # 在计算梯度时保留计算图，以便后续可以继续计算高阶梯度
        # 确保计算图在反向传播后不会被释放
        create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    # 计算每个样本梯度的L2范数（即梯度的模长）
    gradients_norm = gradients.norm(2, dim=1)
    return lambda_gp * ((gradients_norm - 1) ** 2).mean()


# 训练判别器
def d_train(x):
    disc_model.zero_grad()
    batch_size = x.shape[0]
    x = x.to(device)
    d_real = disc_model(x)
    input_z = creat_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z)
    d_generated = disc_model(g_output)
    d_loss = d_generated.mean() - d_real.mean() + gradient_penalty(x.data, g_output.data)
    d_loss.backward()
    disc_optimizer.step()
    return d_loss.data.item()


# 训练生成器
def g_train(x):
    gen_model.zero_grad()
    batch_size = x.shape[0]
    input_z = creat_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z)
    d_generated = disc_model(g_output)
    g_loss = -d_generated.mean()
    g_loss.backward()
    gen_optimizer.step()
    return g_loss.data.item()


def create_samples(g_model, input_z):
    g_output = g_model(input_z)
    images = torch.reshape(g_output, (batch_size, *image_size))
    return (images + 1) / 2.0


# 训练模型
fixed_z = creat_noise(batch_size, z_size, mode_z).to(device)
epoch_samples = []
num_epochs = 100
torch.manual_seed(1)
critic_iterations = 5
for epoch in range(1, num_epochs + 1):
    gen_model.train()
    d_losses, g_losses = [], []
    for i, (x, _) in enumerate(mnist_dl):
        d_loss = 0
        # 单独训练判别器多次而生成器只训练一次
        for _ in range(critic_iterations):
            d_loss = d_train(x)
        d_losses.append(d_loss)
        g_losses.append(g_train(x))
    print(f'Epoch {epoch:03d} | D Loss >>'
          f'  {torch.FloatTensor(d_losses).mean():.4f}')
    gen_model.eval()
    epoch_samples.append(create_samples(gen_model, fixed_z).detach().cpu().numpy())

# 在训练结束后保存模型
save_models(gen_model, disc_model)

# 展示模型生成的样本
matplotlib.use('Qt5Agg')
selected_epochs = [1, 2, 4, 10, 50, 100]
fig = plt.figure(figsize=(10, 14))
for i, epoch in enumerate(selected_epochs):
    for j in range(5):
        ax = fig.add_subplot(6, 5, i * 5 + j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(
                -0.06, 0.5, f'Epoch {epoch}',
                rotation=90, size=14, color='black',
                horizontalalignment='right',
                verticalalignment='center',
                transform=ax.transAxes
            )
        image = epoch_samples[epoch - 1][j]
        ax.imshow(image, cmap='gray_r')
plt.show()