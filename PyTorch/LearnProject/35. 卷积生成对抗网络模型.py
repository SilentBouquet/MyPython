import os
import matplotlib
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


# 保存模型
def save_models(generator, discriminator):
    # 创建保存目录（如果不存在）
    save_path = '../My_Model/卷积GAN'
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
        # 批量归一化层，用于加速训练并稳定网络
        nn.BatchNorm2d(n_filters * 4),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 3, 2, 1, bias=False),
        nn.BatchNorm2d(n_filters * 2),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
        nn.BatchNorm2d(n_filters),
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
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters * 2, n_filters * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
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
mode_z = 'uniform'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen_model = make_generator_network(z_size, n_filters).to(device)
disc_model = Discriminator(n_filters).to(device)
mnist_dl = torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print(gen_model)
print(disc_model)

# 定义损失函数和优化器
loss_fn = nn.BCELoss()
gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=0.0003)
disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=0.0002)


# 训练判别器
def d_train(x):
    disc_model.zero_grad()
    batch_size = x.shape[0]
    x = x.to(device)
    d_labels_real = torch.ones(batch_size, 1, device=device)
    d_proba_real = disc_model(x)
    d_loss_real = loss_fn(d_proba_real, d_labels_real)
    input_z = creat_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z)
    d_proba_fake = disc_model(g_output)
    d_labels_fake = torch.zeros(batch_size, 1, device=device)
    d_loss_fake = loss_fn(d_proba_fake, d_labels_fake)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    disc_optimizer.step()
    return d_loss.data.item(), d_proba_real.detach(), d_proba_fake.detach()


# 训练生成器
def g_train(x):
    gen_model.zero_grad()
    batch_size = x.shape[0]
    input_z = creat_noise(batch_size, z_size, mode_z).to(device)
    g_labels_real = torch.ones(batch_size, 1, device=device)
    g_output = gen_model(input_z)
    d_proba_fake = disc_model(g_output)
    g_loss = loss_fn(d_proba_fake, g_labels_real)
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
all_d_losses = []
all_g_losses = []
all_d_real = []
all_d_fake = []
num_epochs = 100

for epoch in range(1, num_epochs + 1):
    gen_model.train()
    d_losses, g_losses = [], []
    d_vals_real, d_vals_fake = [], []
    for i, (x, _) in enumerate(mnist_dl):
        d_loss, d_proba_real, d_proba_fake = d_train(x)
        d_losses.append(d_loss)
        g_losses.append(g_train(x))
        d_vals_real.append(d_proba_real.mean().cpu())
        d_vals_fake.append(d_proba_fake.mean().cpu())

    all_d_losses.append(torch.tensor(d_losses).mean())
    all_g_losses.append(torch.tensor(g_losses).mean())
    all_d_real.append(torch.tensor(d_vals_real).mean())
    all_d_fake.append(torch.tensor(d_vals_fake).mean())
    print(f'Epoch {epoch:03d} | Avg Losses >>'
          f'  G/D {all_g_losses[-1]:.4f}/{all_d_losses[-1]:.4f}'
          f'  [D-Real: {all_d_real[-1]:.4f}'
          f'  D-Fake: {all_d_fake[-1]:.4f}]')
    gen_model.eval()
    epoch_samples.append(
        create_samples(gen_model, fixed_z).detach().cpu().numpy()
    )

# 在训练结束后保存模型
save_models(gen_model, disc_model)

# 学习曲线可视化
matplotlib.use('Qt5Agg')
fig = plt.figure(figsize=(16, 6))
ax = fig.add_subplot(1, 2, 1)
plt.plot(all_g_losses, label='Generator Loss')
half_d_losses = [all_d_loss / 2 for all_d_loss in all_d_losses]
plt.plot(half_d_losses, label='Discriminator Loss')
plt.legend(fontsize=16)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Loss', size=15)
ax = fig.add_subplot(1, 2, 2)
plt.plot(all_d_real, label='Real: $D(\mathbf{x})$)')
plt.plot(all_d_fake, label='Fake: $D(\mathbf{x})$)')
plt.legend(fontsize=20)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Discriminator output', size=15)
plt.tight_layout()
plt.show()

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