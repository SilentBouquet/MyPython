import matplotlib
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def make_generator_network(
        input_size=20,
        num_hidden_layers=2,
        num_hidden_units=256,
        num_output_units=784):
    model = nn.Sequential()
    for i in range(num_hidden_layers):
        model.add_module('fc_g{}'.format(i), nn.Linear(input_size, num_hidden_units))
        model.add_module('relu_g{}'.format(i), nn.LeakyReLU())
        input_size = num_hidden_units
    model.add_module(f'fc_g{num_hidden_layers}', nn.Linear(input_size, num_output_units))
    model.add_module('tanh_g', nn.Tanh())
    return model


def make_discriminator_network(
        input_size,
        num_hidden_layers=2,
        num_hidden_units=256,
        num_output_units=1):
    model = nn.Sequential()
    for i in range(num_hidden_layers):
        # bias=False 表示在全连接层中不使用偏置项
        model.add_module('fc_d{}'.format(i), nn.Linear(input_size, num_hidden_units, bias=False))
        model.add_module('relu_d{}'.format(i), nn.LeakyReLU())
        model.add_module('dropout', nn.Dropout(0.5))
        input_size = num_hidden_units
    model.add_module(f'fc_d{num_hidden_layers}', nn.Linear(input_size, num_output_units))
    model.add_module('sigmoid', nn.Sigmoid())
    return model


image_size = (28, 28)
z_size = 40
gen_hidden_layers = 2
gen_hidden_size = 256
disc_hidden_layers = 2
disc_hidden_size = 256
torch.manual_seed(0)
gen_model = make_generator_network(
    input_size=z_size,
    num_hidden_layers=gen_hidden_layers,
    num_hidden_units=gen_hidden_size,
    # np.prod 用于计算数组中元素的乘积
    num_output_units=np.prod(image_size)
)
print(gen_model)
disc_model = make_discriminator_network(
    input_size=np.prod(image_size),
    num_hidden_layers=disc_hidden_layers,
    num_hidden_units=disc_hidden_size,
)
print(disc_model)

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
example, label = next(iter(mnist_dataset))
print(f'Min: {example.min()}, Max: {example.max()}')
print(example.shape)


# 生成随机向量
def creat_noise(batch_size, z_size, mode_z):
    if mode_z == 'normal':
        return torch.randn(batch_size, z_size)
    elif mode_z == 'uniform':
        return torch.rand(batch_size, z_size) * 2 - 1


# 打印一批样本
batch_size = 32
dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False)
input_real, real_label = next(iter(dataloader))
# view 用于重塑张量形状
input_real = input_real.view(batch_size, -1)
torch.manual_seed(0)
mode_z = 'uniform'
input_z = creat_noise(batch_size, z_size, mode_z)
print('input-z -- shape: ', input_z.shape)
print('input-real -- shape: ', input_real.shape)
g_output = gen_model(input_z)
print('g_output -- shape: ', g_output.shape)
d_proba_real = disc_model(input_real)
d_proba_fake = disc_model(g_output)
print('Disc. (real) -- shape: ', d_proba_real.shape)
print('Disc. (fake) -- shape: ', d_proba_fake.shape)

# 定义训练参数
loss = nn.BCELoss()
# 计算生成器的损失，目标是让判别器将生成的假样本误认为真实样本
g_labels_real = torch.ones_like(d_proba_fake)
g_loss = loss(d_proba_fake, g_labels_real)
print(f'Generator Loss: {g_loss.item():.4f}')
# 计算判别器的损失，分别对真实样本和假样本进行判别，并计算对应的损失
d_labels_real = torch.ones_like(d_proba_real)
d_labels_fake = torch.zeros_like(d_proba_fake)
d_loss_real = loss(d_proba_real, d_labels_real)
d_loss_fake = loss(d_proba_fake, d_labels_fake)
print(f'Discriminator Losses: Real {d_loss_real.item():.4f} Fake {d_loss_fake.item():.4f}')

# 定义数据加载器和网络模型
batch_size = 64
torch.manual_seed(1)
np.random.seed(1)
mnist_dl = torch.utils.data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gen_model = gen_model.to(device)
disc_model = disc_model.to(device)
g_optimizer = torch.optim.Adam(gen_model.parameters())
d_optimizer = torch.optim.Adam(disc_model.parameters())


# 训练判别器
def d_train(x):
    disc_model.zero_grad()
    batch_size = x.shape[0]
    x = x.view(batch_size, -1).to(device)
    d_labels_real = torch.ones(batch_size, 1, device=device)
    d_proba_real = disc_model(x)
    d_loss_real = loss(d_proba_real, d_labels_real)
    input_z = creat_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z)
    d_proba_fake = disc_model(g_output)
    d_labels_fake = torch.zeros(batch_size, 1, device=device)
    d_loss_fake = loss(d_proba_fake, d_labels_fake)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data.item(), d_proba_real.detach(), d_proba_fake.detach()


# 训练生成器
def g_train(x):
    gen_model.zero_grad()
    batch_size = x.shape[0]
    input_z = creat_noise(batch_size, z_size, mode_z).to(device)
    g_labels_real = torch.ones(batch_size, 1, device=device)
    g_output = gen_model(input_z)
    d_proba_fake = disc_model(g_output)
    g_loss = loss(d_proba_fake, g_labels_real)
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()


def create_samples(g_model, input_z):
    g_output = g_model(input_z)
    images = torch.reshape(g_output, (batch_size, *image_size))
    return (images + 1) / 2.0


fixed_z = creat_noise(batch_size, z_size, mode_z).to(device)
epoch_samples = []
all_d_losses = []
all_g_losses = []
all_d_real = []
all_d_fake = []
num_epochs = 100

for epoch in range(1, num_epochs + 1):
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
    epoch_samples.append(
        create_samples(gen_model, fixed_z).detach().cpu().numpy()
    )

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
