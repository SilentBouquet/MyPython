import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset, DataLoader

# 加载处理好的数据
file_path = 'processed_transactions.xlsx'
data = pd.read_excel(file_path)

# 特征和标签
X = data.drop(columns=['是否欺诈'])  # 特征
y = data['是否欺诈']  # 标签

# 确保所有特征都是数值型
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype(str).astype('category').cat.codes

# 使用SMOTE处理类别不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 转换为 NumPy 数组
X = X_resampled.values.astype(np.float32)
y = y_resampled.values.astype(np.float32)


# 定义数据集类
class TransactionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建数据集
dataset = TransactionDataset(X, y)

# 创建数据加载器
batch_size = 64
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 定义生成器（Generator）
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        return self.model(z)


# 定义判别器（Discriminator）
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)


# 超参数
input_dim = X.shape[1]  # 输入特征维度
latent_dim = 100  # 噪声维度
lr = 0.0001  # 学习率
num_epochs = 100  # 训练轮数
n_critic = 5  # 每训练5次判别器后训练1次生成器
clip_value = 0.1  # 梯度裁剪值

# 初始化生成器和判别器
generator = Generator(latent_dim, input_dim)
discriminator = Discriminator(input_dim)

# 定义优化器
optimizer_G = optim.RMSprop(generator.parameters(), lr=lr)
optimizer_D = optim.RMSprop(discriminator.parameters(), lr=lr)

# 创建保存模型的目录
model_dir = "../My_Model"
os.makedirs(model_dir, exist_ok=True)


# 训练 WGAN
for epoch in range(num_epochs):
    for i, (real_samples, _) in enumerate(data_loader):
        # 训练判别器
        for _ in range(n_critic):
            optimizer_D.zero_grad()

            # 使用真实样本
            real_validity = discriminator(real_samples)
            real_loss = -torch.mean(real_validity)

            # 使用生成样本
            z = torch.randn(batch_size, latent_dim)
            gen_samples = generator(z)
            fake_validity = discriminator(gen_samples.detach())
            fake_loss = torch.mean(fake_validity)

            # 总损失
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # 梯度裁剪
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # 训练生成器
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        gen_samples = generator(z)
        gen_validity = discriminator(gen_samples)
        g_loss = -torch.mean(gen_validity)
        g_loss.backward()
        optimizer_G.step()

    # 打印损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# 保存最终模型
torch.save(generator.state_dict(), os.path.join(model_dir, 'generator_final.pth'))
torch.save(discriminator.state_dict(), os.path.join(model_dir, 'discriminator_final.pth'))
print("Final models saved.")


# 生成样本
def generate_samples(num_samples):
    z = torch.randn(num_samples, latent_dim)
    generated_samples_normalized = generator(z)
    generated_samples = generated_samples_normalized.clone()
    generated_samples_df = pd.DataFrame(generated_samples_normalized.detach().numpy(), columns=X_resampled.columns)
    return generated_samples_df.values


# 生成一些样本
generated_data = generate_samples(1000)
print("Generated samples shape:", generated_data.shape)