import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json
import time
import copy
import warnings
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# 忽略torchvision的deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 超参数设置
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
NUM_CLASSES = 5
IMG_SIZE = 224
EARLY_STOPPING_PATIENCE = 10  # 早停耐心值


class HerbResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(HerbResNet, self).__init__()
        # 使用新的weights参数替代deprecated的pretrained参数
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.backbone = models.resnet50(weights=weights)

        # 修改最后的全连接层
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0


def setup_data_loaders():
    """设置数据加载器"""
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder('train', transform=train_transform)
    val_dataset = datasets.ImageFolder('val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset.classes


def train_step(model, train_loader, criterion, optimizer, device):
    """单个训练步骤"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate_step(model, val_loader, criterion, device):
    """单个验证步骤"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy, all_preds, all_targets


def train_model():
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 加载数据
    print('加载数据集...')
    train_loader, val_loader, class_names = setup_data_loaders()

    # 加载中文类别映射
    with open('class_mapping.json', 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)

    print(f'训练集: {len(train_loader.dataset)} 样本')
    print(f'验证集: {len(val_loader.dataset)} 样本')
    print(f'类别: {[class_mapping.get(name, name) for name in class_names]}')

    # 计算总训练步数
    total_steps = NUM_EPOCHS * len(train_loader)
    print(f'总训练步数: {total_steps} (训练{NUM_EPOCHS}轮 × {len(train_loader)}批次)')

    # 初始化模型
    print('初始化模型...')
    model = HerbResNet(num_classes=NUM_CLASSES).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # 早停机制
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    # 训练记录
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    current_step = 0

    print('\n开始训练...')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        # 训练阶段
        train_loss, train_acc = train_step(model, train_loader, criterion, optimizer, device)
        current_step += len(train_loader)

        # 验证阶段
        val_loss, val_acc, val_preds, val_targets = validate_step(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': class_names
            }, 'best_model.pth')

        # 计算训练时间
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # 输出训练进度
        progress = (epoch + 1) / NUM_EPOCHS * 100
        print(f'[{epoch + 1:2d}/{NUM_EPOCHS}] 步数:{current_step:4d}/{total_steps} '
              f'进度:{progress:5.1f}% | '
              f'训练: {train_acc:5.1f}%(Loss:  {train_loss:.3f}) | '
              f'验证: {val_acc:5.1f}%(Loss:  {val_loss:.3f}) | '
              f'LR:{current_lr:.1e} | {epoch_time:.1f}s')

        # 最佳模型提示
        if val_acc > best_val_acc:
            print(f'新的最佳模型! 验证准确率: {val_acc:.2f}%')

        # 每10轮显示详细报告
        if (epoch + 1) % 10 == 0:
            print(f'\n第{epoch + 1}轮分类报告:')
            target_names = [class_mapping.get(name, name) for name in class_names]
            report = classification_report(val_targets, val_preds, target_names=target_names, digits=3)
            print(report)

        # 早停检查
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f'\n早停触发! 在第{epoch + 1}轮停止训练')
            print(f'验证准确率已连续{EARLY_STOPPING_PATIENCE}轮无改善')
            break

    # 保存最后的模型
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'class_names': class_names,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }, 'last_model.pth')

    print('=' * 80)
    print(f'训练完成!')
    print(f'最佳验证准确率: {best_val_acc:.2f}%')
    print(f'模型已保存: best_model.pth, last_model.pth')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    return train_losses, train_accs, val_losses, val_accs


def plot_training_curves(train_losses, train_accs, val_losses, val_accs):
    """绘制训练曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(train_losses, label='训练损失', color='blue')
    ax1.plot(val_losses, label='验证损失', color='red')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失')
    ax1.set_title('训练和验证损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(train_accs, label='训练准确率', color='blue')
    ax2.plot(val_accs, label='验证准确率', color='red')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('准确率 (%)')
    ax2.set_title('训练和验证准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print('训练曲线已保存: training_curves.png')


if __name__ == '__main__':
    # 开始训练
    train_losses, train_accs, val_losses, val_accs = train_model()

    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)