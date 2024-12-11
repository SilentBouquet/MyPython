# 用存储在本地硬盘的文件创建数据集
import os
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class ImagesDataset:
    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        label = self.labels[idx]
        return file, label


imgdir_path = pathlib.Path('../cat_dog_images')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
print(file_list)

fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(file_list):
    img = Image.open(file)
    print('Image shape: ', np.array(img).shape)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    # os.path.basename()用于从完整的文件路径中提取文件名
    ax.set_title(os.path.basename(file), size=15)
plt.tight_layout()
plt.show()

# 为数据集添加标签，并且将大小转换一致
labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_list]
img_height, img_width = 80, 120
# transforms.Compose()用于将多个图像转换操作组合成一个操作序列。它会按照列表中的顺序依次执行每个操作
# transforms.ToTensor()用于将一个 PIL 图像或者一个ndarray的图像转换为 torch.Tensor
# transforms.Resize()用于将图像调整为指定大小
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((img_height, img_width))])
image_dataset = ImagesDataset(file_list, labels)

fig = plt.figure(figsize=(10, 6))
for i, (file, label) in enumerate([i for i in image_dataset]):
    img = Image.open(file)
    img = transform(img)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    # img.numpy()用于将 PyTorch 张量（img）转换为一个 NumPy 数组
    # transpose((1, 2, 0))方法用于将数组从 CxHxW 格式转换为 HxWxC 格式
    ax.imshow(img.numpy().transpose((1, 2, 0)))
    ax.set_title(f'{label}"', size=15)
plt.tight_layout()
plt.show()