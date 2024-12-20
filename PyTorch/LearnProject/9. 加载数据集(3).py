# 从torchvision.datasets库中获取数据集
import torch
import torchvision
from itertools import islice
from matplotlib import pyplot as plt

image_path = '../'
celeba_dataset = torchvision.datasets.CelebA(
    image_path, download=False, split='train', target_type='attr'
)

# 检查数据对象是否属于torch.utils.data.Dataset类
assert isinstance(celeba_dataset, torch.utils.data.Dataset)

# 获取第一个样本
# iter(celeba_dataset)：这个函数调用将 celeba_dataset 转换为一个迭代器
# next()：这个函数用于获取迭代器的下一个元素，如果没有更多的元素，它将抛出一个 StopIteration 异常
example = next(iter(celeba_dataset))
print(example)

fig = plt.figure(figsize=(12, 8))
# islice允许你从迭代器中获取一定数量的切片
for i, (image, label) in islice(enumerate(iter(celeba_dataset)), 18):
    ax = fig.add_subplot(3, 6, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image)
    ax.set_title(f'{label[31]}', size=15)
plt.show()


image_path = '../'
celeba_dataset = torchvision.datasets.MNIST(image_path, train=True, download=False)

assert isinstance(celeba_dataset, torch.utils.data.Dataset)

example = next(iter(celeba_dataset))
print(example)

fig = plt.figure(figsize=(15, 6))
for i, (image, label) in islice(enumerate(iter(celeba_dataset)), 10):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image)
    ax.set_title(f'{label}', size=15)
plt.show()