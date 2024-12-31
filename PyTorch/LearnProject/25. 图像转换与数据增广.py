import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms

# 加载CelebA数据集，目标标签为attr（属性标签）
image_path = '../'
celeba_train_dataset = torchvision.datasets.CelebA(
    image_path, split='train', download=False, target_type='attr'
)
celeba_valid_dataset = torchvision.datasets.CelebA(
    image_path, split='valid', download=False, target_type='attr'
)
celeba_test_dataset = torchvision.datasets.CelebA(
    image_path, split='test', download=False, target_type='attr'
)

# 不同类型的图像转换
fig = plt.figure(figsize=(16, 8))
# 裁剪到边界框
ax = fig.add_subplot(251)
img, attr1 = celeba_train_dataset[0]
ax.set_title('Crop to a bounding box', size=13)
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(img)
ax = fig.add_subplot(256)
img_cropped = transforms.functional.crop(img, 50, 20, 128, 128)
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(img_cropped)
# 水平翻转
ax = fig.add_subplot(252)
img, attr2 = celeba_valid_dataset[1]
ax.set_title('Flip (horizontal)', size=13)
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(img)
ax = fig.add_subplot(257)
img_flipped = transforms.functional.hflip(img)
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(img_flipped)
# 调整对比度
ax = fig.add_subplot(253)
img, attr3 = celeba_valid_dataset[2]
ax.set_title('Adjust contrast', size=13)
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(img)
ax = fig.add_subplot(258)
img_adj_contrast = transforms.functional.adjust_contrast(img, contrast_factor=2)
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(img_adj_contrast)
# 调节亮度
ax = fig.add_subplot(254)
img, attr4 = celeba_valid_dataset[3]
ax.set_title('Adjust brightness', size=13)
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(img)
ax = fig.add_subplot(259)
img_brightness = transforms.functional.adjust_brightness(img, brightness_factor=1.3)
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(img_brightness)
# 居中裁剪并调整大小
ax = fig.add_subplot(255)
img, attr5 = celeba_valid_dataset[4]
ax.set_title('Center crop and resize', size=13)
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(img)
ax = fig.add_subplot(2, 5, 10)
img_center_crop = transforms.functional.center_crop(img, [0.7 * 218, 0.7 * 178])
img_resized = transforms.functional.resize(img_center_crop, size=(218, 178))
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(img_resized)
plt.show()

# 图像的随机转换
torch.manual_seed(1)
fig = plt.figure(figsize=(16, 12))
for i, (img, attr) in enumerate([celeba_train_dataset[0], celeba_train_dataset[1], celeba_train_dataset[2]]):
    ax = fig.add_subplot(3, 4, i * 4 + 1, xticks=[], yticks=[])
    ax.imshow(img)
    if i == 0:
        ax.set_title('Original', size=13)
    ax = fig.add_subplot(3, 4, i * 4 + 2, xticks=[], yticks=[])
    img_transform = transforms.Compose([
        transforms.RandomCrop([178, 178])
    ])
    img_cropped = img_transform(img)
    ax.imshow(img_cropped)
    if i == 0:
        ax.set_title('Step 1:  Random Crop', size=13)
    ax = fig.add_subplot(3, 4, i * 4 + 3, xticks=[], yticks=[])
    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip()
    ])
    img_flipped = img_transform(img_cropped)
    ax.imshow(img_flipped)
    if i == 0:
        ax.set_title('Step 2:  Random Flip', size=13)
    ax = fig.add_subplot(3, 4, i * 4 + 4, xticks=[], yticks=[])
    img_resized = transforms.functional.resize(img_flipped, size=(128, 128))
    ax.imshow(img_resized)
    if i == 0:
        ax.set_title('Step 3:  Resize', size=13)
plt.show()