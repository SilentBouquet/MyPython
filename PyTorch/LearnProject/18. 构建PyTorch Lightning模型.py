import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class MultiLayerPerceptron(pl.LightningModule):
    def __init__(self, image_shape=(1, 28, 28), hidden_units=(32, 16)):
        super().__init__()
        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.val_acc = Accuracy(task='multiclass', num_classes=10)
        self.test_acc = Accuracy(task='multiclass', num_classes=10)
        input_size = image_shape[0] * image_shape[1] * image_shape[2]
        all_layers = [nn.Flatten()]
        for hidden_unit in hidden_units:
            all_layers.append(nn.Linear(input_size, hidden_unit))
            all_layers.append(nn.ReLU())
            input_size = hidden_unit
        all_layers.append(nn.Linear(hidden_units[-1], 10))
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        # log 方法用于记录训练过程中的各种指标，比如损失值、准确率等
        # prog_bar=True 参数表示将这个指标显示在进度条中
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        # val_acc.compute()计算在验证步骤中累积的准确率，并且重置内部跟踪的计数器
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class MnistDataModel(pl.LightningDataModule):
    def __init__(self, data_path='../'):
        super().__init__()
        self.data_path = data_path
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        MNIST(self.data_path, train=True, download=True)

    def setup(self, stage=None):
        mnist_all = MNIST(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=False
        )

        self.train, self.val = random_split(mnist_all, [55000, 5000], generator=torch.Generator().manual_seed(0))
        self.test = MNIST(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False
        )

    def train_dataloader(self):
        # num_workers指定了用于数据加载的子进程数量，可以加速数据的加载过程
        # persistent_workers=True：这个参数指定了工作进程在数据加载完毕后是否应该保持活动状态。
        # 如果设置为 True，则工作进程在数据加载完毕后不会立即终止，而是保持活动状态，这样可以减少频繁创建和销毁进程的开销
        return DataLoader(self.train, batch_size=64, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64, num_workers=4, persistent_workers=True)


if __name__ == '__main__':
    torch.manual_seed(0)
    model = MnistDataModel()
    mnistclassifier = MultiLayerPerceptron()
    # 设置PyTorch中float32矩阵乘法的精度。设置为'high'可以提高矩阵乘法的精度
    torch.set_float32_matmul_precision('high')
    # 定义回调函数列表，其中包含了一个ModelCheckpoint实
    # 回调函数会在训练过程中保存性能最好的模型（基于验证集准确率val_acc）。
    # save_top_k=1表示只保存性能最好的一个模型，mode='max'表示选择性能指标最大值的模型
    callbacks = [ModelCheckpoint(save_top_k=1, mode='max', monitor="val_acc")]
    # 创建TensorBoardLogger实例
    logger = TensorBoardLogger(save_dir="../Runs", name="mnist_experiment")
    if torch.cuda.is_available():
        trainer = pl.Trainer(max_epochs=10, callbacks=callbacks, accelerator="gpu", devices=1, logger=logger)
    else:
        trainer = pl.Trainer(max_epochs=10, callbacks=callbacks, logger=logger)
    # 训练过程。model=mnistclassifier指定了要训练的模型，datamodule=model指定了提供数据的数据模块
    trainer.fit(model=mnistclassifier, datamodule=model)