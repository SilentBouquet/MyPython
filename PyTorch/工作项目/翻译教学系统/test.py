import sys
import torch


def print_environment_info():
    # 获取 Python 版本
    python_version = sys.version.replace('\n', ' ')

    # 获取 PyTorch 版本
    pytorch_version = torch.__version__

    # 获取 CUDA 版本
    cuda_version = torch.version.cuda

    # 获取 cuDNN 版本
    cudnn_version = torch.backends.cudnn.version()

    # 获取 Torch 版本（即 PyTorch 版本）
    torch_version = torch.__version__

    # 打印环境信息
    print(f"Python 版本: {python_version}")
    print(f"PyTorch 版本: {pytorch_version}")
    print(f"CUDA 版本: {cuda_version}")
    print(f"cuDNN 版本: {cudnn_version}")
    print(f"Torch 版本: {torch_version}")


if __name__ == "__main__":
    print_environment_info()