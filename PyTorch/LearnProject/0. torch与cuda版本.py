import torch


def print_torch_cuda_info():
    print(f"PyTorch版本: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA是否可用: 是")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA不可用，请确保已安装CUDA和cuDNN，并且PyTorch已正确配置为使用GPU。")


if __name__ == "__main__":
    print_torch_cuda_info()