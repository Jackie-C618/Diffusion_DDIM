"""
================================================================================
dataset.py —— 加载 MNIST，并把像素线性变换到 [-1,1]
================================================================================

【位置】
        config.py（BATCH_SIZE / DATA_DIR / NORM_* / IMAGE_SIZE / CHANNELS）
              ↓
        【dataset.py】 → DataLoader，每次吐出 (x, y)
              ↓
        train.py 取 x 当作 x_0（标签 y 在 DDPM/DDIM 里根本不用，
                 因为这是「无条件生成」，不按类别条件去噪）
              ↓
        sample_ddim.py 的反演演示也会用它取一张真实图做编码实验

【功能】
    1) 下载 / 读取 MNIST（第一次运行会自动下载到 DATA_DIR）
    2) 把 PIL 图片变成 PyTorch 张量，并线性变换到 [-1,1]
    3) 包成 DataLoader，按 BATCH_SIZE 一批一批地喂给训练循环

【关于线性变换】
    MNIST的数据集为颜色取值为[0,1]的灰度图，为方便正态分布处理，线性变换到[-1,1]
    ToTensor()              :  PIL 的 0~255 整数  ÷255  → [0,1] 浮点
    Normalize(0.5, 0.5)    :  (x - 0.5)/0.5 = 2x-1     → [-1,1]
    反变换（出图时）：(x + 1)/2 = 0.5x + 0.5           → [0,1]

================================================================================
"""
from torchvision import datasets, transforms      # datasets.MNIST 负责下载/读取；transforms 负责图片预处理
from torch.utils.data import DataLoader          # 把 Dataset 包成「按批吐数据」的迭代器

from config import BATCH_SIZE, DATA_DIR, NORM_MEAN, NORM_STD, IMAGE_SIZE, CHANNELS


def get_transform():

    return transforms.Compose([
        # ToTensor有两个功能
        # 1、将其转换为Tensor形式，由(Height,Wide,ChannelNumber) 排成 PyTorch 需要的 (C,H,W)
        # 2、归一化到 [0,1]，变为 float 格式
        # 单张图出来是 (1,28,28)。
        transforms.ToTensor(),                          # 归一化[0,1]

        # (x - mean)/std，把 [0,1] 线性搬到 [-1,1]。
        # 注意参数必须是 tuple（哪怕只有 1 个通道，也要写成 (0.5,) 而不是 0.5），
        # 因为 Normalize 会按「通道」逐个配对；写成标量会报错。
        # 由于 Normalize 两个变量需要为 tuple 类型（多通道处理），因此写为 (0.5,)
        transforms.Normalize((NORM_MEAN,), (NORM_STD,)), 
    ])


def get_mnist_dataloader(train=True):
    """构造 MNIST 的 DataLoader。

    参数 train=True → 60000 张训练集；False → 10000 张测试集。
    DDIM 的反演演示也用这份数据取真实图片，因为要验证「编码→重建」。
    """
    ds = datasets.MNIST(
        root=DATA_DIR, 
        train=train,
        download=True,
        transform=get_transform(), 
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,  # 每批次多少张；出来的 x 形状是 (B,1,28,28)
        shuffle=train,          # 训练集打乱（避免按标签顺序学），测试集不打乱
        num_workers=0,          # =0 时用主进程加载，num_workers: DataLoader 加载数据的子进程数，一般大数据集设置大于零
                                # 大数据集可设 >0 开多进程加速，但 Windows 多进程容易因 spawn 出问题，这里保持 0。
    )  


# # 自检：取一个 batch，确认形状与值域
# def test_dataset():
#     loader = get_mnist_dataloader()          # 首次会下载 MNIST
#     x, y = next(iter(loader))                # iter → 迭代器；next → 取第一批
#     print("[dataset] batch 形状:", tuple(x.shape), "标签形状:", tuple(y.shape))
#     print("[dataset] 像素范围: min=%.3f max=%.3f" % (x.min(), x.max()))
#     # 由于 Normalize 只做线性变换不做裁剪，因此 assert 的检查是必要的，检查输入是否越界
#     assert x.shape[1:] == (CHANNELS, IMAGE_SIZE, IMAGE_SIZE)   # 单张必须是 (1,28,28)
#     assert x.min() >= -1.01 and x.max() <= 1.01, "像素应在 [-1,1]"  # 断言失败时的报错信息
#     # 这里放 0.01 的余量是给浮点误差留空间，不是允许越界。
#     print("[dataset] 自检通过 ✅")


# # 只有直接运行本文件时才执行以下语句；被 import 时不会下载数据、不会打印。
# if __name__ == "__main__":
#     test_dataset()
