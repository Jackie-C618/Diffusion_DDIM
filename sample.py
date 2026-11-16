"""
================================================================================
sample.py —— DDPM 采样

    本文件直接用 `torch.load(CKPT)`，只在当前工作目录找权重；
    在 DDIM 的采样中更改了这一点，find_ckpt() 自动去隔壁目录借。
================================================================================
"""
import torch                              
import matplotlib
matplotlib.use("Agg")                      
import matplotlib.pyplot as plt

from config import CKPT, SAMPLES_PNG, DEVICE, SEED  
from model import UNet                              
from diffusion import sample, plot_sampling_trajectory  
                                                       


def save_grid(x, n=4, save_path=SAMPLES_PNG):
    """
    画图：n×n 采样训练后的数字

    参数：
        x : (B,1,28,28) 的张量，值域约 [-1,1]
        n : 网格边长，会画 n×n 个子图，所以要保证 B >= n*n
    """
    # 反归一化：先 clamp 到数据区间（对称裁剪），再把 [-1,1] → [0,1]
    # .cpu() 是必须的：matplotlib 只吃 numpy/CPU 张量，GPU 张量会报错
    x = (x.clamp(-1, 1) * 0.5 + 0.5).cpu()          # [-1,1] -> [0,1]
    fig, axs = plt.subplots(n, n, figsize=(n, n))
    for i, ax in enumerate(axs.flat):               # .flat 把 2D 子图数组摊平成迭代器
        ax.imshow(x[i, 0], cmap="gray")             # x[i,0] → 第 i 张图的 (28,28) 灰度矩阵
        ax.axis("off")                              # 隐去坐标轴
    plt.tight_layout()                              # 自动调节边距
    plt.savefig(save_path)
    plt.close()
    print(f"[sample] 已保存生成图网格 -> {save_path}")


def generate(n=16):
    """加载权重 → 采样 → 出图。这是 DDPM 侧的完整流程。"""
    torch.manual_seed(SEED)                         # 固定种子；但 DDPM 每步抽新噪声，所以这里只固定起点，结果仍会随机
    model = UNet().to(DEVICE)                       # 按 config 建一个同结构空网络
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
                                                    # 把训练好的参数灌进去；
                                                    # map_location 让 GPU 训的权重也能在 CPU 上加载
    model.eval()                                    # 关掉 dropout/BN 的训练行为（本模型主要是习惯性写法）
    # n=16 → 一次生成 16 张；steps_show=6 表示轨迹图大约存 6 帧
    x, traj = sample(model, shape=(n, 1, 28, 28), steps_show=6)
    save_grid(x, n=int(n ** 0.5))                   # 16 → 4×4 网格（n 必须让 n*n <= 图片数）
    plot_sampling_trajectory(traj)                  # 去噪演化图
    return x


if __name__ == "__main__":
    generate()
