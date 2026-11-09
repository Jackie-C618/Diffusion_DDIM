"""
sample_ddim.py — 加载训练好的模型，用 DDIM 采样器生成图片

★ 和 Diffusion_DDPM/sample.py 逐行对照，只有 3 处不同：
    1) import 从 `from diffusion import sample` 换成 `from ddim import ddim_sample`
    2) 多了 --steps / --eta 两个命令行参数（控制跳步数与随机性）
    3) 输出文件名加 `_ddim` 后缀，方便和 DDPM 的结果并排对比
  模型、权重、schedule、画图函数**全部沿用 DDPM，没有重复实现**
  （model.py 仅补回一处既有修复，见 README）。

💡 怎么跑：
    python sample_ddim.py                    # 默认 50 步、eta=0（确定性）
    python sample_ddim.py --steps 10        # 只要 10 步，看速度极限
    python sample_ddim.py --steps 100       # 100 步，通常更清晰
    python sample_ddim.py --eta 1           # eta=1，行为接近 DDPM
    python sample_ddim.py --interp          # 隐空间插值演示（DDIM 独有）

    需要 ckpt_last.pt。当前目录没有时，会自动去隔壁已训好的 DDPM 目录借
    （因为 DDIM 和 DDPM 用的是同一个 ε_θ，权重完全通用）。
"""
import argparse
import os
import torch
import matplotlib
matplotlib.use("Agg")

from config import CKPT, DEVICE, SEED
from model import UNet
from schedule import make_schedule
from sample import save_grid                    # ★ 复用 DDPM 的网格图，零改动
from diffusion import plot_sampling_trajectory  # ★ 复用 DDPM 的轨迹图，零改动
from ddim import ddim_sample, ddim_interpolate  # ★ 唯一实质差异：换成 DDIM 采样器

DDIM_SAMPLES_PNG = "samples_ddim.png"
DDIM_TRAJ_PNG = "sampling_traj_ddim.png"
DDIM_INTERP_PNG = "interpolation_ddim.png"


def find_ckpt():
    """找权重：优先当前目录，其次隔壁两个 DDPM 目录里已训好的那份。"""
    candidates = [
        CKPT,                                        # 本目录自己 train.py 训出来的
        os.path.join("..", "ddpm_mnist", CKPT),       # 实验版（你已训好 ckpt）
        os.path.join("..", "Diffusion_DDPM", CKPT),   # 整理版
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "没找到 %s。请先运行 python train.py 训练，"
        "或把 DDPM 训好的 ckpt_last.pt 复制到本目录。" % CKPT
    )


def load_model():
    """加载 DDPM 训练好的 ε_θ（DDIM 直接复用，一行都不用改）。"""
    ckpt = find_ckpt()
    print("[sample_ddim] 使用权重:", ckpt)
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def generate(n=16, ddim_steps=50, eta=0.0):
    """生成 n 张图，保存 4×4 网格 + 去噪轨迹。"""
    torch.manual_seed(SEED)
    model = load_model()
    sched = make_schedule()

    x, traj = ddim_sample(
        model, sched, shape=(n, 1, 28, 28),
        ddim_steps=ddim_steps, eta=eta, device=DEVICE,
    )
    # 画图函数直接复用 DDPM 版：正好说明「变的只有采样，出图逻辑完全一样」
    save_grid(x, n=int(n ** 0.5), save_path=DDIM_SAMPLES_PNG)
    plot_sampling_trajectory(traj, save_path=DDIM_TRAJ_PNG)
    print("[sample_ddim] 完成：%d 步采样（DDPM 需 1000 步）" % ddim_steps)
    return x


def interpolate_demo(n=4, ddim_steps=50):
    """【可选】两个随机起点之间插值，展示 DDIM 确定性带来的语义连续变化。

    ❖ 这图反映什么：
       从左到右数字平滑" morph "过渡，而不是一堆无关噪声。
       因为 η=0 时 x_T 与生成图是一一对应的，隐空间才有连续语义。
       DDPM 每步都抽新噪声，做不出这种可控插值。
    """
    torch.manual_seed(SEED)
    model = load_model()
    sched = make_schedule()
    imgs = ddim_interpolate(model, sched, n=n * n,
                            ddim_steps=ddim_steps, eta=0.0,
                            seed=SEED, device=DEVICE)
    save_grid(imgs, n=n, save_path=DDIM_INTERP_PNG)
    return imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDIM 采样（复用 DDPM 权重）")
    parser.add_argument("--steps", type=int, default=50,
                        help="采样步数，默认 50；DDPM 固定 1000 步")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="随机性：0=确定性 DDIM，1≈DDPM")
    parser.add_argument("--n", type=int, default=16, help="生成几张图（默认 16）")
    parser.add_argument("--interp", action="store_true",
                        help="改为生成隐空间插值图")
    args = parser.parse_args()

    if args.interp:
        interpolate_demo(n=4, ddim_steps=args.steps)
    else:
        generate(n=args.n, ddim_steps=args.steps, eta=args.eta)
