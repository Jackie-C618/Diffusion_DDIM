"""
================================================================================
sample_ddim.py —— DDIM 调用窗口及相关测试
================================================================================

【演示示例】
    generate()          纯噪声 → 生成图，看 DDIM 跳步采样的成果
    interpolate_demo()  两个 x_T 之间插值，看确定性带来的语义隐空间
    inversion_demo()    真实图 → x_T → 重建，看反演是否可逆

【💡 参数示例】
    python sample_ddim.py                      # 默认 50 步、eta=0
    python sample_ddim.py --steps 10       
    python sample_ddim.py --steps 100       
    python sample_ddim.py --eta 0.5            # 任意实数 η
    python sample_ddim.py --eta 1              # 行为接近 DDPM
    python sample_ddim.py --interp             # 隐空间插值演示
    python sample_ddim.py --inversion         # 反演 + 重建对比

    需要 ckpt_last.pt。当前目录没有时，会自动去文件夹找已训好的 DDPM 目录借
================================================================================
"""
import argparse                                  # 解析 --steps / --eta 等命令行参数
import os                                        # 拼路径、判断权重文件是否存在
import torch                                
import matplotlib
matplotlib.use("Agg")                       
import matplotlib.pyplot as plt               

from config import CKPT, DEVICE, SEED
from model import UNet                          
from schedule import make_schedule              
from sample import save_grid                   
from diffusion import plot_sampling_trajectory   
from ddim import ddim_sample, ddim_invert, ddim_interpolate
                                               
DDIM_SAMPLES_PNG = "samples_ddim.png"
DDIM_TRAJ_PNG = "sampling_traj_ddim.png"
DDIM_INTERP_PNG = "interpolation_ddim.png"
DDIM_INVERSION_PNG = "inversion_ddim.png" 

def find_ckpt():
    """
    找权重：优先当前目录，其次 DDPM 目录里已训好的那份。
    """
    candidates = [
        CKPT,                                         # 本目录训练的
        os.path.join("..", "Diffusion_DDPM", CKPT),   # DDPM 训练的
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
    model = UNet().to(DEVICE)                        # 先按 config 建一个同结构的空网络
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                                                    # 导入参数；map_location 兼容跨设备加载
    model.eval()                                    # 切到推理模式
    return model


def generate(n=16, ddim_steps=50, eta=0.0):
    """
    从纯噪声生成 n 张图，保存 4×4 网格 + 去噪轨迹。
    η 可传任意实数；η≠0 时每次结果不同，η=0 时连跑两次完全一致。
    """
    torch.manual_seed(SEED)                          # 固定种子 → 起点噪声可复现
    model = load_model()
    sched = make_schedule()

    x, traj = ddim_sample(
        model, sched, shape=(n, 1, 28, 28),
        ddim_steps=ddim_steps, eta=eta, device=DEVICE,
    )
    # 画图函数使用 DDPM 中的函数即可
    save_grid(x, n=int(n ** 0.5), save_path=DDIM_SAMPLES_PNG)
    plot_sampling_trajectory(traj, save_path=DDIM_TRAJ_PNG)
    print("[sample_ddim] 完成：%d 步采样（DDPM 需 1000 步），eta=%s" % (ddim_steps, eta))
    return x


def interpolate_demo(n=4, ddim_steps=50):
    """
    两个随机起点之间插值，展示 DDIM 确定性带来的语义连续变化。
       从左到右数字平滑过渡，而不是一堆无关噪声。
       因为 η=0 时 x_T 与生成图是一一对应的，隐空间才有连续语义。
       DDPM 每步都抽新噪声，做不出这种可控插值。
    """
    torch.manual_seed(SEED)
    model = load_model()
    sched = make_schedule()
    # n*n 张图正好填满 n×n 网格（save_grid 要求图片数 >= 网格数）
    imgs = ddim_interpolate(model, sched, n=n * n,
                            ddim_steps=ddim_steps, eta=0.0,
                            seed=SEED, device=DEVICE)
    save_grid(imgs, n=n, save_path=DDIM_INTERP_PNG)
    return imgs


def plot_inversion_compare(x0, x_recon, save_path=DDIM_INVERSION_PNG):
    """
    画反演对比图：上排原图、下排重建图（2 行 × n 列）。
       η=0 且模型训练充分时，下排应和上排几乎一样，证明DDIM 反演是可逆编码。
    """
    # 反归一化（和 sample.save_grid 完全一样的处理），并搬到 CPU 供 matplotlib 用
    a = (x0.clamp(-1, 1) * 0.5 + 0.5).cpu()
    b = (x_recon.clamp(-1, 1) * 0.5 + 0.5).cpu()
    n = a.shape[0]
    fig, axs = plt.subplots(2, n, figsize=(n * 1.4, 2.8))
    axs = axs.reshape(2, n)                          # n=1 时也要保持 2D，方便统一索引
    for j in range(n):
        axs[0, j].imshow(a[j, 0], cmap="gray")      # 上排：真实图
        axs[1, j].imshow(b[j, 0], cmap="gray")      # 下排：反演后重建的图
        axs[0, j].axis("off")
        axs[1, j].axis("off")
    axs[0, 0].set_ylabel("x0")                      # 只在第一列标行名，避免重复
    axs[1, 0].set_ylabel("recon")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[sample_ddim] 已保存反演对比图 -> {save_path}")


def inversion_demo(n=4, ddim_steps=50, eta=0.0):
    """
    真实图 → 反演 → 再采样 → 重建，并量化误差。

    流程：
        x_0 (来自 MNIST，[-1,1])
          --ddim_invert-->  x_T (隐变量，看起来像噪声)
          --ddim_sample-->  x_0' (重建图)
        err = mean(|x_0 - x_0'|)

    ⚠️ 只有 η=0 时反演与采样才严格互逆。η≠0 会在反演和采样里各注入
       随机噪声，重建结果会「相似但不同」，这适合做随机图像编辑，
       但不适合用来验证可逆性。
    """
    torch.manual_seed(SEED)
    model = load_model()
    sched = make_schedule()

    # ① 取 n 张真实图片。标签 _ 用不上：无条件生成/反演都不看类别。
    from dataset import get_mnist_dataloader        # 延迟 import：只有这个演示需要数据
    loader = get_mnist_dataloader()
    x0, _ = next(iter(loader))
    x0 = x0[:n].to(DEVICE)                          # 只取前 n 张，形状 (n,1,28,28)

    # ② 反演：把真实图沿 τ 升序方向编码成隐变量 x_T
    x_T, _ = ddim_invert(model, x0, sched, ddim_steps=ddim_steps,
                         eta=eta, device=DEVICE)

    # ③ 重建：把刚得到的 x_T 作为起点，沿 τ 降序方向采样回来
    x_recon, _ = ddim_sample(model, sched, shape=x0.shape,
                             ddim_steps=ddim_steps, eta=eta,
                             x_T=x_T, device=DEVICE)

    # ④ 量化误差并画图。MAE 越小说明反越接近可逆；训练充分的模型
    #    在 50~100 步时通常能到较小值。
    err = (x0 - x_recon).abs().mean().item()
    print("[sample_ddim] 反演重建 MAE = %.4f  (steps=%d, eta=%s)"
          % (err, ddim_steps, eta))
    if eta != 0:
        print("  提示：eta≠0 时反演含随机性，不保证精确重建；想验证可逆性请用 --eta 0")
    plot_inversion_compare(x0, x_recon)
    return x_recon


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDIM 采样 / 插值 / 反演）")
    parser.add_argument("--steps", type=int, default=50,
                        help="采样步数，默认 50；DDPM 固定 1000 步")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="随机性系数，可为任意实数：0=确定性，1≈DDPM，"
                             "负数按 |eta| 处理，>1 噪声主导")
    parser.add_argument("--n", type=int, default=16, help="生成几张图（默认 16）")
    parser.add_argument("--interp", action="store_true",
                        help="改为生成隐空间插值图")
    parser.add_argument("--inversion", action="store_true",
                        help="改为反演 + 重建对比")
    args = parser.parse_args()

    # 三种演示互斥，按优先级选一个；--inversion 与 --interp 都给了时以反演为准
    if args.inversion:
        inversion_demo(n=4, ddim_steps=args.steps, eta=args.eta)
    elif args.interp:
        interpolate_demo(n=4, ddim_steps=args.steps)
    else:
        generate(n=args.n, ddim_steps=args.steps, eta=args.eta)
