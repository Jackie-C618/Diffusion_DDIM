"""
================================================================================
diffusion.py —— DDPM 的前向加噪和反向采样
================================================================================
"""
import torch                                       

from config import T, DEVICE                     
from schedule import make_schedule             


def q_sample(x0, t, noise, sched=None):
    """
    前向过程
    参数：
        x0, noise : (B, C, H, W)，noise ε ~ N(0,I)
        t         : (B,) 
    返回：
        x_t       : (B, C, H, W)

    为什么不能真的循环 t 次：t 最大 999、一个 batch 128 个样本、
    每个样本 t 还不同，逐步循环既慢又难写成批处理。闭式解一步搞定。
    """
    if sched is None: 
        sched = make_schedule()

    # sched 里的表都是 1D 的 (T,)；t 是 (B,) 的一串索引。
    # gather(0, t) 按每个样本的 t 取出对应常数 → (B,)。
    # view 函数，第一个写-1就会自动计算，比如view(-1,3),其运行原理为，将函数维度除以3，得到具体数字
    # -1相当于一个占位符，意思是这一维度的具体大小由计算机算
    # .view(-1,1,1,1) 升成 (B,1,1,1) 后，才能和 (B,C,H,W) 的图片广播相乘。
    # 漏掉 reshape 会变成 (B,) 与 (B,C,H,W) 对齐失败，直接形状报错。
    # 对照 ddim.py：采样时整个 batch 共用同一个 t，就直接索引取标量，不需要 gather。
    sqrt_ab = sched["sqrt_alpha_bars"].gather(0, t).view(-1, 1, 1, 1)
    sqrt_1mab = sched["sqrt_one_minus_alpha_bars"].gather(0, t).view(-1, 1, 1, 1)
    return sqrt_ab * x0 + sqrt_1mab * noise         # x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε


# def test_q_sample():
#     """
#     自检：t=0 时 x_t≈x_0；t=T-1 时 x_t≈纯噪声。

#     这条断言会报错,beta_start设定为1e-4，ᾱ_0=0.9999 而非 1，
    
#     想让它通过，可在 schedule.py 里把 alpha_bars[0] 置为 1。
#     """
#     sched = make_schedule()
#     x0 = torch.ones(2, 1, 4, 4)                     # 用全 1 的小图，方便肉眼判断
#     noise = torch.randn(2, 1, 4, 4)                 # 标准正态噪声
#     xt0 = q_sample(x0, torch.tensor([0, 0]), noise, sched)
#     xtT = q_sample(x0, torch.tensor([T - 1, T - 1]), noise, sched)
#     assert torch.allclose(xt0, x0, atol=1e-5), "t=0 应几乎等于原图"
#     print("[diffusion] t=0 → x_t≈x0 ✅；t=T-1 → x_t≈噪声 ✅")


def plot_noising(x0, sched=None, save_path="noising.png", steps=(0, 250, 500, 750, 999)):
    """
    画图：同一张图在不同 t 下的加噪过程（保存 noising.png）。
    参数 steps 默认取 5 个代表点，横着排成一行方便对比。
    """
    import matplotlib.pyplot as plt
    if sched is None:
        sched = make_schedule()
    imgs = []
    for t in steps:
        # 加噪不需要梯度；每次重新抽一份噪声，模拟真实训练里的 ε
        with torch.no_grad():
            xt = q_sample(x0, torch.tensor([t]), torch.randn_like(x0), sched)
        # 显示前先做两件事：
        #   clamp(-1,1) 把可能的越界裁回数据区间（对称裁剪）
        #   *0.5+0.5   把 [-1,1] 反归一化成 [0,1]，matplotlib 才能正常显示
        imgs.append((xt[0, 0].clamp(-1, 1) * 0.5 + 0.5).cpu().numpy())
    fig, axs = plt.subplots(1, len(steps), figsize=(len(steps) * 2, 2))
    for ax, (t, im) in zip(axs, zip(steps, imgs)):
        ax.imshow(im, cmap="gray")                  # 单通道 → 灰度色表
        ax.set_title(f"t={t}")
        ax.axis("off")                              # 去掉坐标轴，只看图
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[diffusion] 已保存加噪过程图 -> {save_path}")


@torch.no_grad()                                    # 不画图
def sample(model, sched=None, shape=(1, 1, 28, 28), steps_show=5, device=DEVICE):
    """
    反向过程：从纯噪声 x_T 逐步去噪到 x_0。

    返回 (x, traj)：
      x    : 最终生成图 (B,C,H,W)，值域 [-1,1]
      traj : 中间若干帧，用于画去噪演化图
    """
    if sched is None:
        sched = make_schedule()
    x = torch.randn(shape, device=device)           # 从标准正态中采样一个噪声
    traj = []
    interval = max(1, T // steps_show)              # 每隔多少步存一帧，避免存 1000 张

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device)  # 整个 batch 共用同一个 t，填成 (B,)
        pred = model(x, t_batch)                    # ε_θ(x_t, t)：预测掺进去的噪声
        beta = sched["betas"][t]                    # β_t
        sac = sched["sqrt_recip_alphas"][t]          # 1/√α_t
        so = sched["sqrt_one_minus_alpha_bars"][t]   # √(1-ᾱ_t)
        # 最后一步 t=0 不再加噪声，否则会在已经去好噪声的图上又叠一层噪。
        # 原论文为 t_prev is None 时 x = pred_x0
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        # 后验均值 + 随机噪声：第一项是「预测均值」，第二项让采样保持随机性。
        x = sac * (x - (1 - beta) / so * pred) + torch.sqrt(sched["tilde_betas"][t]) * z
        if t % interval == 0:
            traj.append(x.clone())                  # .clone() 是必须的：x 后面会被原地覆盖
    return x, traj


def plot_sampling_trajectory(traj, save_path="sampling_traj.png"):
    """
    画图：展示 x_T → x_0 的去噪演化。

    """
    import matplotlib.pyplot as plt
    n = len(traj)
    fig, axs = plt.subplots(1, n, figsize=(n * 2, 2))
    for ax, im in zip(axs, traj):
        # 同样先 clamp 再反归一化；只取第 0 张图、第 0 个通道来展示
        ax.imshow((im[0, 0].clamp(-1, 1) * 0.5 + 0.5).cpu().numpy(), cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[diffusion] 已保存采样轨迹图 -> {save_path}")


if __name__ == "__main__":
    test_q_sample()
    # 使用一张随机图演示加噪过程
    demo = torch.randn(1, 1, 28, 28)
    plot_noising(demo)
