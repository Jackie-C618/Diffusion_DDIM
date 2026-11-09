"""
ddim.py — DDIM 采样器（本文件夹唯一的新逻辑）

★ 最重要的一句话：
    DDIM **不改训练、不改模型、不改 schedule**，只把「反向采样」这一步换掉。
    所以 config / schedule / dataset / train / diffusion / sample 全部
    与 Diffusion_DDPM 逐字节相同；model.py 只补回一处 DDPM 版既有的 bug 修复
    （MidBlock 透传 t_emb，见 README），与 DDIM 无关。
    直接复用同一份 ckpt_last.pt 就能生成。

DDPM 的反向过程是「马尔可夫链」：每一步都必须走、且每一步都重新抽噪声 z。
DDIM 构造了另一个**非马尔可夫**的前向过程，它和 DDPM 有完全相同的
边缘分布 q(x_t|x_0)，因此可以用同一个 ε_θ，但采样时：
    · 允许 **跳步**（1000 步 → 50 步，快 20 倍）
    · η=0 时 **不加随机噪声**，整个过程确定性（同一个 x_T 永远得到同一张图）

核心公式（Song, Meng & Ermon, ICLR 2021, Eq.12 / Eq.16）：
    1) 由 x_t 和预测噪声反推干净图：
         pred_x0 = ( x_t - √(1-ᾱ_t) · ε_θ(x_t,t) ) / √ᾱ_t
    2) 再"重新加噪"到子序列里的前一步 t'（t' < t）：
         σ       = η · √( (1-ᾱ_t')/(1-ᾱ_t) ) · √( 1 - ᾱ_t/ᾱ_t' )
         x_t'    = √ᾱ_t' · pred_x0 + √(1-ᾱ_t' - σ²) · ε_θ + σ · z

    η=0 → σ=0 → 上式退化为完全确定性：
         x_t' = √ᾱ_t' · pred_x0 + √(1-ᾱ_t') · ε_θ
    η=1 → σ 接近 DDPM 的后验标准差，行为接近 DDPM（但仍可跳步）。
    最后一步 t'=0（记作 t'=-1，ᾱ=1）时，√(1-1)=0，直接 x = pred_x0。

💡 单独测试本文件：
    python ddim.py
    会用一个临时小 UNet 验证「确定性」「跳步」「无 NaN」，不需要 ckpt。
"""
import math
import torch

from config import T, DEVICE
from schedule import make_schedule


def make_ddim_timesteps(T=T, ddim_steps=50):
    """生成 DDIM 采样的时间步子序列 τ（升序，含 0 和 T-1）。

    例：T=1000, ddim_steps=50 → [0, 20, 40, ..., 999]（间隔约 20）
    采样时我们**从大到小**遍历这个序列，即 999 → ... → 0。
    DDPM 必须走完 0..999 全部 1000 步；DDIM 只走 τ 里这 50 个点。

    返回：python list[int]，长度 = ddim_steps
    """
    assert 2 <= ddim_steps <= T, "ddim_steps 需在 [2, T] 之间"
    # linspace 一定包含两端：起点 0、终点 T-1。.long() 向下取整，与主流实现一致。
    return torch.linspace(0, T - 1, ddim_steps).long().tolist()


@torch.no_grad()
def ddim_sample(model, sched=None, shape=(1, 1, 28, 28),
                ddim_steps=50, eta=0.0, steps_show=6,
                x_T=None, device=DEVICE):
    """DDIM 反向采样：从纯噪声出发，只走 ddim_steps 步生成图片。

    参数（和 diffusion.sample 尽量对齐，方便并排对照）：
        model       : 训练好的 ε_θ，接口 model(x, t) → 预测噪声
        sched       : make_schedule() 的返回值；None 时自动算
        shape       : 生成图片的形状 (B,C,H,W)
        ddim_steps  : 采样步数（DDPM 固定是 T=1000 步）
        eta         : 随机性。0=确定性 DDIM；1≈DDPM
        steps_show  : 轨迹图记录几帧（仅用于可视化）
        x_T         : 可选，指定起始噪声。传入后可复现/对比同一个起点
        device      : cpu / cuda

    返回 (x, traj)：
        x    : 最终生成图 (B,C,H,W)，值域约 [-1,1]
        traj : 中间若干帧，可直接喂给 diffusion.plot_sampling_trajectory
    """
    if sched is None:
        sched = make_schedule()

    # 只需要 schedule 里这两张表：√ᾱ_t 和 √(1-ᾱ_t)，以及 ᾱ_t 本身。
    # 注意：DDPM 训练时一个 batch 里每个样本 t 不同，所以 q_sample 必须 gather+reshape；
    # 而采样时整个 batch 共用同一个 t，直接索引取出 0-D 标量张量即可与 (B,C,H,W) 广播。
    # 这正是「训练按样本索引、采样按整批索引」的区别，对照 diffusion.q_sample 看更清楚。
    alpha_bars = sched["alpha_bars"]
    sqrt_ab = sched["sqrt_alpha_bars"]
    sqrt_1mab = sched["sqrt_one_minus_alpha_bars"]

    tau = make_ddim_timesteps(len(alpha_bars), ddim_steps)
    B = shape[0]
    x = torch.randn(shape, device=device) if x_T is None else x_T.to(device)
    traj = []
    interval = max(1, len(tau) // steps_show)

    # 从最噪的一端（tau[-1] ≈ T-1）走向最干净的一端（tau[0] = 0）
    for i in range(len(tau) - 1, -1, -1):
        t = tau[i]
        t_prev = tau[i - 1] if i > 0 else None      # None 表示下一步就是干净图 x_0

        # ---- 1) 让网络预测这一步加进去的噪声 ----
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        pred_eps = model(x, t_batch)

        # ---- 2) 由前向公式 x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε 反解出 x_0 ----
        pred_x0 = (x - sqrt_1mab[t] * pred_eps) / sqrt_ab[t]
        # 可选：把 pred_x0 裁到 [-1,1] 能稳住未收敛模型，但标准 DDIM 不裁，
        # 这里保持和论文一致（训练好的模型自然落在区间内）。

        if t_prev is None:
            # ---- 3a) 终点：ᾱ_t' = 1 ⇒ √(1-1)=0、σ=0，于是 x = pred_x0 ----
            # 显式写出来，既贴合论文边界，也避开 t=0 时 1-ᾱ_t=0 的除零。
            x = pred_x0
        else:
            # ---- 3b) 常规步：按 DDIM 公式"重新加噪"到 t' ----
            ab_t = alpha_bars[t].item()
            ab_prev = alpha_bars[t_prev].item()

            # η 控制的随机噪声标准差（论文 Eq.16）
            sigma = eta * math.sqrt((1 - ab_prev) / (1 - ab_t)) \
                        * math.sqrt(max(1 - ab_t / ab_prev, 0))
            # "指向 x_t' 的方向"系数；max 防止浮点误差导致根号内为负
            dir_coef = math.sqrt(max(1 - ab_prev - sigma ** 2, 0))
            z = torch.randn_like(x) if eta > 0 else 0.0

            x = math.sqrt(ab_prev) * pred_x0 + dir_coef * pred_eps + sigma * z

        if i % interval == 0:
            traj.append(x.clone())

    return x, traj


def slerp(a, b, lam):
    """球面线性插值：在高维隐空间里把 a、b 当作方向做插值（比直线插值更自然）。

    a, b : (B,C,H,W)；lam : 0~1 的标量。lam=0 → a，lam=1 → b。
    """
    shape = a.shape
    fa, fb = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    # 归一化到单位方向
    ua = fa / fa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ub = fb / fb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dot = (ua * ub).sum(-1, keepdim=True).clamp(-1, 1)
    theta = torch.acos(dot)
    so = torch.sin(theta).clamp(min=1e-8)
    w0 = torch.sin((1 - lam) * theta) / so
    w1 = torch.sin(lam * theta) / so
    return (w0 * fa + w1 * fb).reshape(shape)


def ddim_interpolate(model, sched=None, n=8, ddim_steps=50, eta=0.0,
                     seed=42, device=DEVICE):
    """【可选进阶】两个随机起点之间插值，看生成结果平滑过渡。

    ❖ 这反映什么：
       DDIM 是确定性的 ⇒ x_T 构成一个有语义的"隐空间"，在隐空间里插值，
       生成的数字会连续变形。DDPM 每步都抽新噪声，做不到这种可控插值。
       返回 (n,1,28,28) 的张量，可直接用 sample.save_grid 画成网格。
    """
    if sched is None:
        sched = make_schedule()
    torch.manual_seed(seed)
    z0 = torch.randn(1, 1, 28, 28, device=device)
    z1 = torch.randn(1, 1, 28, 28, device=device)
    outs = []
    for lam in torch.linspace(0, 1, n, device=device):
        z = slerp(z0, z1, lam.item())
        x, _ = ddim_sample(model, sched, shape=(1, 1, 28, 28),
                           ddim_steps=ddim_steps, eta=eta, x_T=z, device=device)
        outs.append(x)
    return torch.cat(outs, dim=0)


def test_ddim():
    """自检：不依赖 ckpt，用一个临时小 UNet 验证 DDIM 的三条核心性质。"""
    from model import UNet

    device = DEVICE
    sched = make_schedule()
    # 极小模型只为验证"公式能跑通"，不追求生成质量
    model = UNet(base=8, depth=1, attn_at=-1).to(device)
    model.eval()
    shape = (2, 1, 28, 28)

    # ① 跳步：50 步真的只用 50 个点（打印前 5 个和最后 1 个）
    tau = make_ddim_timesteps(T, 50)
    print("[ddim] τ 子序列 (共 %d 个): 前5=%s ... 末=%d" % (len(tau), tau[:5], tau[-1]))
    assert len(tau) == 50 and tau[-1] == T - 1 and tau[0] == 0

    # ② 确定性：同一个 x_T、eta=0，两次采样结果必须逐像素完全相同
    torch.manual_seed(0)
    x_T = torch.randn(shape, device=device)
    x1, traj = ddim_sample(model, sched, shape=shape, ddim_steps=10,
                           eta=0.0, x_T=x_T, device=device)
    x2, _ = ddim_sample(model, sched, shape=shape, ddim_steps=10,
                        eta=0.0, x_T=x_T, device=device)
    assert torch.equal(x1, x2), "eta=0 时必须完全确定性"
    print("[ddim] 确定性 ✅（同 x_T 两次结果逐像素相同）")

    # ③ 形状 / 有限值 / 轨迹帧数
    assert x1.shape == shape
    assert torch.isfinite(x1).all(), "输出出现 NaN/Inf，检查 σ 的根号内是否变负"
    assert len(traj) >= 1
    print("[ddim] 形状 %s ✅  轨迹 %d 帧 ✅  数值有限 ✅" % (tuple(x1.shape), len(traj)))

    # ④ eta=1（近似 DDPM）也不能出 NaN
    x3, _ = ddim_sample(model, sched, shape=shape, ddim_steps=10,
                        eta=1.0, x_T=x_T, device=device)
    assert torch.isfinite(x3).all()
    print("[ddim] eta=1 数值有限 ✅；eta=0 与 eta=1 结果不同 =",
          not torch.equal(x1, x3))
    print("[ddim] 自检全部通过 ✅")


if __name__ == "__main__":
    test_ddim()
