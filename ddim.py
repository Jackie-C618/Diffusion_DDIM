"""
================================================================================
ddim.py —— DDIM 采样器 / 反演器（本文件夹唯一的新逻辑）
================================================================================

【DDPM 与 DDIM 区别】
    DDPM 的反向过程是马尔可夫链：t 与 t-1 必须相邻，每步都抽新噪声 z。
    DDIM 构造的是非马尔可夫过程：允许从 t 直接跳到子序列里任意更早点 t，且 η=0 时完全不抽噪声 → 确定性 → x_T 与生成图一一对应。

【公式】
    记 t 为噪声更多的一步、s 为噪声更少的一步（s < t）。

    (A) 由 x_t 与预测噪声反推干净图（采样与反演都用）：
            pred_x0 = ( x_t - √(1-ᾱ_t) · ε_θ(x_t,t) ) / √ᾱ_t

    (B) DDIM 采样，从 t 走到 s：
            σ       = |η| · √( (1-ᾱ_s)/(1-ᾱ_t) ) · √( 1 - ᾱ_t/ᾱ_s )
            x_s     = √ᾱ_s · pred_x0 + √(1-ᾱ_s - σ²) · ε_θ + σ · z
        η=0 ⇒ σ=0 ⇒ 完全确定性；η=1 ⇒ σ 等于 DDPM 的后验标准差。

    (C) DDIM 反演，从 s 倒推回 t（时间增大方向）：
            ε_s     = ε_θ(x_s, s)                 ← 注意用 s 作为时间步
            pred_x0 = ( x_s - √(1-ᾱ_s)·ε_s ) / √ᾱ_s
            x_t     = √ᾱ_t · pred_x0 + √(1-ᾱ_t - σ²) · ε_s + σ · z
        用同一个 ε_s 同时做「预测 x_0」和「重新加噪」，η=0 时
        反演与采样互为逆运算（忽略模型误差与离散化误差）。

【η 可取任意实数】
    论文把 η 当作 [0,1] 的插值系数，本次复现采用如下规则：
      · η = 0        σ=0，完全确定性，可逆
      · 0 < η < 1    介于确定性 DDIM 与 DDPM 之间
      · η = 1        σ≈DDPM 后验标准差，行为最接近 DDPM（但仍可跳步）
      · η > 1        σ² 可能超过「目标步的噪声预算」，方向系数被max(...,0) 截断为 0，噪声主导、画质变糊但不会 NaN
      · η < 0        标准差必须非负，故实际用 |η|。因为 z~N(0,I) 关于 0
                     对称，σ·z 与 -σ·z 同分布，所以 η 与 |η| 统计等价。
    注：判断「要不要抽 z」必须看 |η|>0，不能写 eta>0，
       否则负 η 时噪声项被静默丢掉、退化成错误的确定性路径

【关于反演的说明】
    标准 DDIM inversion 只定义 η=0（确定性才可逆）。本实现的 (C) 式
    把 σ 项也带进反演，是对称化的实验性扩展：η≠0 时反演会注入随机性，
    再采样回来就不能精确重建原图（会得到「相似但不同」的结果），
    这反而可用于带随机性的图像编辑，想要精确重建请用 η=0

================================================================================
"""
import math                                  
import torch                        

from config import T, DEVICE           
from schedule import make_schedule  


# ----------------------------------------------------------------------------
# 零件 1：时间步子序列 τ
# ----------------------------------------------------------------------------
def make_ddim_timesteps(T=T, ddim_steps=50):
    """生成 DDIM 采样的时间步子序列 τ（升序，含 0 和 T-1）。

    例：T=1000, ddim_steps=50 → [0, 20, 40, 61, 81, ..., 999]（间隔约 20）
      · 采样时从大到小遍历：999 → ... → 0（噪声 → 干净图）
      · 反演时从小到大遍历：0 → ... → 999（干净图 → 噪声）

    DDPM 必须走完 0..999 全部 1000 步；DDIM 只走 τ 里这 50 个点，
    所以网络只被调用 50 次 —— 这就是加速的来源。

    返回：python list[int]，长度 = ddim_steps
    """
    assert 2 <= ddim_steps <= T, "ddim_steps 需在 [2, T] 之间"
    # linspace 一定包含两端：起点 0、终点 T-1。.long() 向下取整，与主流实现一致。
    # 用浮点 linspace 再取整，比手动 range(0,T,step) 更能保证末点恰好是 T-1。
    return torch.linspace(0, T - 1, ddim_steps).long().tolist()


# ----------------------------------------------------------------------------
# 零件 2：一对时间步的 σ（采样与反演共用，保证两者用的是同一个定义）
# ----------------------------------------------------------------------------
def _sigma(eta, ab_t, ab_s):
    """计算「从 t（更噪）↔ s（更干净）」这一对时间步的随机噪声标准差。

        σ(η) = |η| · √( (1-ᾱ_s)/(1-ᾱ_t) ) · √( 1 - ᾱ_t/ᾱ_s )

    参数：
        eta : 任意实数。取 abs 是因为标准差必须非负，且 z 分布对称
        ab_t: ᾱ_t，t 更噪 → 数值更小
        ab_s: ᾱ_s，s 更干净 → 数值更大
    返回：非负 python float

    为什么两个根号：
        · 第一个根号把「目标步的噪声预算 (1-ᾱ_s)」按当前步归一化
        · 第二个根号来自 ᾱ_t/ᾱ_s 的比值，保证 η=1 时 σ 正好等于
          DDPM 的后验标准差 σ̃_t = √((1-ᾱ_s)/(1-ᾱ_t)·β_t)
    分母：若 schedule 把 ᾱ_0 置成严格 1，t=0 时 1-ᾱ_t=0；
    这里用 max(...,1e-12) 兜底，避免除零（正常路径不会走到）。
    """
    eta_abs = abs(eta)                              # 负 η 与 |η| 统计等价（z 对称）
    denom = max(1 - ab_t, 1e-12)                    # 当前步的噪声量，作为分母
    ratio = max(1 - ab_t / ab_s, 0)                 # 恒为正（ᾱ_t < ᾱ_s）；max 防浮点误差
    return eta_abs * math.sqrt((1 - ab_s) / denom) * math.sqrt(ratio)


# ----------------------------------------------------------------------------
# 零件 3：DDIM 采样（纯噪声 → 生成图）
# ----------------------------------------------------------------------------
@torch.no_grad()                                    # 采样是推理，不建计算图
def ddim_sample(model, sched=None, shape=(1, 1, 28, 28),
                ddim_steps=50, eta=0.0, steps_show=6,
                x_T=None, device=DEVICE):
    """
    DDIM 反向采样：从纯噪声出发，只走 ddim_steps 步生成图片。
    """
    if sched is None:
        sched = make_schedule()

    # 另注意：DDPM 训练时一个 batch 里每个样本 t 不同，q_sample 必须
    # gather+reshape；而采样时整批共用同一个 t，直接索引取出 0-D 标量张量
    # 就能与 (B,C,H,W) 广播。对照 diffusion.q_sample 看这个区别很清楚。
    alpha_bars = sched["alpha_bars"]
    sqrt_ab = sched["sqrt_alpha_bars"]
    sqrt_1mab = sched["sqrt_one_minus_alpha_bars"]

    tau = make_ddim_timesteps(len(alpha_bars), ddim_steps)  # 升序子序列
    B = shape[0]
    # 起点 x_T：外部没给就抽 N(0,I)。这个先验必须和前向过程 t=T 的
    # 边际分布对齐（要求 ᾱ_{T-1}≈0），否则生成会崩。
    x = torch.randn(shape, device=device) if x_T is None else x_T.to(device)
    traj = []
    interval = max(1, len(tau) // steps_show)       # 每隔几步存一帧，别存满 50 张

    # 从最噪的一端（tau[-1] ≈ T-1）走向最干净的一端（tau[0] = 0）
    for i in range(len(tau) - 1, -1, -1):
        t = tau[i]                                  # 当前步（更噪）
        t_prev = tau[i - 1] if i > 0 else None      # 目标步（更干净）；None 表示终点 x_0

        # ----  让网络预测这一步掺进去的噪声 ----
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)  # 整批同一个 t
        pred_eps = model(x, t_batch)                # ε_θ(x_t, t)，形状与 x 相同

        # ----  由前向公式 x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε 反解出 x_0 ----
        # 这是把闭式加噪公式「倒着解」：把 ε 换成网络的预测 ε̂，
        # 得到的 pred_x0 就是网络当前认为的干净图。
        pred_x0 = (x - sqrt_1mab[t] * pred_eps) / sqrt_ab[t]
        # 可选：把 pred_x0 裁到 [-1,1] 能稳住未收敛模型，但标准 DDIM 不裁，
        # 这里保持和论文一致（训练好的模型自然落在区间内）。

        if t_prev is None:
            # ---- a 终点：ᾱ_t' = 1 ⇒ √(1-1)=0、σ=0，于是 x = pred_x0 ----
            # 显式写出来，既贴合论文边界，也避开 t=0 时 1-ᾱ_t=0 的除零。
            x = pred_x0
        else:
            # ---- b 常规步：按 DDIM 公式「重新加噪」到 t' ----
            ab_t = alpha_bars[t].item()             # ᾱ_t（当前，更噪）
            ab_prev = alpha_bars[t_prev].item()     # ᾱ_t'（目标，更干净）

            # σ：这一对时间步的随机噪声标准差，|η| 保证非负（见 _sigma）
            sigma = _sigma(eta, ab_t, ab_prev)
            # 「指向 x_t' 的方向」系数：目标步的噪声预算 (1-ᾱ_t') 扣掉 σ²。
            # max(...,0) 是 η>1 时的保护：σ² 超标就把方向项截为 0，
            # 结果变糊但不会出现 NaN。
            dir_coef = math.sqrt(max(1 - ab_prev - sigma ** 2, 0))
            #  判断要不要抽 z 必须看 |η|。写成 eta>0 的话，负 η 会静默
            #   丢掉噪声项，退化成一条错误的确定性路径（旧版 bug，已修）。
            z = torch.randn_like(x) if abs(eta) > 0 else 0.0

            # 三项相加：预测的信号 + 指向目标步的方向 + 随机扰动
            x = math.sqrt(ab_prev) * pred_x0 + dir_coef * pred_eps + sigma * z

        if i % interval == 0:
            traj.append(x.clone())                  # clone：x 之后会被覆盖，必须拷贝

    return x, traj


# ----------------------------------------------------------------------------
# 零件 4：DDIM 反演（真实图 → 隐变量 x_T）
# ----------------------------------------------------------------------------
@torch.no_grad()
def ddim_invert(model, x0, sched=None, ddim_steps=50, eta=0.0,
                steps_show=6, device=DEVICE):
    """DDIM 反演 / 编码：把一张真实图片沿确定性路径倒推回隐变量 x_T。

    用途：
      重建：x_0 --反演--> x_T --采样--> x_0'，η=0 时 x_0' 应接近 x_0
      编辑：在 x_T 上做一点修改（或与另一张图的 x_T 插值）再采样回去
      语义插值：两张真实图各自反演成 x_T，在隐空间插值后生成过渡图

    与采样方向的对照：
        采样：用 x_t 处的 ε_θ(x_t, t)，走向更小的 t'
        反演：只有 x_s 已知，所以用 ε_θ(x_s, s)（时间步填 s）近似，再把它重新加噪到更大的 t。不是 t
    """
    if sched is None:
        sched = make_schedule()

    alpha_bars = sched["alpha_bars"]
    sqrt_ab = sched["sqrt_alpha_bars"]
    sqrt_1mab = sched["sqrt_one_minus_alpha_bars"]

    tau = make_ddim_timesteps(len(alpha_bars), ddim_steps)  # 升序：从 0 走到 T-1
    B = x0.shape[0]
    x = x0.to(device)                               # 反演起点是真实图，不是随机噪声
    traj = []
    interval = max(1, len(tau) // steps_show)

    # 与采样相反：这里从小到大遍历 τ，每步从 s 跳到 t
    for i in range(len(tau) - 1):
        s = tau[i]                                  # 当前已知点
        t = tau[i + 1]                             # 要推出的下一点

        # ① 在已知点 x_s 上让网络预测噪声。时间步填 s，不是 t。
        #    因为 x_t 此刻还不存在，无法计算 ε_θ(x_t,t)；用 ε_θ(x_s,s)
        #    作近似正是 DDIM inversion 的标准做法。
        t_batch = torch.full((B,), s, device=device, dtype=torch.long)
        pred_eps = model(x, t_batch)                # ε_s

        # ② 用同一个 ε_s 反解出噪声更少的图（把 t 换成 s）
        pred_x0 = (x - sqrt_1mab[s] * pred_eps) / sqrt_ab[s]

        # ③ 取这对 (t, s) 的 ᾱ。注意 t 噪声更多 → ᾱ_t < ᾱ_s。
        ab_t = alpha_bars[t].item()
        ab_s = alpha_bars[s].item()

        # ④ σ 与采样共用同一个 _sigma，保证「同一对时间步」定义一致。
        #    这里分母是 (1-ᾱ_t)（t 噪声更强，值更大），所以 s=0 时也不会除零。
        sigma = _sigma(eta, ab_t, ab_s)
        # ⑤ 方向系数：目标步 t 的噪声预算 (1-ᾱ_t) 扣掉 σ²。
        dir_coef = math.sqrt(max(1 - ab_t - sigma ** 2, 0))
        z = torch.randn_like(x) if abs(eta) > 0 else 0.0

        # ⑥ 重新加噪到 t：√ᾱ_t·pred_x0 是信号项，dir_coef·ε_s 是方向项。
        #    η=0 时 σ=0，这一步与采样公式严格互逆。
        x = math.sqrt(ab_t) * pred_x0 + dir_coef * pred_eps + sigma * z

        if i % interval == 0:
            traj.append(x.clone())

    # 循环结束时 x 对应 tau[-1] ≈ T-1，即隐变量 x_T
    return x, traj


# ----------------------------------------------------------------------------
# 零件 5：球面线性插值（给隐空间插值用）
# ----------------------------------------------------------------------------
def slerp(a, b, lam):
    """
    球面线性插值：在高维隐空间里把 a、b 当作方向做插值。
    a, b : (B,C,H,W)；lam : 0~1 的标量。lam=0 → a，lam=1 → b。
    为什么不用线性插值：高维高斯样本的模长集中在 √d 附近，
    直接 lerp 会让中间点模长变短、偏离 N(0,I)，生成质量下降；
    slerp 沿「球面」走，保持模长。
    """
    shape = a.shape
    fa, fb = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)  # 展平成 (B,D)
    # 归一化到单位方向；clamp 防止零向量除零
    ua = fa / fa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ub = fb / fb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    dot = (ua * ub).sum(-1, keepdim=True).clamp(-1, 1)  # 夹角余弦，clamp 防 acos 越界
    theta = torch.acos(dot)                              # 夹角
    so = torch.sin(theta).clamp(min=1e-8)                # 分母；接近 0 时退化成线性插值
    w0 = torch.sin((1 - lam) * theta) / so
    w1 = torch.sin(lam * theta) / so
    return (w0 * fa + w1 * fb).reshape(shape)            # 折回 (B,C,H,W)


# ----------------------------------------------------------------------------
# 零件 6：两个随机起点之间插值
# ----------------------------------------------------------------------------
def ddim_interpolate(model, sched=None, n=8, ddim_steps=50, eta=0.0,
                     seed=42, device=DEVICE):
    """
    两个随机起点之间插值，看生成结果平滑过渡。
       DDIM 是确定性的 ⇒ x_T 构成一个有语义的"隐空间"，在隐空间里插值，
       生成的数字会连续变形。DDPM 每步都抽新噪声，做不到这种可控插值。
       返回 (n,1,28,28) 的张量，可直接用 sample.save_grid 画成网格。

    注意：η≠0 时插值不再平滑（每步都注入新随机性），想看语义过渡请用 η=0。
    """
    if sched is None:
        sched = make_schedule()
    torch.manual_seed(seed)
    z0 = torch.randn(1, 1, 28, 28, device=device)     # 隐变量 A
    z1 = torch.randn(1, 1, 28, 28, device=device)     # 隐变量 B
    outs = []
    for lam in torch.linspace(0, 1, n, device=device):  # lam 从 0 到 1 共 n 个点
        z = slerp(z0, z1, lam.item())                   # 在隐空间取插值点
        x, _ = ddim_sample(model, sched, shape=(1, 1, 28, 28),
                           ddim_steps=ddim_steps, eta=eta, x_T=z, device=device)
        outs.append(x)
    return torch.cat(outs, dim=0)                       # (n,1,28,28)


# # ----------------------------------------------------------------------------
# # 自检：不依赖 ckpt，用临时小 UNet 把新加的能力全部过一遍
# # ----------------------------------------------------------------------------
# def test_ddim():
#     """验证：跳步 / 确定性 / 任意实数 η / 反演往返 / 数值有限。"""
#     from model import UNet

#     device = DEVICE
#     sched = make_schedule()
#     # 极小模型只为验证「公式能跑通」，不追求生成质量；
#     # 用 base=8、depth=1、关掉注意力，CPU 上几秒就能跑完。
#     model = UNet(base=8, depth=1, attn_at=-1).to(device)
#     model.eval()
#     shape = (2, 1, 28, 28)

#     # ① 跳步：50 步真的只用 50 个点，且两端是 0 和 T-1
#     tau = make_ddim_timesteps(T, 50)
#     print("[ddim] τ 子序列 (共 %d 个): 前5=%s ... 末=%d" % (len(tau), tau[:5], tau[-1]))
#     assert len(tau) == 50 and tau[-1] == T - 1 and tau[0] == 0

#     # ② 确定性：同一个 x_T、eta=0，两次采样结果必须逐像素完全相同
#     torch.manual_seed(0)
#     x_T = torch.randn(shape, device=device)
#     x1, traj = ddim_sample(model, sched, shape=shape, ddim_steps=10,
#                            eta=0.0, x_T=x_T, device=device)
#     x2, _ = ddim_sample(model, sched, shape=shape, ddim_steps=10,
#                         eta=0.0, x_T=x_T, device=device)
#     assert torch.equal(x1, x2), "eta=0 时必须完全确定性"
#     print("[ddim] 确定性 ✅（同 x_T 两次结果逐像素相同）")

#     # ③ 形状 / 有限值 / 轨迹帧数
#     assert x1.shape == shape
#     assert torch.isfinite(x1).all(), "输出出现 NaN/Inf，检查 σ 的根号内是否变负"
#     assert len(traj) >= 1
#     print("[ddim] 形状 %s ✅  轨迹 %d 帧 ✅  数值有限 ✅" % (tuple(x1.shape), len(traj)))

#     # ④ η 取任意实数：负、零、分数、1、大于 1，全部不能出 NaN
#     for eta in (-2.0, -1.0, -0.3, 0.0, 0.5, 1.0, 2.0):
#         xe, _ = ddim_sample(model, sched, shape=shape, ddim_steps=10,
#                             eta=eta, x_T=x_T, device=device)
#         assert torch.isfinite(xe).all(), "eta=%s 出现 NaN/Inf" % eta
#     print("[ddim] η ∈ {-2,-1,-0.3,0,0.5,1,2} 全部数值有限 ✅")

#     # ⑤ 负 η 必须真的有随机性，不能被静默变成确定性（旧版 bug 的回归测试）
#     xneg, _ = ddim_sample(model, sched, shape=shape, ddim_steps=10,
#                           eta=-1.0, x_T=x_T, device=device)
#     assert not torch.equal(x1, xneg), "eta=-1 应引入随机性"
#     print("[ddim] η<0 噪声项生效 ✅（|η| 处理正确）")

#     # ⑥ 反演：形状正确、数值有限、轨迹非空
#     x0 = torch.rand(shape, device=device) * 2 - 1      # 伪造一张 [-1,1] 的「真实图」
#     z_inv, inv_traj = ddim_invert(model, x0, sched, ddim_steps=10,
#                                   eta=0.0, device=device)
#     assert z_inv.shape == shape and torch.isfinite(z_inv).all()
#     assert len(inv_traj) >= 1
#     print("[ddim] 反演 x_0 → x_T ✅  形状 %s  轨迹 %d 帧"
#           % (tuple(z_inv.shape), len(inv_traj)))

#     # ⑦ 往返：把反演得到的 x_T 再采样回来，数值必须有限。
#     #    这里打印重建误差供参考；小模型没训练过，误差不会很小，
#     #    用真实 ckpt 时（见 sample_ddim.py --inversion）才有意义。
#     x_recon, _ = ddim_sample(model, sched, shape=shape, ddim_steps=10,
#                              eta=0.0, x_T=z_inv, device=device)
#     err = (x0 - x_recon).abs().mean().item()
#     assert torch.isfinite(x_recon).all()
#     print("[ddim] 往返 x_0 → x_T → x_0' ✅  重建 MAE = %.4f（未训练模型，仅示意）" % err)

#     # ⑧ 反演在 η≠0 时也不能出 NaN
#     for eta in (-1.0, 0.5, 1.5):
#         zi, _ = ddim_invert(model, x0, sched, ddim_steps=10,
#                             eta=eta, device=device)
#         assert torch.isfinite(zi).all(), "反演 eta=%s 出现 NaN/Inf" % eta
#     print("[ddim] 反演 η ∈ {-1, 0.5, 1.5} 数值有限 ✅")

#     print("[ddim] 自检全部通过 ✅")


# if __name__ == "__main__":
#     test_ddim()
