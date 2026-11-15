"""
================================================================================
schedule.py —— 计算前向加噪所需的参数
================================================================================

【在整条链路里的位置】
            config.py（T / BETA_START / BETA_END）
                        ↓
           【schedule.py】 → 返回一张 (T,) 常数的字典 sched
                        ↓
        ┌───────────────┼────────────────┬─────────────────┐
   diffusion.py      train.py        ddim.py         sample.py
   (q_sample 加噪)   (训练时加噪)   (DDIM 采样/反演)   (DDPM 采样)

【DDIM 相关】
    ddim.py 只需要 alpha_bars / sqrt_alpha_bars / sqrt_one_minus_alpha_bars三个参数。
    因为 DDIM 的公式只涉及全局的 ᾱ ，完全不用逐局部的 β_t。
    这也解释了为什么 DDIM 能跳步：它不依赖相邻步的马尔可夫转移。

================================================================================
"""
import torch                                                    # 张量运算
from config import T, BETA_START, BETA_END, DEVICE                # 从 config 拿超参，避免两处写死不一致


def make_schedule(T=T, beta_start=BETA_START, beta_end=BETA_END, device=DEVICE):
    """计算噪声 schedule，返回 dict，所有值形状都是 (T,)。

    参数写成默认值的形式，是为了既能被其它模块无参调用（用 config 的配置），
    也能在测试里临时传 T=200 之类做小实验，不用改 config.py。

    对应论文符号：
      betas                     : β_t
      alphas                    : α_t = 1 - β_t
      alpha_bars                : ᾱ_t = α_1·α_2·…·α_t
      sqrt_alpha_bars           : √ᾱ_t
      sqrt_one_minus_alpha_bars : √(1 - ᾱ_t)
      sqrt_recip_alphas         : 1/√α_t
      tilde_betas               : ~β_t
    """
    #  β_t：在 [beta_start, beta_end] 之间等间隔取 T 个数（线性 schedule）。
    #  Improved DDPM 会换成 cosine schedule，让 ᾱ_t 下降更平缓。
    betas = torch.linspace(beta_start, beta_end, T, device=device)  # 等间距

    alphas = 1.0 - betas

    alpha_bars = torch.cumprod(alphas, dim=0)            # 累乘，在 dim = 0 维度进行。得到 ᾱ_t

    sqrt_alpha_bars = torch.sqrt(alpha_bars)

    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)

    # torch.cat tensor拼接
    alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])  # ᾱ_{t-1}，首位补 ᾱ_{-1}=1

    tilde_betas = (1.0 - alpha_bars_prev) / (1.0 - alpha_bars) * betas

    tilde_betas[0] = 0.0

    # 打包成字典返回
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "tilde_betas": tilde_betas,
    }

# def plot_schedule(sched, save_path="schedule.png"):
#     """画图：
#        噪声 schedule 曲线
#        β_t（红）：每一步加多少噪声
#        ᾱ_t（绿）：累积保留的信号比例
#        √(1-ᾱ_t)（蓝）：噪声权重
#        信号随时间线性消失、噪声逐渐增强，也是换cosine schedule 时对比曲线形状是否更平滑的依据。
#     """
#     import matplotlib.pyplot as plt                               # 只在画图时才 import
#     t = torch.arange(len(sched["betas"])).cpu().numpy()           # x 轴：0..T-1；numpy 才能喂给 plt
#     plt.figure(figsize=(7, 4))
#     plt.plot(t, sched["betas"].cpu().numpy(), label=r"$\beta_t$")
#     plt.plot(t, sched["alpha_bars"].cpu().numpy(), label=r"$\bar\alpha_t$")
#     plt.plot(t, sched["sqrt_one_minus_alpha_bars"].cpu().numpy(),
#              label=r"$\sqrt{1-\bar\alpha_t}$")
#     plt.xlabel("timestep t")
#     plt.ylabel("value")
#     plt.title("Noise Schedule (linear)")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)                                       # 存盘而非弹窗（无界面的服务器环境可运行）
#     plt.close()                                                  # 及时释放，避免循环里内存堆积
#     print(f"[schedule] 已保存曲线图 -> {save_path}")

# if __name__ == "__main__":
#     plot_schedule(make_schedule())
