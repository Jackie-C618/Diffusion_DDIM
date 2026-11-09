# 🎯 DDIM on MNIST

> 在 [Diffusion_DDPM]的基础上复现 **DDIM**（Denoising Diffusion Implicit Models），
> **DDIM 不改训练，只改采样**
---


### 🟡 Bug修复`model.py` 的 `MidBlock`

复现时实测发现，`DDPM/model.py` 的 bottleneck 用了
`nn.Sequential(ResidualBlock, Attention, ResidualBlock)`，但 `Sequential`
只会把上一层输出喂给下一层，**第一个 `ResidualBlock` 收不到 `t_emb`**：

```
TypeError: ResidualBlock.forward() missing 1 required positional argument: 't_emb'
```
---

## 📂 目录结构

```
Diffusion_DDIM/
├── config.py          ⚙️ （同 DDPM）
├── schedule.py        📈 （同 DDPM）
├── model.py           🧠 （同 DDPM，仅补回 MidBlock 修复）
├── dataset.py         🗂️ （同 DDPM）
├── train.py           🚀 （同 DDPM）
├── diffusion.py       🔀 （同 DDPM）保留 DDPM sample() 作对照
├── sample.py          🖼️ （同 DDPM）DDPM 1000 步采样入口
├── ddim.py            ✨ DDIM 采样核心（唯一新逻辑）
├── sample_ddim.py     ✨ DDIM 采样入口（--steps / --eta / --interp）
├── .gitignore         🚫
└── README.md
```

---

## 🧮 DDPM vs DDIM 

**共同点**：

```
x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε          →  loss = MSE(ε, ε_θ(x_t, t))
```

**DDPM 反向一步**（马尔可夫链：`t` 与 `t-1` 必须相邻，每步都抽新噪声）：

```
x_{t-1} = (1/√α_t) · ( x_t - (β_t / √(1-ᾱ_t)) · ε_θ ) + √β_t · z,   z ~ N(0,I)
```

**DDIM 反向一步**（非马尔可夫：`t'` 可以是子序列里任意一个更早的点）：

```
# ① 由 x_t 与预测噪声反推干净图
pred_x0 = ( x_t - √(1-ᾱ_t) · ε_θ ) / √ᾱ_t

# ② η 决定这一步注入多少随机性
σ    = η · √( (1-ᾱ_t') / (1-ᾱ_t) ) · √( 1 - ᾱ_t / ᾱ_t' )

# ③ 重新"加噪"到 t'
x_t' = √ᾱ_t' · pred_x0 + √(1 - ᾱ_t' - σ²) · ε_θ + σ · z
```

`η = 0` ⇒ `σ = 0`，`z` 项消失：

```
x_t' = √ᾱ_t' · pred_x0 + √(1-ᾱ_t') · ε_θ        ← 完全确定性
```

最后一步 `t' = 0`（`ᾱ = 1`）时 `√(1-1)=0`，于是 `x = pred_x0`，即最终生成图。

### 三个关键差别

| | DDPM | DDIM (η=0) |
|---|---|---|
| 随机性 | 每步都抽 `z`，同一个 `x_T` 每次结果不同 | 不抽取噪声，`x_T` ↔ 生成图一一对应 |
| 步数 | 走满 `T` 步 | 只走子序列 τ，**可跳步** |
| 变量 | 局部的 `β_t`、`α_t` | 全局的 `ᾱ_t`、`ᾱ_t'` |

---

## 🎛️ 两个新参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `ddim_steps` | 50 | 采样步数。从 `[0, T-1]` 等间隔取这么多个点（`make_ddim_timesteps`）。**50 步 ≈ 比 DDPM 快 20 倍**；10 步也能出图但更糊 |
| `eta` | 0.0 | 随机性开关。`0` = 确定性 DDIM；`1` ≈ DDPM 的后验噪声量 |

> `T=1000` 仍在 `config.py` 里，但那是**训练时的总步数**（决定 schedule 长度）；
> DDIM 采样时用多少个点由 `ddim_steps` 单独控制，两者不要混淆。

---

## ⚙️ 安装

```bash
pip install torch torchvision matplotlib tqdm
```

> 💡 若报 `No module named 'torch'` / `torchvision`：说明当前终端的 `python`
> 不是装了 PyTorch 的那个解释器。双击本目录的 `run_sample_ddim.bat`，
> 写一个.bat文件用绝对路径运行：
```@echo off
cd /d "%~dp0"
 `...python\versions\3.13.12\python.exe sample_ddim.py`
pause
```
---

## 🧪 分步测试

```bash
python ddim.py          # ✅ 自检：确定性 / 跳步子序列 / 无 NaN（不需要 ckpt）
python model.py         # ✅ UNet 前向 / 反向自检（补回 MidBlock 后通过）
```

`ddim.py` 的自检会用一个临时小 UNet 验证：同一个 `x_T` 在 `η=0` 下两次采样
**逐像素完全相同**——这是 DDIM 最本质的性质。

---

## 🚀 运行

```bash
# 方式 1：直接复用 DDPM 训好的权重
python sample_ddim.py

# 方式 2：重新训练
python train.py
python sample_ddim.py
```

调参：

```bash
python sample_ddim.py --steps 10       # 极速 10 步
python sample_ddim.py --steps 100      # 100 步，通常更清晰
python sample_ddim.py --eta 1          # 加入随机性，接近 DDPM
python sample_ddim.py --interp         # 隐空间插值（DDIM 独有）
```

---

## 🔬 对比实验

同一个 `ckpt_last.pt`，跑两次：

```bash
# 先把权重放到本目录：sample.py 是零改动复制的，它没有自动找权重的逻辑，
# 而 sample_ddim.py 会自动去 ../ddpm_mnist/ 借，所以只有 DDPM 这边需要复制。
cp ../ddpm_mnist/ckpt_last.pt .

python sample.py          # DDPM：1000 步，出 samples.png
python sample_ddim.py     # DDIM： 50 步，出 samples_ddim.png
```

观察三件事：
1. **耗时**：DDIM 只调用网络 50 次 vs DDPM 1000 次；
2. **质量**：50 步 DDIM 通常已经能看出数字轮廓，100 步更稳；
3. **可复现性**：`python sample_ddim.py` 连跑两次，`samples_ddim.png` 应**完全相同**；
   而 `python sample.py` 连跑两次结果不同

---

## 📊 图文件

| 图文件 | 来源 | 它反映什么 |
|---|---|---|
| `samples_ddim.png` | `sample_ddim.py` | DDIM 最终 4×4 生成网格 —— 判断跳步采样有没有成功 |
| `sampling_traj_ddim.png` | `sample_ddim.py` | 50 步（而非 1000 步）的去噪演化 —— 每帧跨度很大，仍逐步成型 |
| `interpolation_ddim.png` | `--interp` | 两个 `x_T` 之间插值，数字平滑过渡 —— **确定性带来的语义隐空间** |
| `samples.png` | `sample.py`（DDPM） | 对照组：1000 步马尔可夫采样的结果 |
| `schedule.png` / `noising.png` / `loss_curve.png` | 同 DDPM | 见 DDPM 版 README |

---

## 🧩 与 DDIM 论文的出入

- 论文（Song, Meng & Ermon, *ICLR 2021*）还给了一个很有用的 **反演 / inversion**：
  把真实图片沿确定性路径编码回 `x_T`，从而实现图像编辑。本复现**只做了采样与插值**。
- 论文的 `η` 可取任意实数；我们只暴露 `0`（确定性）与 `1`（≈DDPM）两个典型值。
- 子序列 τ 用最简单的**等间隔 `linspace`**；论文也提到二次间隔等其它取法，
  在步数很少时会影响质量。
- 论文有连续时间 / 概率流 ODE 的视角（DDIM 是它的离散特例），本复现只写离散版。
- 沿用了 DDPM 版的所有简化：线性 schedule、无 EMA、简化 U-Net、方差固定为 `β_t`。

---

## ⭐ 排错清单

- **`FileNotFoundError: ckpt_last.pt`** → 先跑 `python train.py`，或把 DDPM 的权重复制过来。
- **生成全是灰噪** → 权重没训好；或 `--steps` 太小（试 50→100）。
- **输出 NaN** → 检查 `σ` 根号内；代码里已用 `max(..., 0)` 保护，正常不会触发。
- **`--steps` 设得比 `T` 大** → `make_ddim_timesteps` 会断言报错，把步数降到 `≤1000`。
- **形状报错** → 确认 `sample_ddim.py` 里 `n` 是完全平方数（默认 16 = 4×4）。

---

## 📚 参考

- Ho et al., *Denoising Diffusion Probabilistic Models* (2020) — DDPM
- Song, Meng & Ermon, *Denoising Diffusion Implicit Models* (ICLR 2021) — DDIM
