# 🎯 DDIM on MNIST

> 在 [Diffusion_DDPM](../Diffusion_DDPM) 的基础上复现 **DDIM**（Denoising Diffusion Implicit Models）
> ⭐ DDIM 相比 DDPM 不改训练，只改采样

---

## 🔍 对于 DDPM 的改动 

| 文件 | 状态 | 说明 |
|---|---|---|
| `config.py` | 🟢 **相同** | 超参数 |
| `schedule.py` | 🟢 **相同** | 加噪参数表 |
| `dataset.py` | 🟢 **相同** | 载入数据 |
| `train.py` | 🟢 **相同**| 训练 |
| `diffusion.py` | 🟢 **相同** | DDPM 前向过程 |
| `sample.py` | 🟢 **相同**| DDPM 反向过程 |
| `model.py` | 🟡 **更新**  | U-net网络 |
| `ddim.py` | 🔵 **新增** | 采样、反演|
| `sample_ddim.py` | 🔵 **新增** | DDIM 调用窗口及相关测试 |
| `.gitignore` | 🟢 **相同** | — |
| `README.md` | 🟡 **更新** | — |

---

## 📂 目录结构

```
Diffusion_DDIM/
├── config.py           ⚙️ 超参数
├── schedule.py         📈 加噪过程的参数表
├── model.py            🧠 U-Net
├── dataset.py          🗂️ 数据集
├── train.py            🚀 训练
├── diffusion.py        🔀 正向加噪过程
├── sample.py           🖼️ DDPM 反向过程
├── ddim.py             ✨ DDIM 采样 / 反演 
├── sample_ddim.py      ✨ DDIM 调用窗口及相关测试
├── .gitignore          🚫
└── README.md
```

---

## 🧮 公式

**前向过程**：

```
x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε          →  loss = MSE(ε, ε_θ(x_t, t))
```

记 **t = 噪声更多的一步、s = 噪声更少的一步**（s < t，两者可以是 τ 里任意相邻两点）。

**(A) 由加噪图反推原始图**：

```
pred_x0 = ( x_t - √(1-ᾱ_t) · ε_θ(x_t, t) ) / √ᾱ_t
```

**(B) DDIM 采样：t → s**

```
σ    = |η| · √( (1-ᾱ_s) / (1-ᾱ_t) ) · √( 1 - ᾱ_t/ᾱ_s )
x_s  = √ᾱ_s · pred_x0 + √(1 - ᾱ_s - σ²) · ε_θ + σ · z
```

`η = 0` ⇒ `σ = 0`，`z` 项消失 ⇒ 完全确定性。
最后一步 `s = 0`（`ᾱ = 1`）时 `√(1-1)=0`，于是 `x = pred_x0`，即最终生成图。

**(C) DDIM 反演：s → t**

```
ε_s     = ε_θ(x_s, s)                         ← 时间步填 s，不是 t
pred_x0 = ( x_s - √(1-ᾱ_s) · ε_s ) / √ᾱ_s
σ       = |η| · √( (1-ᾱ_s) / (1-ᾱ_t) ) · √( 1 - ᾱ_t/ᾱ_s )   ← 与采样同一个 σ
x_t     = √ᾱ_t · pred_x0 + √(1 - ᾱ_t - σ²) · ε_s + σ · z
```

> ⭐反演时 `x_t` 还不存在，无法算 `ε_θ(x_t,t)`，
> 所以**用已知点 `x_s` 处的 `ε_θ(x_s, s)` 近似**，并用同一个 `ε_s`
> 同时做「预测 x₀」和「重新加噪」。这正是 η=0 时反演与采样互逆的原因。

**DDPM 反向一步**（对照组，马尔可夫链，`t` 与 `t-1` 必须相邻）：

```
x_{t-1} = (1/√α_t) · ( x_t - (β_t / √(1-ᾱ_t)) · ε_θ ) + √β_t · z,   z ~ N(0,I)
```

### 对比

| | DDPM | DDIM |
|---|---|---|
| 随机性 | 每步都抽 `z`，同一个 `x_T` 每次结果不同 | **不抽噪声**，`x_T` ↔ 生成图一一对应 |
| 步数 | 必须走满 `T` 步 | 只走子序列 τ，**可跳步** |
| 参数 | 局部的 `β_t`、`α_t` | 全局的 `ᾱ_t`、`ᾱ_s`（所以能跨步） |
| 其他 | — | 语义插值、**反演 / 图像编辑** |

---

## 🎛️ η 可取任意实数

`--eta` **可为任意实数**，
代码里用 `|η|` 计算标准差（标准差必须非负，且 `z~N(0,I)` 关于 0 对称，
所以 `σ·z` 与 `-σ·z` 同分布，负 η 与 `|η|` 统计等价）

| η | 行为 | 典型用途 |
|---|---|---|
| `0` | `σ=0`，完全确定性、可逆 | 复现、反演重建、语义插值 |
| `0 < η < 1` | 介于确定性 DDIM 与 DDPM 之间 | 给样本增加随机性 |
| `1` | `σ` ≈ DDPM 后验标准差 | 最接近 DDPM（但仍可跳步） |
| `η > 1` | `σ²` 超过目标步噪声预算，方向系数被 `max(...,0)` 截为 0，噪声主导 | 画质会变糊但**不会 NaN** |
| `η < 0` | 按 `|η|` 处理，与正数统计等价 | 验证符号不影响分布 |

> ⚠️ 判断「要不要抽 z」必须看 `|η| > 0`。旧版写成 `eta > 0`，
> 负 η 时噪声项被静默丢掉、退化成错误的确定性路径——本版已修，
> `python ddim.py` 里有专门的回归测试
> `T=1000` 仍在 `config.py`，但那是**训练时的总步数**；采样步数由`ddim_steps` 单独控制，两者不要混淆。

---

## 🔁 反演 inversion

反演把一张**真实图片**沿确定性路径倒推回隐变量 `x_T`：

```
x₀ --ddim_invert--> x_T --ddim_sample--> x₀'
```

**功能**

- **重建验证**：η=0 且模型训练充分时 `x₀' ≈ x₀`，证明 `x_T` 是一个可逆编码；
- **图像编辑**：在 `x_T` 上做一点修改再采样回来，得到「保留原图结构、局部改变」的结果；
- **真实图插值**：两张真实图各自反演成 `x_T`，`slerp` 插值后生成过渡图。

**调试运行**

```bash
python sample_ddim.py --inversion                 # 默认 50 步、eta=0
python sample_ddim.py --inversion --steps 100    # 步数更多，重建误差更小
python sample_ddim.py --inversion --eta 0.5      # 带随机性，重建相似但不同
```

会打印 `反演重建 MAE`，并保存 `inversion_ddim.png`（上排原图、下排重建）。

> ⚠️ **只有 η=0 时反演与采样才严格互逆**。η≠0 会在反演和采样里各注入
> 随机噪声，重建结果会「相似但不同」——这适合随机图像编辑，不适合验证可逆性。
> 标准 DDIM inversion 论文里只定义 η=0；本实现把 σ 项带进反演是**对称化的
> 实验性扩展**。


---

## ⚙️ 安装

```bash
pip install torch torchvision matplotlib tqdm
```

> 💡 若有报错 `No module named 'torch'` / `torchvision`，可写脚本用绝对路径运行：
```
@echo off
cd /d "%~dp0"
...\python\versions\3.13.12\python.exe(Python路径) sample_ddim.py
pause
```
---

## 🚀 运行

```bash
# 方式 1：直接复用 DDPM 训好的权重
python sample_ddim.py
#   自动按顺序找 ckpt_last.pt：本目录 → ../ddpm_mnist/ → ../Diffusion_DDPM/

# 方式 2：重新训练
python train.py
python sample_ddim.py
```

调试：

```bash
python sample_ddim.py --steps 10         
python sample_ddim.py --steps 100        
python sample_ddim.py --eta 0.5         
python sample_ddim.py --eta -1        
python sample_ddim.py --eta 2          
python sample_ddim.py --interp        
python sample_ddim.py --inversion     
```

---

## 🔬 对比实验

同一个 `ckpt_last.pt`下运行：

```bash
cp ../ddpm_mnist/ckpt_last.pt .

python sample.py          # DDPM：1000 步
python sample_ddim.py     # DDIM： 50 步
```

测试：
1. **时间**：DDIM 只调用网络 50 次 vs DDPM 1000 次；
2. **质量**：50 步 DDIM 通常已经能看出数字轮廓，100 步更稳；
3. **验证**：`python sample_ddim.py` 连跑两次，`samples_ddim.png` 应**完全相同**；
   而 `python sample.py` 连跑两次结果不同——这就是「确定性」最直观的证据。

---

## 📊 图文件

| 图文件 | 来源 | 它反映什么 |
|---|---|---|
| `samples_ddim.png` | `sample_ddim.py` | DDIM 最终 4×4 生成网格 —— 判断跳步采样有没有成功 |
| `sampling_traj_ddim.png` | `sample_ddim.py` | 50 步（而非 1000 步）的去噪演化 —— 每帧跨度很大，仍逐步成型 |
| `interpolation_ddim.png` | `--interp` | 两个 `x_T` 之间插值，数字平滑过渡 —— **确定性带来的语义隐空间** |
| `inversion_ddim.png` | `--inversion` | 上排真实图 / 下排反演重建图 —— **DDIM 可逆编码** |
| `samples.png` | `sample.py`| 对照组：1000 步马尔可夫采样的结果 |
| `schedule.png` / `noising.png` / `loss_curve.png` | 同 DDPM | 见 DDPM 版 README |

---

## 🧩 与 DDIM 论文的出入

- 论文（Song, Meng & Ermon, *ICLR 2021*）的 **inversion** 只定义 `η=0`；
  本实现把它推广到任意实数 η（对称地把 σ 项带进反演），η≠0 时**不再保证可逆**，
  属于实验性扩展，不是论文原样。
- 论文把 η 当作 `[0,1]` 的插值系数；本实现接受**任意实数**，负 η 按 `|η|` 处理、
  η>1 用 `max(...,0)` 保护方向系数（论文未讨论这些区间）。
- 子序列 τ 用最简单的**等间隔 `linspace`**；论文也提到二次间隔等其它取法，
  在步数很少时会影响质量。
- 论文有连续时间 / 概率流 ODE 的视角（DDIM 是它的离散特例），本复现只写离散版。
- 反演用的是「`ε_θ(x_s, s)` 近似 `ε_θ(x_t, t)`」的标准离散近似，
  步数少时误差会累积；论文/后续工作有更高阶的反演方案。
- 沿用了 DDPM 版的所有简化：无 EMA、简化 U-Net。

---

## ⭐ 排错清单

- **`FileNotFoundError: ckpt_last.pt`** → 先跑 `python train.py`，或把 DDPM 的权重复制过来。
- **生成全是灰噪** → 权重没训好；或 `--steps` 太小（试 50→100）。
- **输出 NaN** → 检查 `σ` 根号内；代码已用 `max(...,0)` 与 `1e-12` 分母保护，正常不会触发。
- **`--eta` 很大时图变糊** → 预期行为（σ² 超标、方向项被截断），不是 bug。
- **反演重建误差大** → 检查是否 `--eta 0`、ckpt 是否训够、`--steps` 是否足够。
- **`--steps` 设得比 `T` 大** → `make_ddim_timesteps` 会断言报错，把步数降到 `≤1000`。
- **形状报错** → 确认 `--n` 是完全平方数（默认 16 = 4×4）。

---

## 📚 参考

- Ho et al., *Denoising Diffusion Probabilistic Models* (2020) — DDPM
- Song, Meng & Ermon, *Denoising Diffusion Implicit Models* (ICLR 2021) — DDIM

---
## 其他
> ⚠️ 可以用 AST 比对两个文件相同(BeyondCompare)
> Python 的注释和 docstring 不参与执行：

```bash
cd Diffusion_DDIM
python - <<'PY'
import ast, pathlib
def logic(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):                       # 剥掉模块/类/函数的 docstring
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
               and isinstance(b[0].value.value, str):
                node.body = b[1:]
    return ast.dump(tree, include_attributes=False)
for f in ["config.py","schedule.py","diffusion.py","dataset.py","train.py","sample.py"]:
    a = logic(pathlib.Path("../Diffusion_DDPM", f))
    b = logic(f)
    print(("逻辑相同  " if a == b else "逻辑不同  ") + f)
PY
```
---
