"""
================================================================================
 模块：train.py —— 训练循环（DDPM 与 DDIM 完全共用的部分）
================================================================================

【每个 batch 】
    1) x0    = 一个 batch 的原图             （来自 DataLoader）
    2) t     = 随机步数 (B,)                （每个样本独立随机）
    3) noise = 随机噪声 ε ~ N(0,I)
    4) xt    = q_sample(x0, t, noise)        （闭式一步加噪）
    5) pred  = model(xt, t)                  （预测噪声 ε_θ）
    6) loss  = MSE(pred, noise)              （和真噪声比）
    7) zero_grad → backward → step          （PyTorch 自动求导 + 更新）

【为什么 t 要随机，而不是从 0 走到 T】
    闭式解 x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε 让我们能直接跳到任意 t，
    所以一个 batch 里每个样本可以抽不同的 t。这样一次 forward 就能
    覆盖所有噪声强度，训练效率高得多。这也是 DDPM 论文的核心技巧。

================================================================================
"""
import torch                                  
from torch.optim import Adam                  # 优化器使用 Adam
from tqdm import tqdm                         # 进度条
import matplotlib
matplotlib.use("Agg")                         # 用非交互后端，无弹窗，只往文件里写
import matplotlib.pyplot as plt

from config import T, LR, EPOCHS, CKPT, DEVICE, SEED
from dataset import get_mnist_dataloader       
from model import UNet, test_model             
from diffusion import q_sample                 
from schedule import make_schedule             


def save_loss_curve(losses, save_path="loss_curve.png"):
    """
    画图：loss 曲线
    """
    plt.figure(figsize=(7, 4))
    plt.plot(losses)                      
    plt.xlabel("step")
    plt.ylabel("MSE loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()    


def train():
    torch.manual_seed(SEED)                   # 种子
    sched = make_schedule()                   # 常数表
    loader = get_mnist_dataloader()           # MNIST
    model = UNet().to(DEVICE)                 # 建网络并搬到CPU/GPU
    opt = Adam(model.parameters(), lr=LR)     # 优化器接管 model 的全部可学习参数

    losses = []                              # 收集每个 step 的 loss，供画曲线
    step = 0                                 # 全局步数计数（跨 epoch 累加）
    for epoch in range(EPOCHS):
        # tqdm 只包一层进度条；loader 每次吐 (图片, 标签)。
        # 标签 y 用不上 → 用 _ 接住：这是「无条件生成」，不按数字类别条件去噪。
        for x0, _ in tqdm(loader, desc=f"epoch {epoch+1}/{EPOCHS}"):  # 进度条
            x0 = x0.to(DEVICE)               #  从训练集中采样
            B = x0.shape[0]                  #  取当前批大小；最后一个 batch 可能不足 BATCH_SIZE，所以必须动态读，不能写死 BATCH_SIZE
            t = torch.randint(0, T, (B,), device=DEVICE)
                                             #  随机取一个时刻用于训练。含 0 与论文 Uniform{1..T} 略有不同
            noise = torch.randn_like(x0)     #  从标准正态中采样一个噪声，噪声的 tensor 大小和输入x0大小相同
            xt = q_sample(x0, t, noise, sched) # 加噪
            pred = model(xt, t)              #  网络预测噪声
            loss = torch.nn.functional.mse_loss(pred, noise) # MSE loss

            # PyTorch 训练：
            opt.zero_grad()                  #  清空上一轮的梯度（调用 .backward 时是默认累加的）
            loss.backward()                  #  自动求导，把梯度填进每个参数的 .grad
            opt.step()                       #  按 Adam 规则更新参数

            losses.append(loss.item())       # .item() 取具体数值，保存在losses列表中
            step += 1
            if step % 200 == 0:              # 每 200 步打印一次
                print(f"  step {step}  loss {loss.item():.4f}")

        # 一个 epoch 结束：保存权重 + 更新 loss 曲线。
        # 只存 state_dict（参数字典）而不是整个 model，加载时更灵活、也更小。
        torch.save(model.state_dict(), CKPT)
        save_loss_curve(losses)
        print(f"[train] epoch {epoch+1} 完成，已保存 {CKPT} 与 loss_curve.png")

    return model


if __name__ == "__main__":
    test_model() 
    train()
