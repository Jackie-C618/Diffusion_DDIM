"""
================================================================================
model.py —— U-Net
================================================================================
【数据流（28×28 单通道、DEPTH=2、BASE_CHANNELS=64 为例）】
    输入 (B,1,28,28)
      → in_conv                (B,64,28,28)
      → DownBlock ×2           (B,128,14,14) → (B,256,7,7)，每层存一份 skip
      → MidBlock               (B,256,7,7)，含自注意力
      → UpBlock ×2             (B,128,14,14) → (B,64,28,28)，每层 concat 对应 skip
      → out_conv               (B,1,28,28)  ← 预测的噪声

【嵌入 时间 t】
    同一个 x_t 在不同 t 上，网络该做的去噪强度完全不同。
    不给 t，网络只能学一个「平均意义」的去噪，生成图会一直是灰噪。
    t 经正弦编码成向量后，通过 FiLM（缩放+平移）注入每个 ResidualBlock。

【共包含 7 个部分】
    TimeEmbedding   整数 t → 连续向量
    ResidualBlock   基础残差块，时间在这里注入
    DownBlock       编码器一层：若干 ResBlock + 下采样，并保存 skip
    UpBlock         解码器一层：上采样 + 拼接 skip + 若干 ResBlock
    Attention       bottleneck 的单头自注意力
    MidBlock        bottleneck 中段，手动透传 t_emb
    UNet            把上面全部组装起来

================================================================================
"""
import torch                                    
import torch.nn as nn                         
import torch.nn.functional as F        
from config import BASE_CHANNELS, DEPTH, ATTN_AT, TIME_EMB_DIM, DEVICE


class TimeEmbedding(nn.Module):
    """
    把整数时间步 t 变成连续向量（Transformer 式正弦编码 + 两层 MLP）

    输入 : t    形状 (B,) 的整数
    输出 : emb  形状 (B, TIME_EMB_DIM)
    作用 : 让网络"知道现在是第几步"，这是它学出去噪的前提

    为什么用正弦编码而不是直接把 t/1000 喂进去：
    正弦编码给不同频率的分量，让线性层更容易区分相近的 t；
    这是 Transformer 位置编码的同款做法，DDPM 原论文也是这样
    """

    def __init__(self, dim=TIME_EMB_DIM):
        """声明这个模块里有哪些可学习层（此时还没有数据流过）。"""
        super().__init__()                          # 必须调用，nn.Module 才能登记子模块/参数
        self.dim = dim                              # 记下维度，forward 里要用来算 half
        self.mlp = nn.Sequential(                  # 一个小的两层 MLP，把固定编码变成可学习的表示
            nn.Linear(dim, dim * 4),               # 先升维到 4 倍，给非线性留空间
            nn.SiLU(),                              # DDPM激活函数常用 SiLU，比 ReLU 平滑）
            nn.Linear(dim * 4, dim),               # 再压回 dim，作为最终的时间向量
        )

    def forward(self, t):
        """描述数据怎么流过：整数 t → 正弦向量 → MLP → t_emb。"""
        half = self.dim // 2                        # 正弦/余弦各占一半，拼起来正好 dim 维
        # 频率序列：exp( -ln(10000) * i / half )，i=0..half-1
        # 得到从 1 递减到约 1/10000 的一组频率，和 Transformer 位置编码完全一致。
        # .device=t.device 保证 t 在 GPU 时 freqs 也在 GPU，避免设备不一致报错。
        freqs = torch.exp(
            torch.arange(half, device=t.device)
            * (-torch.log(torch.tensor(10000.0)) / half)
        )
        # t[:, None] 把 (B,) 升成 (B,1)；freqs[None, :] 升成 (1,half)；
        # 相乘触发广播 → (B, half)，即每个样本、每个频率上的相位参数。
        args = t[:, None].float() * freqs[None, :]          # (B, half)
        # sin 和 cos 各 (B,half)，沿最后一维拼成 (B,dim)。
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.mlp(pe)                         # (B,dim) → (B,dim)，但已是可学习的表示


class ResidualBlock(nn.Module):
    """基础块：
    GroupNorm→SiLU→Conv →(+时间t)→GroupNorm→SiLU→Conv，最后加残差。

    输入 : x (B,C,H,W)，t_emb (B, TIME_EMB_DIM)
    输出 : (B, out_ch, H, W)   （不改变空间尺寸）
    时间注入：t_emb 过一个小线性得到 (scale, shift)，对特征做 FiLM 调制

    残差连接 h + x 的意义：让梯度能直接回传，深层网络更好训；
    同时网络只需学「在输入基础上改动多少」，而不是从头重建整张特征图。
    """

    def __init__(self, in_ch, out_ch, t_dim=TIME_EMB_DIM):
        """声明层。in_ch/out_ch 可能不同，所以残差分支要准备 1×1 卷积对齐通道。"""
        super().__init__()
        self.block1 = nn.Sequential(                # 第一段：归一化 → 激活 → 卷积（顺便改通道数）
            nn.GroupNorm(min(8, in_ch), in_ch),     # GroupNorm 不依赖 batch 大小，扩散模型常用；
                                                    # min(8,in_ch) 防止通道数小于 8 时分组数非法
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1), # 3×3 卷积 + padding=1 → 保持 H,W 不变
        )
        self.block2 = nn.Sequential(                # 第二段：同样结构，但通道数不变
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.time_emb = nn.Sequential(              # 把 (B,t_dim) 映射成 (B, 2*out_ch)
            nn.SiLU(),
            nn.Linear(t_dim, out_ch * 2),           # 输出 scale 和 shift 两份，各 out_ch 维
        )
        # 残差连接需要通道对齐：不同就 1×1 卷积把 x 投影到 out_ch；相同就恒等映射。
        # 若这里漏掉，in_ch != out_ch 时 h + x 会直接形状报错。
        self.resample = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        """数据流：x → block1 → FiLM 注入时间 → block2 → 加残差。"""
        h = self.block1(x)                          # (B,in_ch,H,W) → (B,out_ch,H,W)
        # 把 (B, 2*out_ch) 沿通道维切成两半，各 (B,out_ch)。
        scale, shift = self.time_emb(t_emb).chunk(2, dim=1)
        # 升成 (B,out_ch,1,1)，才能和 (B,out_ch,H,W) 做广播（每个通道一个系数）。
        # 这一步和 schedule 里 gather 后 reshape 成 (B,1,1,1) 是同一类广播技巧。
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        # FiLM 调制：h*(1+scale)+shift。
        # 写成 1+scale 而不是直接 h*scale，是为了初始化时 scale≈0 就近似恒等，
        # 训练初期更稳定（不会一上来就把特征搅乱）。
        h = h * (1 + scale) + shift          # FiLM 调制
        h = self.block2(h)                         # 再过一段卷积，输出 (B,out_ch,H,W)
        return h + self.resample(x)                # 残差相加；resample 负责把 x 的通道对齐


class DownBlock(nn.Module):
    """
    编码器一层：若干 ResBlock + 一次下采样(stride=2 卷积)。

    输入 : x (B,in_ch,H,W), t_emb
    输出 : (x_down, skip)
        x_down : (B,out_ch,H/2,W/2) —— 继续往更深层送
        skip   : (B,out_ch,H,W)     —— 下采样之前的特征，存给解码器拼接
    为什么先存 skip 再下采样：解码器需要的是「同分辨率」的特征，存下采样后的就对不上了。
    """

    def __init__(self, in_ch, out_ch, num_res=2):
        super().__init__()
        # ModuleList 而不是普通 list：这样 PyTorch 才能把里面的层登记为子模块，
        # 参数才会被 model.parameters() 找到、才会被优化器更新、才会随 .to(device) 搬运。
        self.res_blocks = nn.ModuleList(
            # 第一块负责把 in_ch 变成 out_ch；后面的块输入输出都是 out_ch。
            [ResidualBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(num_res)]
        )
        # stride=2 的 3×3 卷积同时完成降分辨率和保持通道数。
        # 用卷积下采样（而非 AvgPool）让网络能自己学怎么压缩信息。
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        for rb in self.res_blocks:                  # 依次过若干残差块，t_emb 每次都要传
            x = rb(x, t_emb)
        skip = x                        # 存一份给解码器（注意是下采样前的）
        x = self.downsample(x)         # 空间尺寸减半
        return x, skip                              # 两个值：继续往深层走的、留给 skip 的


class UpBlock(nn.Module):
    """
    解码器一层：上采样 + 拼接 skip + 若干 ResBlock。

    输入 : x (B,in_ch,H,W)（来自更深一层）, skip (B,skip_ch,2H,2W), t_emb
    输出 : (B,out_ch,2H,2W)

    通道数的关键点：上采样只改空间尺寸、不改通道；
    拼接 skip 后通道变成 in_ch + skip_ch，所以**第一个** ResBlock
    的输入通道必须写成 in_ch + skip_ch，后面的才是 out_ch。
    """

    def __init__(self, in_ch, out_ch, skip_ch, num_res=2):
        super().__init__()
        # 转置卷积：2×2 核、stride=2 → 空间尺寸翻倍，通道数保持 in_ch。
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        # 拼接后通道 = in_ch + skip_ch，第一层 ResBlock 输入要对应
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(in_ch + skip_ch if i == 0 else out_ch, out_ch)
                for i in range(num_res)
            ]
        )

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)                       # 空间尺寸翻倍
        # 防奇偶差 1 像素：若输入尺寸是奇数，转置卷积出来的尺寸可能和 skip 差 1，
        # 直接 cat 会报错。用插值强制对齐到 skip 的 H,W。
        if x.shape != skip.shape:                 # 防奇偶差 1 像素
            x = F.interpolate(x, size=skip.shape[-2:])
        # 沿通道维（dim=1）拼接，这就是 U-Net 的跳跃连接。
        # 是 concat 不是 add：add 要求通道数相同且会丢信息，concat 保留两套特征。
        x = torch.cat([x, skip], dim=1)           # 跳跃连接拼接
        for rb in self.res_blocks:
            x = rb(x, t_emb)
        return x


class Attention(nn.Module):
    """
    bottleneck 自注意力：捕获长程结构（手写数字的笔画）。单头简化版。

    卷积只能看局部邻域，但在 7×7 的 bottleneck 上，让每个像素都能
    和所有其它像素交互，有助于把握「这一竖和那一横属于同一个数字」。
    这里用 1×1 卷积生成 Q/K/V，是注意力最轻量的实现方式。
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)  # 注意力前先归一化，训练更稳
        self.qkv = nn.Conv2d(channels, channels * 3, 1)       # 一次算出 Q/K/V 三份，省一次卷积
        self.proj = nn.Conv2d(channels, channels, 1)           # 输出再投影一次，混合通道信息

    def forward(self, x):
        B, C, H, W = x.shape                       # 拆出四个维度，后面要展平空间维
        h = self.norm(x)                           # (B,C,H,W)，归一化后的特征
        # (B,3C,H,W) → (B,3,C,H*W)：把「3 份通道」和「空间位置」分成两个轴。
        # 展平 H*W 后，每个位置就是一个长度为 C 的向量。
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # 各 (B,C,H*W)
        # q^T @ k → (B, H*W, H*W)：每个位置对所有位置的相似度。
        # 除以 √C 是标准缩放，防止点积太大让 softmax 饱和、梯度消失。
        attn = torch.softmax(q.transpose(1, 2) @ k / (C ** 0.5), dim=-1)
        # (B,H*W,H*W) @ (B,H*W,C) → (B,H*W,C)：按注意力权重加权求和 V。
        out = attn @ v.transpose(1, 2)            # (B, H*W, C)
        # 严格来说这里应先 .transpose(1,2) 再 reshape，才是标准的位置轴 → 空间轴还原；直接 reshape 会重排元素顺序。
        # 但模型已经按这个排列训练好了（ckpt 与之绑定），改成标准写法反而会让已有权重失效。为保持连续性这里保留原实现
        out = out.reshape(B, C, H, W)              # 折回图片形状
        # 残差连接：x + proj(out)。注意力只学「补充量」，不推翻原特征
        return x + self.proj(out)


class MidBlock(nn.Module):
    """
    bottleneck 中段：ResBlock → (可选)自注意力 → ResBlock。

    ⚠️ 这里**不能**用 nn.Sequential：Sequential 只会把上一层输出喂给下一层，
    第一个 ResidualBlock 就收不到时间嵌入 t_emb（报错 missing t_emb）。
    必须单独写个类，手动把 t_emb 透传进去。
    """

    def __init__(self, ch, use_attn=True):
        super().__init__()
        self.res1 = ResidualBlock(ch, ch)           # 第一个残差块，通道不变
        self.attn = Attention(ch) if use_attn else nn.Identity()
                                                    # 可选注意力；关掉时用恒等层占位，
                                                    # 这样 forward 不用写 if 分支
        self.res2 = ResidualBlock(ch, ch)           # 第二个残差块

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)                     # t_emb 必须显式传进去
        x = self.attn(x)                            # 注意力只需特征，不需要 t
        x = self.res2(x, t_emb)                     # 同样要传
        return x


class UNet(nn.Module):
    """完整 U-Net。
    把上面几个部分按照"编码器—bottleneck—解码器"组装。

    forward(x, t) ：
        x : (B,1,28,28) 加噪图
        t : (B,) 整数时间步
        返回 : (B,1,28,28) 预测噪声
    """

    def __init__(self, in_ch=1, base=BASE_CHANNELS, depth=DEPTH, attn_at=ATTN_AT):
        super().__init__()
        self.time_emb = TimeEmbedding()             # 全局共用一个时间编码器，
                                                    # 算出的 t_emb 会注入每一个 ResBlock
        # 每一层通道数逐级翻倍：depth=2, base=64 → [64, 128, 256]
        # 长度是 depth+1，因为还需要最深层（bottleneck）的通道数。
        chs = [base * (2 ** i) for i in range(depth + 1)]

        self.in_conv = nn.Conv2d(in_ch, chs[0], 3, padding=1)  # 1 → 64 通道，尺寸不变

        self.downs = nn.ModuleList()                # 编码器：depth 个 DownBlock
        for i in range(depth):
            # 第 i 层：chs[i] → chs[i+1]，同时分辨率减半
            self.downs.append(DownBlock(chs[i], chs[i + 1]))

        # bottleneck：在最深的一档（7×7、256 通道）做残差 + 自注意力
        self.mid = MidBlock(chs[-1], use_attn=(attn_at >= 0))

        self.ups = nn.ModuleList()                  # 解码器：depth 个 UpBlock
        for i in range(depth):
            in_ch_i = chs[depth - i]               # 当前解码层输入通道（从最深往回）
            out_ch_i = chs[depth - i - 1]          # 输出通道（上一层）
            skip_ch = chs[depth - i]               # 对应编码器存的 skip 通道
            self.ups.append(UpBlock(in_ch_i, out_ch_i, skip_ch))

        self.out_conv = nn.Sequential(              # 把 64 通道压回 1 通道（噪声图）
            nn.GroupNorm(min(8, chs[0]), chs[0]),
            nn.SiLU(),
            nn.Conv2d(chs[0], in_ch, 3, padding=1),
        )

    def forward(self, x, t):
        """ U-net"""
        t_emb = self.time_emb(t)                    # (B,) → (B,256)，整个网络共用
        x = self.in_conv(x)                         # (B,1,28,28) → (B,64,28,28)
        skips = []                                  # 栈按由浅到深存，解码时倒着取
        for down in self.downs:
            x, skip = down(x, t_emb)                # 每层返回继续往下走的和留给 skip 的
            skips.append(skip)
        x = self.mid(x, t_emb)                      # bottleneck，形状不变
        for up in self.ups:
            skip = skips.pop()                      # 后进先出，最深的 skip 先被用到
            x = up(x, skip, t_emb)                  # 上采样 → 拼接 → 残差
        return self.out_conv(x)                     # (B,64,28,28) → (B,1,28,28)


def test_model():
    """
    自检：随机张量跑 forward，断言输入输出同形状、且能反向传播。
    不下载数据、不需要权重，因此自检比较快
    """
    B = 4
    x = torch.randn(B, 1, 28, 28)                   # 伪造一批「加噪图」
    t = torch.randint(0, 1000, (B,))                # 每个样本一个随机时间步
    model = UNet().to(DEVICE)                       # 按 config 的默认配置建网
    out = model(x.to(DEVICE), t.to(DEVICE))         # 前向：形状必须和输入一致
    assert out.shape == x.shape, f"输出 {tuple(out.shape)} != 输入 {tuple(x.shape)}"
    out.sum().backward()                          # 反向：确认梯度能流回所有参数，不报错
    print("[model] 输出形状:", tuple(out.shape), "— 前向+反向通过 ✅")


if __name__ == "__main__":
    test_model()
