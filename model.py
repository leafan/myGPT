# my implementation for a minimum GPT version
# The code was manually rewritten line-by-line (not copy-pasted...)
# from: https://github.com/karpathy/nanoGPT

# code 不考虑性能, 更多是可读性的方向编码

import torch
import torch.nn as nn
from torch.nn import functional as F

import math


# 定义全局config
#  我们定义一个mini版本的参数
class GPTConfig:
    # 命名为block_size而不是 sequence_length, 因为这表示最大长度, sequence表示可变长序列
    block_size: int = 128   # 序列长度, 也就是一次能够处理的输入(逻辑上还包括输出)的token数
    vocab_size: int = 50304 # GPT2的字典, 核心是英文, 这里不缩小, 直接使用英文data训练

    n_layer:    int = 4     # transformer 层数, 每多一层就多一个transformer block
    n_head:     int = 4     # attention头数目, 需要能被 n_embd 整除
    n_embd:     int = 256   # 特征向量维度, GPT-2默认是768, 越大能够表示的特征能力就越强

    dropout:    float = 0.2 # 我们训练样本小, 容易过拟合, 可以启用 dropout
    bias:       bool = True # 是否使用偏置, 默认启用

# 定义基础模块: 层归一化
# 核心实现: 通过调整每层神经元的输出分布，使其均值为0、方差为1，从而稳定训练过程
# 本类其实核心是使用 layer_norm 函数进行forward, 包装成类是未来 方便理解与调用
class LayerNorm(nn.Module):

    # ndim: 维需归一化的特征维度(如嵌入向量的长度)
    # bias: Bool类型, 确认是否使用偏置变量(如启用则多一维参数, 增强近似效果)
    def __init__(self, ndim, bias=True):
        super().__init__()

        # nn.Parameter 表示该参数可进行梯度训练
        self.weight = nn.Parameter(torch.ones(ndim))
        
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim)) 

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


# FFN(Feed Forward Network)层, 也叫 MLP(Multi-Layer Perceptron)层
# 逻辑是先升维再降维, 增加网络参数容量; 引入非线性, 在自注意力层之后形成互补, 增强模型表达能力
class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 全连接层实现升维
        self.c_fc   = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)

        # 通过将输入x与标准正态分布的累积分布函数Φ(x)相乘(GELU(x)=x⋅Φ(x) ), 引入非线性, 增强模型表达能力
        # 模拟神经元激活的随机性, , 保留的概率越高, 反之则可能被抑制
        # 常用一个初等函数的近似形式: x = 0.5x * (1 + tanh( 2/π(x + 0.044715x^3))) 来计算
        self.gelu   = nn.GELU()

        # 降维成input维度
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)

        # 执行dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 每一步的作用以及维度变化请参考该类的 init函数 定义
        x   = self.c_fc(x)
        x   = self.gelu(x)
        x   = self.c_proj(x)
        x   = self.dropout(x)

        return x


# SelfAttention 模块
# Casual的含义则是 语言输出时不能提前知道后边的token, 因此使用掩码来覆盖, 因此成为causal
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 参见config说明, 多头是为了增加多样性, 需与特征向量长度匹配
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # 注意力矩阵, 分为 k, q, v, 3个合并写到一起
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)

        # 全连接输出层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # attn 自注意力矩阵 随机dropout
        self.attn_dropout  = nn.Dropout(config.dropout)

        # proj输出层(最后的全连接) 随机dropout
        self.proj_drop      = nn.Dropout(config.dropout)

        # register_buffer 将掩码注册为模型的非可训练参数​(不参与梯度更新), 但会随模型保存/加载
        # 命名为 bias 是PyTorch社区的惯例(虽与偏置无关), 因其在注意力得分中通过加法生效(bias也是加法)
        self.register_buffer(
            "bias",
            # tril(low)用来生成block_size*block_size的下三角矩阵(左下数据保留, 右上为0)
            # 确保第 t 个位置只能关注位置 <= t 的输入, 防止未来信息泄露
            torch.tril(torch.ones(config.block_size, config.block_size))
                # 转成适合的维度, 适配多头注意力计算的 (Batch, num_head, T, T) 得分矩阵
                # 1类似于一个占位符, 可实际匹配其维度值
                .view(1, 1, config.block_size, config.block_size)
        )


    def forward(self, x):
        # 输入为 x, 维度是: batch size(批次大小), sequence length(序列长度), embedding dim(特征向量)
        B, T, C = x.size()

        # 对 输入张量x 与 特征权重qkv 计算, 完成后的shape为 [ batch_size, seq_len, 3 * n_embd ]
        qkv = self.c_attn(x)
        
        # split函数作用: 对矩阵做切分, 沿dim维度每 n_embd(第一个参数) 个元素分割一次
        # dim: inter类型(如0, 1, 2等), 表示分割沿张量的 x+1 维度进行, 决定分割操作的作用方向
        # 所以 q, k, v 维度均为 [batch_size, seq_len, n_embd]
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # view函数用于改变张量的维度, 这里表示将q按照 n_head(多头数量, 用于增加特征记忆能力) 拆分
        # view后 q.shape=(B, T, n_head, head_dim), 其中, C = n_head * head_dim
        q = q.view(B, T, self.n_head, C//self.n_head)

        # transpose: 转置函数. 表示将arg0和arg1的维度交换
        # 这里相当于把 q 由(B, T, n_head, head_dim) 变成(B, n_head, T, head_dim) 
        # 这么处理的目的: 方便 各个头独立计算, 最后再汇总
        q = q.transpose(1, 2)

        # k与v类似计算, 这里合并简化
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)


        # 计算自注意力 Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) ) V

        att = q @ k.transpose(-2, -1)              # 计算Q和键K的点积 Q*K^T, 输出形状为(B,nh,T,T)
        att  = att * (1.0/math.sqrt(k.size(-1)))    # 乘以缩放因子, 防止点积值过大导致梯度不稳定

        # 屏蔽无效(未来不应该可知)注意力
        # [:, :, :T, :T] 表示切片操作, 动态截取当前序列长度T的部分, 形状变为(1, 1, T, T)
        # == 0​：生成布尔掩码. 也就是说bias值为0的位置需要被掩码屏蔽, 被赋值为 -inf
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)    # 对 -inf 的位置 softmax 后变为0
        att = self.attn_dropout(att)    # 执行dropout操作
        
        # 执行最后的 att = (Q*K^T) * V
        # 维度变化: (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v

        # 执行转置, 也就是将多头与sequence调换回来, 维度: (B, T, nh, hs)
        # contiguous: 当执行 transpose 后, shape变了, 但内存中的物理存储顺序并未改变
        # view() 方法要求输入张量必须是内存连续的, 因此调用 .contiguous() 会重新分配内存保证连续
        y = y.transpose(1, 2).contiguous()

        # 合并最后2个维度, 回到输入维度(B, T, C)
        y = y.view(B, T, C)

        y = self.c_proj(y)      # 执行全连接层
        y = self.proj_drop(y)   # 执行drop

        return y


# 定义一个 transform 网络架构
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 具体定义见其definition
        self.ln_1   = LayerNorm(config.n_embd, bias=config.bias)
        self.attn   = CausalSelfAttention(config)
        self.ln_2   = LayerNorm(config.n_embd, bias=config.bias)
        self.ffn    = FFN(config)

    def forward(self, x):
        y = self.ln_1(x)    # 先对输入特征进行归一化处理
        y = self.attn(y)    # 计算自注意力
        y = self.ln_2(x+y)  # 对应 transformer图中的第一个 Add&Norm

        y = y + self.ffn(y) # 对应 transformer图中的 FFN和第二个 Add&Norm

        return y


# GPT class 定义
class myGPT(nn.Module):
    def __init__(self):
        super().__init__()

        config = GPTConfig
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # 定义 transformer 结构
        self.transformer = nn.ModuleDict(dict(
            # token特征向量, vocab里面每个单词有 n_embd 个
            wte = nn.Embedding(config.vocab_size, config.n_embd),

            # 位置矩阵, 相当于每个位置有  n_embd 个特征
            wpe = nn.Embedding(config.block_size, config.n_embd),

            drop = nn.Dropout(config.dropout),

            # 定义 n_layer 个transformer模块
            h = nn.ModuleList([
                        TransformerBlock(config) for _ in range(config.n_layer)
            ]),

            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 输出头(language model head), 将输出对应到 vocab_size 每个token的概率
        # 共享权重, OpenAI的GPT-2通过权重共享(wte与lm_head绑定)实现了参数高效性, 这一设计被广泛引用
        self.lm_head.weight = nn.Parameter(self.transformer.wte.weight)

        # 初始化weights, 通过apply递归地对模型及其所有子模块应用一个指定的函数 _init_weights
        self.apply(self._init_weights)

        # 这段代码实现了GPT-2论文中提出的残差投影层特殊缩放初始化方法, 应用缩放后的正态分布初始化
        # 目的: 初始化标准差按 0.02 / sqrt(2*layer) 比例缩小, 缓解梯度累积带来的数值不稳定
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer))

        # 打印参数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M\n")


    # targets: 用于确认是否为训练模式, 如果是, 返回loss
    # 只有顶层module需要targets, 因为最后计算loss
    def forward(self, x, targets=None):
        b, t    = x.size() # batch_size, sequence_len
        pos     = torch.arange(0, t, dtype=torch.long) # sequence长度的位置向量

        # 将x乘上(带上)token特征向量, shape: (b, t, n_embd)
        tok_emb = self.transformer.wte(x)
        # 将pos乘上(带上)位置特征向量, shape: (t, n_embd)
        pos_emb = self.transformer.wpe(pos)

        # 在PyTorch中, 当两个张量的维度不完全一致时, 会自动进行维度扩展使它们能够相加
        # 因为 tok_emb比pos_emb 维度高, 所以相加后效果等价于每一个 batch上都加上 pos_emb

        # 这里nanoGPT是根据config.dropout来决定的, 但我认为不应该dropout
        # 因为使用者是针对 输入的 tok_emb与pos_emb 相加后再执行dropout
        # 但这个向量我觉得很重要, 不应该dropout, 所以赋值为0
        # x = self.transformer.drop(tok_emb + pos_emb)
        x = tok_emb + pos_emb

        # 执行 n_layer 遍 transformer 逻辑
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # 训练模式: 计算所有位置的 logits(x等价于 x[:, :, :]), 用于计算交叉熵损失
            # 展平为[batch*seq_len, vocab_size]
            logits = self.lm_head(x)
            logits = logits.view(-1, logits.size(-1))
        else:
            # 推理模式: 只需最后一个位置的 logits(如 x[:, [-1], :]), 以自回归方式生成文本
            logits = self.lm_head(x[:, [-1], :])

        return logits
    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
