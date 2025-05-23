# myGPT implementation for study

代码基本 copy/modify from karpathy(head of AI at Tesla) [Karpathy implementation of GPT](https://github.com/karpathy/nanoGPT). 本仓库主要是用来记录之前复制/修改的代码.

## Test Run
```bash
git clone xxx.git
cd xxx

python ./train.py # 执行train.py就会load所有model流程, 按需加日志即可
```


## model.py(网络架构)

核心架构实现解释参考: [transformer介绍](https://github.com/leafan/notes/blob/main/ai/llm/transformer.md)

最终的 myGPT 实现时, 将 token特征矩阵与位置矩阵相加作为input输入给model最后达成输出:

```python
class myGPT(nn.Module):
    ...

    def forward(self, x, targets=None):
        b, t    = x.size() # batch_size, sequence_len
        pos     = torch.arange(0, t, dtype=torch.long) # sequence长度的位置向量

        # 将x乘上(带上)token特征向量, shape: (b, t, n_embd)
        tok_emb = self.transformer.wte(x)
        # 将pos乘上(带上)位置特征向量, shape: (t, n_embd)
        pos_emb = self.transformer.wpe(pos)

        # 在PyTorch中, 当两个张量的维度不完全一致时, 会自动进行维度扩展使它们能够相加
        # 因为 tok_emb比pos_emb 维度高, 所以相加后效果等价于每一个 batch上都加上 pos_emb
        x = tok_emb + pos_emb

        # 执行 n_layer 遍 transformer 逻辑
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x[:, [-1], :])
        return logits
```

## train.py(训练逻辑)

训练逻辑写的很粗糙, 没有去找标准的vocab以及对应数据去训练, 而是随机了一个向量用来测试程序流畅性
实际上这里还是有很多改进学习点的, 记为todo..

**TODO**:
- 使用官方vocab与dataset来训练数据
- 支持输出模型性能(正确率)
- 学习率等超参调配
- 微调逻辑适配
- ...

## predict.py(预测逻辑)

预测逻辑非常简单, 其实就是 train 函数不传target参数即可.