# myGPT 简要训练
# 不考虑性能与实用性, 以学习理论为目的

# 训练次数
epoches = 3


from model import myGPT, GPTConfig
import torch
import time
from datetime import datetime

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[33m'  # 黄色文本
RESET = '\033[0m'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {GREEN}{device}{RESET}")

def train_main():
    print(f"{GREEN}start trainning...{RESET}\n")
    
    model = myGPT().to(device)

    # 打印模型详情
    print(model)
    
    # 打印参数量
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Parameters: {param.numel()}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n\nTotal params: {total_params:,}\n\n")


    model.train()

    # 定义优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    do_train(model, optimizer)


    print(f"\n{RED}train finished...{RESET}")

def do_train(model, optimizer):
    
    # 模拟数据(随机生成, 假设 myGPT 输入是序列, 输出是预测下一个 token)
    seq_length = 16
    inputs = torch.randint(0, GPTConfig.vocab_size, (1, seq_length)).to(device)
    targets = torch.randint(0, GPTConfig.vocab_size, (1, seq_length)).view(-1).to(device)

    for epoch in range(epoches):
        print(f"start epoch {epoch} now...")
        epoch_start = time.time() 

        # 前向传播
        outputs = model(inputs, targets)

        # 假设损失函数
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        # 反向传播和优化（需要定义优化器）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_time = time.time() - epoch_start
        print(f"Training epoch {epoch} finished, consume: {epoch_time:.2f}s, Loss: {loss.item():.4f}\n")



train_main()

