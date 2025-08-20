import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) 超参数
batch_size, lr, num_epochs = 256, 0.1, 10

# 2) 数据集（Fashion-MNIST）
transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3) 定义网络（Softmax 回归 = Flatten + Linear）
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),            # 28×28 → 784
            nn.Linear(28 * 28, 256), # 隐藏层，256个神经元
            nn.ReLU(),               # 激活函数
            nn.Linear(256, 10)       # 输出层，10个神经元
        )

    def forward(self, x):
        return self.net(x)

# 4) 创建模型与权重初始化
net = SimpleNN()

# 判断当前模块 m 是否是线性层（全连接层），如果是线性层，才会进行后续的权重初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)   # 权重服从均值为0，标准差为0.01的正态分布
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # 线性层如果有偏置 m.bias，则将偏置初始化为全0张量

net.apply(init_weights)

# 5) 使用 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# 6) 损失函数与优化器
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=lr)

# 7) 测试集准确率评估
@torch.no_grad()
def evaluate_accuracy():
    net.eval()     # 将模型切换到评估模式
    correct, total = 0, 0
    for X, y in test_iter:
        X, y = X.to(device), y.to(device)
        logits = net(X)     # 将输入数据 X 传入模型 net，得到模型的输出（通常称为logits）
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    net.train()    # 将模型切换回训练模式
    return correct / total

# 8) 训练
for epoch in range(1, num_epochs + 1):
    total_loss, total_num = 0.0, 0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        logits = net(X)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        bs = y.size(0)      # y.size(0) 是本批次实际的样本数量，与超参数batch_size可能不同
        total_loss += loss.item() * bs
        total_num  += bs

    test_acc = evaluate_accuracy()
    print(f"epoch {epoch:d} | loss {total_loss/total_num:.6f} | test acc {test_acc:.3f}")