import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) 超参数
batch_size, lr, num_epochs = 64, 0.3, 10

# 2) 数据集（Fashion-MNIST，灰度 1×28×28）
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# 3) 定义网络（LeNet-5：Conv→AvgPool→Conv→AvgPool→FC）
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            # 输入: [B, 1, 28, 28]
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),     # -> [B, 6, 28, 28]
            nn.AvgPool2d(kernel_size=2, stride=2),                    # -> [B, 6, 14, 14]
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),                # -> [B, 16, 10, 10]
            nn.AvgPool2d(kernel_size=2, stride=2),                    # -> [B, 16, 5, 5]
            nn.Flatten(),                                             # -> [B, 16*5*5=400]
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Dropout(0.5),                                            # Dropout 50%
            nn.Linear(84, 10)                                             # 输出 10 类 logits（不加 softmax）
        )

        # Xavier 初始化（Conv/Linear）
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# 4) 创建模型与 CUDA
net = SimpleNN()
device = torch.device('cuda')

net.to(device)

# 5) 损失函数与优化器（交叉熵内部已包含 softmax）
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=lr)

# 6) 测试集准确率评估
@torch.no_grad()
def evaluate_accuracy():
    net.eval()
    correct, total = 0, 0
    for X, y in test_iter:
        X, y = X.to(device), y.to(device)
        logits = net(X)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    net.train()
    return correct / total

# 7) 训练
for epoch in range(1, num_epochs + 1):
    total_loss, total_num = 0.0, 0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        logits = net(X)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_num  += bs

    test_acc = evaluate_accuracy()
    print(f"epoch {epoch:d} | loss {total_loss/total_num:.6f} | test acc {test_acc:.3f}")
