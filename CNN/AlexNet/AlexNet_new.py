import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) 超参数
batch_size, lr, num_epochs = 128, 0.01, 10

# 2) 数据集（Fashion-MNIST → 3×224×224）
#    灰度扩展为 3 通道；使用经验均值/方差做标准化
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),               # [1,H,W] → [3,H,W]
    transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
])

train_dataset = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True)
test_iter  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)

# 3) 定义网络（AlexNet：features + classifier）
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            # ------- features -------
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # AlexNet 原论文是 6×6，这里用自适应以避免尺寸偏差
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),                                # 256 * 6 * 6 = 9216

            # ------- classifier -------
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 10)                         # FashionMNIST/CIFAR-10 → 10 类
        )

        # 初始化（Conv/Linear 用 Kaiming 更适配 ReLU）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 此处用kaiming初始化Conv会出问题  nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# 4) CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = SimpleNN().to(device)

# 5) 损失函数与优化器（交叉熵内部已包含 softmax）
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
# 可选调度器：逐步衰减学习率
# scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[6, 8], gamma=0.1)

# 6) 测试集准确率评估
@torch.no_grad()
def evaluate_accuracy():
    net.eval()
    correct, total = 0, 0
    for X, y in test_iter:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = net(X)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    net.train()
    return correct / total

# 7) 训练
for epoch in range(1, num_epochs + 1):
    net.train()
    total_loss, total_num = 0.0, 0
    for X, y in train_iter:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = net(X)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_num  += bs

    # scheduler.step()  # 若使用调度器，取消注释

    test_acc = evaluate_accuracy()
    print(f"epoch {epoch:d} | loss {total_loss/total_num:.6f} | test acc {test_acc:.3f}")
