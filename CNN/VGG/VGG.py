import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) 超参数
batch_size, lr, num_epochs = 64, 0.01, 10

# ===== 修复点 #1：把 lambda 换成可 pickle 的顶层函数 =====
def repeat_to_3_channels(x):
    # [1,H,W] -> [3,H,W]
    return x.repeat(3, 1, 1)

# 2) 数据集（Fashion-MNIST → 3×224×224）
#    灰度扩 3 通道；使用 FMNIST 的经验均值/方差做标准化
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(repeat_to_3_channels),  # ✅ 替代 lambda
    transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
])

train_dataset = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# —— 保持不变 —— #
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
test_iter  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# 3) 定义网络（VGG19：2-2-4-4-4 个 3×3 Conv 块 + 5 次 2×2 池化）
class SimpleNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            # block1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 224→112

            # block2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 112→56

            # block3 (×4)
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 56→28

            # block4 (×4)
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 28→14

            # block5 (×4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 14→7

            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),                                          # 512*7*7 = 25088

            # classifier
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

        # —— 初始化：Conv 用 Kaiming，Linear 用 Normal(0,0.01) —— #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# 4) CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SimpleNN(num_classes=10).to(device)

# 5) 损失函数与优化器
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
# 可选调度：更稳
# scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[6, 8], gamma=0.1)

# 6) 测试集准确率评估（保持不变）
@torch.no_grad()
def evaluate_accuracy():
    net.eval()
    correct, total = 0, 0
    for X, y in test_iter:
        X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = net(X)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    net.train()
    return correct / total

# ===== 修复点 #2：把“启动子进程的地方”放进 main 保护 =====
if __name__ == "__main__":
    # 7) 训练（结构不变）
    for epoch in range(1, num_epochs + 1):
        net.train()
        total_loss, total_num = 0.0, 0
        for X, y in train_iter:
            X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)

            logits = net(X)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_num  += bs

        # if 'scheduler' in globals(): scheduler.step()
        test_acc = evaluate_accuracy()
        print(f"epoch {epoch:d} | loss {total_loss/total_num:.6f} | test acc {test_acc:.3f}")
