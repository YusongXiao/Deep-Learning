import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np, random

# 0) 稳定性设置（可复现）
seed = 42
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def repeat_to_3_channels(x):
    # [1,H,W] -> [3,H,W]
    return x.repeat(3, 1, 1)

# 1) 超参数
batch_size, lr, num_epochs = 128, 0.005, 10
clip_max_norm = 5.0

# 2) 数据集（Fashion-MNIST → 3×224×224）
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(repeat_to_3_channels),  # [1,H,W] → [3,H,W]
    transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
])
train_dataset = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
test_iter  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# 3) Inception 模块
class Inception(nn.Module):
    def __init__(self, in_c, c1, c3r, c3, c5r, c5, pool_p):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_c, c1, kernel_size=1), nn.ReLU()
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_c, c3r, kernel_size=1), nn.ReLU(),
            nn.Conv2d(c3r, c3, kernel_size=3, padding=1), nn.ReLU()
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_c, c5r, kernel_size=1), nn.ReLU(),
            nn.Conv2d(c5r, c5, kernel_size=5, padding=2), nn.ReLU()
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_c, pool_p, kernel_size=1), nn.ReLU()
        )

    def forward(self, x):
        y1 = self.b1(x); y2 = self.b2(x); y3 = self.b3(x); y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)

# 4) 辅助分类器
class AuxHead(nn.Module):
    def __init__(self, in_c, num_classes):
        super().__init__()
        self.aux = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(in_c, 128, kernel_size=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024), nn.ReLU(),
            nn.Dropout(p=0.5),                 # 原 0.7 → 0.5
            nn.Linear(1024, num_classes)
        )
    def forward(self, x):
        return self.aux(x)

# 5) GoogLeNet 主体
class SimpleNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNN, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),   # 224→112
            nn.MaxPool2d(3, stride=2, ceil_mode=True),                         # 112→56
            nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),          # 56→56
            nn.MaxPool2d(3, stride=2, ceil_mode=True)                          # 56→28
        )
        self.inc3a = Inception(192, 64, 96, 128, 16, 32, 32)     # 256
        self.inc3b = Inception(256, 128, 128, 192, 32, 96, 64)   # 480
        self.pool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)    # 28→14

        self.inc4a = Inception(480, 192, 96, 208, 16, 48, 64)    # 512
        self.aux1  = AuxHead(512, num_classes)
        self.inc4b = Inception(512, 160, 112, 224, 24, 64, 64)   # 512
        self.inc4c = Inception(512, 128, 128, 256, 24, 64, 64)   # 512
        self.inc4d = Inception(512, 112, 144, 288, 32, 64, 64)   # 528
        self.aux2  = AuxHead(528, num_classes)
        self.inc4e = Inception(528, 256, 160, 320, 32, 128, 128) # 832
        self.pool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)    # 14→7

        self.inc5a = Inception(832, 256, 160, 320, 32, 128, 128) # 832
        self.inc5b = Inception(832, 384, 192, 384, 48, 128, 128) # 1024

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),                                         # 1024
            nn.Dropout(p=0.4),
            nn.Linear(1024, num_classes)
        )

        # 初始化：Conv=Kaiming，Linear=Normal(0,0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.inc3a(x); x = self.inc3b(x)
        x = self.pool3(x)
        x = self.inc4a(x)
        aux1 = self.aux1(x) if self.training else None
        x = self.inc4b(x); x = self.inc4c(x)
        x = self.inc4d(x)
        aux2 = self.aux2(x) if self.training else None
        x = self.inc4e(x)
        x = self.pool4(x)
        x = self.inc5a(x); x = self.inc5b(x)
        logits = self.head(x)
        if self.training:
            return logits, aux1, aux2
        else:
            return logits

# 6) CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SimpleNN(num_classes=10).to(device)

# 7) 损失与优化器
ce = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# 8) 测试集准确率评估
@torch.no_grad()
def evaluate_accuracy():
    net.eval()
    correct, total = 0, 0
    for X, y in test_iter:
        X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        out = net(X)
        logits = out if isinstance(out, torch.Tensor) else out[0]
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    net.train()
    return correct / total


if __name__ == "__main__":
# 9) 训练
    for epoch in range(1, num_epochs + 1):
        net.train()
        total_loss, total_num = 0.0, 0
        for X, y in train_iter:
            X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)

            out = net(X)
            if isinstance(out, tuple):
                main, a1, a2 = out
                loss = ce(main, y) + 0.3 * (ce(a1, y) + ce(a2, y))
                logits = main
            else:
                loss = ce(out, y); logits = out

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_max_norm)
            opt.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_num  += bs

            # 可选：检测异常，便于定位问题
            # if not torch.isfinite(loss): print("Found non-finite loss, skipping batch."); continue

        test_acc = evaluate_accuracy()
        print(f"epoch {epoch:d} | loss {total_loss/total_num:.6f} | test acc {test_acc:.3f}")
