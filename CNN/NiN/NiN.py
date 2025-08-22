import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1) 超参数
batch_size, lr, num_epochs = 128, 0.01, 10

# ---- 小工具：避免 DataLoader 多进程下 lambda 的 pickle 问题
def repeat_to_3_channels(x):
    return x.repeat(3, 1, 1)  # [1,H,W] -> [3,H,W]

# 2) 数据集（Fashion-MNIST → 3×227×227）
transform = transforms.Compose([
    transforms.Resize(227),
    transforms.ToTensor(),
    transforms.Lambda(repeat_to_3_channels),
    transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
])

train_dataset = datasets.FashionMNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True)
test_iter  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)

# 3) 定义网络（NiN：Conv + 1×1 Conv ×2 + GAP，无全连接层）

# ========== 更清晰的 NiN 块（固定带 BN） ==========
def nin_block(in_ch: int, out_ch: int, k: int, s: int, p: int):
    """
    Conv(k×k, bias=False) -> BN -> ReLU
      -> Conv(1×1) -> ReLU
      -> Conv(1×1) -> ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True),
        nn.ReLU(inplace=True),
    )

def nin_block_logits(in_ch: int, out_ch: int, k: int, s: int, p: int):
    """
    分类头版本：最后一层不加 ReLU，输出 logits
    Conv(k×k, bias=False) -> BN -> ReLU
      -> Conv(1×1) -> ReLU
      -> Conv(1×1)         (no ReLU)
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True),  # ← no ReLU
    )

# ========== 模型（与之前结构/尺寸保持一致） ==========
class NiN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # 227 -> Conv(11,s=4,p=0) -> 55 -> MaxPool(3,2) -> 27
            nin_block(3,    96, k=11, s=4, p=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.3),

            # 27 -> Conv(5,s=1,p=2) -> 27 -> MaxPool -> 13
            nin_block(96,  256, k=5,  s=1, p=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(p=0.3),

            # 13 -> Conv(3,s=1,p=1) -> 13 -> MaxPool -> 6
            nin_block(256, 384, k=3,  s=1, p=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 分类头：保持 6×6，通道变 num_classes → GAP → [B, C]
        self.classifier = nn.Sequential(
            nin_block_logits(384, num_classes, k=3, s=1, p=1),  # -> 10×6×6
            nn.AdaptiveAvgPool2d((1, 1)),                       # -> 10×1×1
            nn.Flatten(),                                       # -> [B, 10]
        )

        # 初始化（与 ReLU/BN 匹配）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 4) CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = NiN(num_classes=10).to(device)

# 5) 损失与优化器
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
# 可选：用 Adam 会更稳一些（CPU 上更友好）：
# opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

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

    test_acc = evaluate_accuracy()
    print(f"epoch {epoch:d} | loss {total_loss/total_num:.6f} | test acc {test_acc:.3f}")
