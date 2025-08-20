import torch
import torch.nn as nn

batch_size, lr, num_epochs = 10, 0.03, 12

true_w = torch.tensor([ 2.0, -3.4])
true_b = torch.tensor(4.2)
n = 1000

# 生成数据集
features = torch.randn(n, 2)
labels = features @ true_w + true_b + 0.01 * torch.randn(n)

# 数据加载器
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(features, labels),
    batch_size=batch_size,
    shuffle=True    #随机打乱
)

# 定义一个简单的线性神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 1)   # 输入 2 个特征，输出 1 个预测值
        )

    def forward(self, x):
        return self.net(x)

# 创建模型实例
net = SimpleNN()

# 损失函数
loss_fn = nn.MSELoss()

# 创建优化器
opt = torch.optim.SGD(net.parameters(), lr=lr)

# 训练
for epoch in range(1, num_epochs + 1):
    for X, y in loader:
        pred = net(X).squeeze(-1)          # [B,1] → [B]
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 训练出来的网络用feature向前传播得到所有预测值，用labels进行评估
    with torch.no_grad():
        full_pred = net(features).squeeze(-1)
        full_loss = loss_fn(full_pred, labels)
    print(f"epoch {epoch:d} | loss {full_loss:.6f}")

# 6) 查看参数误差
w = net.net[0].weight.detach().reshape(-1)
b = net.net[0].bias.detach().reshape(())
print("w误差：", true_w - w)
print("b误差：", true_b - b)
