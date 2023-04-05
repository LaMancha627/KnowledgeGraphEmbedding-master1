import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from node2vec import Node2Vec
import numpy as np


class PathDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.tensor(self.paths[idx], dtype=torch.long)


class LSTMPathEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPathEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化LSTM的隐藏状态和记忆单元
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # 前向传播
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = F.normalize(out, p=2, dim=1) # 对输出进行L2正则化，使其在球面上分布
        return out


if __name__ == '__main__':
    # 生成node2vec随机游走序列
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    walks = [list(map(str, walk)) for walk in model.walks]

    # 将随机游走序列转换为PyTorch数据集
    dataset = PathDataset(walks)

    # 使用PyTorch的DataLoader加载数据集
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    # 定义模型并优化器
    model = LSTMPathEmbedding(64, 128, 2, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, paths in enumerate(dataloader):
            optimizer.zero_grad()
            embeddings = model(paths)
            loss = -torch.mean(torch.sum(embeddings ** 2, dim=1))
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # 获取嵌入表示
    embeddings = model(torch.tensor(list(map(str, G.nodes)), dtype=torch.long))
    embeddings = embeddings.detach().numpy()

    # 打印嵌入表示的统计信息
    print('Embedding shape:', embeddings.shape)
    print('Embedding mean:', np.mean(embeddings))
    print('Embedding std:', np.std(embeddings))
