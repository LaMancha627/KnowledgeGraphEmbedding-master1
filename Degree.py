import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

print(os.getcwd())

with open('data/wn18/entities.dict') as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open('data/wn18/relations.dict') as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def get_degree(triples, degree=0):
    '''
    获取实体的度
    '''
    '''for head, relation, tail in triples:
        if head in triples:
            degree += 1
            print(head + "度为:   " + str(degree))
        degree = 0'''
    degree_dict = {}
    a = []
    for head, relation, tail in triples:
        # print(type(triples))
        for i in triples:
            '''#print(i[0])
            if head == i[0] or head == i[2]:
                degree += 1
        degree_dict[head]=degree
        print(str(head)+'degree:   '+str(degree))
        degree = 0'''
            a[i[0]] += 1
            a[i[2]] += 1
    sorted_dict = sorted(degree_dict.items(), key=lambda kv: kv[1])
    print(sorted_dict)
    return degree


def get_network(triples):
    G = nx.Graph()
    for t in triples:
        G.add_node(t[0])
        G.add_node(t[2])
        G.add_edge(t[0], t[2])  # 结点，结点
    # print("所有节点的度:", G.degree)  # 返回所有节点的度
    # print("所有节点的度分布序列:", nx.degree_histogram(G))  # 返回图中所有节点的度分布序列（从1至最大度的出现次数）
    # 无向图度分布曲线
    degree = nx.degree_histogram(G)
    remove = [node for node, degree in dict(G.degree).items() if degree < 30]
    G.remove_nodes_from(remove)
    print(type(remove))
    degrees = [(node, val) for (node, val) in G.degree]
    sorted_degree = sorted(degrees, key=lambda x: x[1], reverse=True)
    # print(type(sorted_degree)) list
    print("排序后：" + str(sorted_degree))
    # 选择前三分之一的元素
    index1 = int(len(sorted_degree) / 3)
    index2 = int(2 * len(sorted_degree) / 3)
    remain = sorted_degree[0:index1]
    # remain =sorted_degree[index1:index2]
    # remain =sorted_degree[index2:]
    # print("保留的节点为：  "+str(remain))
    # remain = [node for node, degree in dict(G.degree).items() if degree > 30]
    # print(remain)
    # print(remain[0])
    remain_triples = []
    i = 0
    # 获取保留节点对应的三元组
    # print("三元组"+str(triples[1][0]))
    for j in range(len(triples)):
        for i in range(len(remain)):
            if triples[j][0] == remain[i][0] or triples[j][0] == remain[i][0]:
                # print(triples[j])
                remain_triples.append(triples[j])
    # print("剩下的三元组有:" + str(remain_triples))
    # print("所有节点的度:", G.degree)  # 返回所有节点的度
    '''  x = range(len(degree))  # 生成X轴序列，从1到最大度
    y = [z / float(sum(degree)) for z in degree]  # 将频次转化为频率，利用列表内涵
    plt.loglog(x, y, color="blue", linewidth=2)  # 在双对坐标轴上绘制度分布曲线
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.title('数据集节点度分布图')
    plt.xlabel("degree")
    plt.ylabel("frequency")
    plt.show()  # 显示图表'''

    return remain_triples


'''def counter(triple):
    return Counter(triple)'''


def get_path(triples):
    G = nx.Graph()
    for t in triples:
        G.add_node(t[0])
        G.add_node(t[2])
        G.add_edge(t[0], t[2])  # 结点，结点
    # 定义node2vec算法参数
    p = 1
    q = 1
    dimensions = 8
    num_walks = 10
    walk_length = 80

    # 使用node2vec算法生成随机游走序列
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q)
    walks = node2vec.walks

    '''# 输出随机游走序列
    for walk in walks:
        print(walk)'''

    return walks


import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import numpy as np


class PathLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(PathLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, vocab_size)

    def forward(self, path):
        embeds = self.embedding(path)
        lstm_out, _ = self.lstm(embeds.view(len(path), 1, -1))
        tag_space = self.hidden2label(lstm_out.view(len(path), -1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores


# 假设我们已经从node2vec中得到了随机游走序列，将其转化为数字索引表示
triple = read_triple('data/wn18/train.txt', entity2id, relation2id)
triples = get_network(triple)
walks = get_path(triples)
vocab = list(set([node for walk in walks for node in walk]))
word_to_ix = {word: i for i, word in enumerate(vocab)}

# 将随机游走序列转化为数字索引序列
indexed_walks = []
for walk in walks:
    indexed_walks.append([word_to_ix[node] for node in walk])
# print(indexed_walks)
padded_sequences = []
for path in indexed_walks:
    path1 = torch.tensor(path)
    padded_sequences.append(path1)
max_len = max([len(seq) for seq in padded_sequences])
padded_sequences1 = pad_sequence([torch.LongTensor(seq) for seq in padded_sequences], batch_first=True, padding_value=0)
# 训练模型
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
model = PathLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(vocab))
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for path in padded_sequences1:
        '''indices = torch.tensor(path)
        print(indices)
        print(type(indices))'''
        model.zero_grad()
        tag_scores = model(path)
        loss = loss_function(tag_scores, torch.tensor(path[1:]))
        loss.backward()
        optimizer.step()

# 得到每个节点的路径嵌入表示
node_embeddings = np.zeros((len(vocab), HIDDEN_DIM))
for i in range(len(vocab)):
    node_embeddings[i] = model.embedding(torch.tensor(i)).detach().numpy()
print("嵌入表示为： " + str(node_embeddings))

'''
triple = read_triple('data/wn18/train.txt', entity2id, relation2id)
triples = get_network(triple)
get_path(triples)
'''
# get_network(triple)
# print(triple[0:5])
# print(type(triple))
# 列表中的元组的元素统计次数并排序
# new_triple = [str(list(j)) for j in triple]
# print(type(new_triple))
# result = (counter(new_triple))
# result = Counter(new_triple)
# print(result)
# sort = sorted(result.items(), key=lambda x: x[1], reverse=True)
# print(sort)
