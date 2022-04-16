import math
import torch
from torch import nn
from d2l import torch as d2l

#@save
def masked_softmax(X, valid_lens):
  """通过在最后一个轴上掩蔽元素来执行softmax操作"""
  # X:3D张量，valid_lens:1D或2D张量
  if valid_lens is None:
    return nn.functional.softmax(X, dim=-1)
  else:
    shape = X.shape
    if valid_lens.dim() == 1:
      valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
      valid_lens = valid_lens.reshape(-1)
    # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
    X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)

#@save
class AdditiveAttention(nn.Module):
  """加性注意力"""
  def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
    super(AdditiveAttention, self).__init__(**kwargs)
    self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
    self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
    self.w_v = nn.Linear(num_hiddens, 1, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, queries, keys, values, valid_lens):
    queries, keys = self.W_q(queries), self.W_k(keys)
    # 在维度扩展后，
    # queries的形状：(batch_size，查询的个数，1，num_hidden)
    # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
    # 使用广播方式进行求和
    x6 = queries.unsqueeze(2)
    x7 = keys.unsqueeze(1)
    features = x6 + x7
    features = torch.tanh(features)
    # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
    # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
    scores = self.w_v(features).squeeze(-1)
    self.attention_weights = masked_softmax(scores, valid_lens)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    x5 = torch.bmm(self.dropout(self.attention_weights), values)
    return x5

tran = torch.rand(2, 2, 4)
print(tran)
print(masked_softmax(tran, torch.tensor([2, 3])))
print('---')



queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values的小批量，两个值矩阵是相同的

x1 = torch.arange(40, dtype=torch.float32)
x2 = x1.reshape(1, 10, 4)
x3 = x2.repeat(
  2, 1, 1)
values = x3
valid_lens = torch.tensor([2, 6])
#
# attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
#                               dropout=0.1)
# attention.eval()
# attention(queries, keys, values, valid_lens)
#
# x4 = attention.attention_weights
# d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
#                   xlabel='Keys', ylabel='Queries')
# d2l.plt.show()

#@save
class DotProductAttention(nn.Module):
  """缩放点积注意力"""
  def __init__(self, dropout, **kwargs):
    super(DotProductAttention, self).__init__(**kwargs)
    self.dropout = nn.Dropout(dropout)

  # queries的形状：(batch_size，查询的个数，d)
  # keys的形状：(batch_size，“键－值”对的个数，d)
  # values的形状：(batch_size，“键－值”对的个数，值的维度)
  # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
  def forward(self, queries, keys, values, valid_lens=None):
    d = queries.shape[-1]
    # 设置transpose_b=True为了交换keys的最后两个维度
    x8 = keys.transpose(1,2)
    scores = torch.bmm(queries, x8) / math.sqrt(d)
    self.attention_weights = masked_softmax(scores, valid_lens)
    return torch.bmm(self.dropout(self.attention_weights), values)

queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)

d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
d2l.plt.show()