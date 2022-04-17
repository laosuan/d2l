import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

#@save
class PositionWiseFFN(nn.Module):
  """基于位置的前馈网络"""
  def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
      **kwargs):
    super(PositionWiseFFN, self).__init__(**kwargs)
    self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
    self.relu = nn.ReLU()
    self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

  def forward(self, X):
    x1 = self.dense1(X)
    x2 = self.relu(x1)
    x3 = self.dense2(x2)
    return x3



#@save
class AddNorm(nn.Module):
  """Residual connection followed by layer normalization."""
  def __init__(self, normalized_shape, dropout, **kwargs):
    super(AddNorm, self).__init__(**kwargs)
    self.dropout = nn.Dropout(dropout)
    self.ln = nn.LayerNorm(normalized_shape)

  def forward(self, X, Y):
    return self.ln(self.dropout(Y) + X)

#@save
class EncoderBlock(nn.Module):
  """Transformer encoder block."""
  def __init__(self, key_size, query_size, value_size, num_hiddens,
      norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
      dropout, use_bias=False, **kwargs):
    super(EncoderBlock, self).__init__(**kwargs)
    self.attention = d2l.MultiHeadAttention(
      key_size, query_size, value_size, num_hiddens, num_heads, dropout,
      use_bias)
    self.addnorm1 = AddNorm(norm_shape, dropout)
    self.ffn = PositionWiseFFN(
      ffn_num_input, ffn_num_hiddens, num_hiddens)
    self.addnorm2 = AddNorm(norm_shape, dropout)

  def forward(self, X, valid_lens):
    Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
    return self.addnorm2(Y, self.ffn(Y))

#@save
class EncoderBlock(nn.Module):
  """Transformer encoder block."""
  def __init__(self, key_size, query_size, value_size, num_hiddens,
      norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
      dropout, use_bias=False, **kwargs):
    super(EncoderBlock, self).__init__(**kwargs)
    self.attention = d2l.MultiHeadAttention(
      key_size, query_size, value_size, num_hiddens, num_heads, dropout,
      use_bias)
    self.addnorm1 = AddNorm(norm_shape, dropout)
    self.ffn = PositionWiseFFN(
      ffn_num_input, ffn_num_hiddens, num_hiddens)
    self.addnorm2 = AddNorm(norm_shape, dropout)

  def forward(self, X, valid_lens):
    Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
    return self.addnorm2(Y, self.ffn(Y))

#@save
class TransformerEncoder(d2l.Encoder):
  """Transformer encoder."""
  def __init__(self, vocab_size, key_size, query_size, value_size,
      num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
      num_heads, num_layers, dropout, use_bias=False, **kwargs):
    super(TransformerEncoder, self).__init__(**kwargs)
    self.num_hiddens = num_hiddens
    self.embedding = nn.Embedding(vocab_size, num_hiddens)
    self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
    self.blks = nn.Sequential()
    for i in range(num_layers):
      self.blks.add_module("block"+str(i),
                           EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                        norm_shape, ffn_num_input, ffn_num_hiddens,
                                        num_heads, dropout, use_bias))

  def forward(self, X, valid_lens, *args):
    # Since positional encoding values are between -1 and 1, the embedding
    # values are multiplied by the square root of the embedding dimension
    # to rescale before they are summed up
    X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
    self.attention_weights = [None] * len(self.blks)
    for i, blk in enumerate(self.blks):
      X = blk(X, valid_lens)
      self.attention_weights[
        i] = blk.attention.attention.attention_weights
    return X

class DecoderBlock(nn.Module):
  # The `i`-th block in the decoder
  def __init__(self, key_size, query_size, value_size, num_hiddens,
      norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
      dropout, i, **kwargs):
    super(DecoderBlock, self).__init__(**kwargs)
    self.i = i
    self.attention1 = d2l.MultiHeadAttention(
      key_size, query_size, value_size, num_hiddens, num_heads, dropout)
    self.addnorm1 = AddNorm(norm_shape, dropout)
    self.attention2 = d2l.MultiHeadAttention(
      key_size, query_size, value_size, num_hiddens, num_heads, dropout)
    self.addnorm2 = AddNorm(norm_shape, dropout)
    self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                               num_hiddens)
    self.addnorm3 = AddNorm(norm_shape, dropout)

  def forward(self, X, state):
    enc_outputs, enc_valid_lens = state[0], state[1]
    # During training, all the tokens of any output sequence are processed
    # at the same time, so `state[2][self.i]` is `None` as initialized.
    # When decoding any output sequence token by token during prediction,
    # `state[2][self.i]` contains representations of the decoded output at
    # the `i`-th block up to the current time step
    if state[2][self.i] is None:
      key_values = X
    else:
      key_values = torch.cat((state[2][self.i], X), axis=1)
    state[2][self.i] = key_values
    if self.training:
      batch_size, num_steps, _ = X.shape
      # Shape of `dec_valid_lens`: (`batch_size`, `num_steps`), where
      # every row is [1, 2, ..., `num_steps`]
      dec_valid_lens = torch.arange(
        1, num_steps + 1, device=X.device).repeat(batch_size, 1)
    else:
      dec_valid_lens = None

    # Self-attention
    X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
    Y = self.addnorm1(X, X2)
    # Encoder-decoder attention. Shape of `enc_outputs`:
    # (`batch_size`, `num_steps`, `num_hiddens`)
    Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
    Z = self.addnorm2(Y, Y2)
    return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(d2l.AttentionDecoder):
  def __init__(self, vocab_size, key_size, query_size, value_size,
      num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
      num_heads, num_layers, dropout, **kwargs):
    super(TransformerDecoder, self).__init__(**kwargs)
    self.num_hiddens = num_hiddens
    self.num_layers = num_layers
    self.embedding = nn.Embedding(vocab_size, num_hiddens)
    self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
    self.blks = nn.Sequential()
    for i in range(num_layers):
      self.blks.add_module("block"+str(i),
                           DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                        norm_shape, ffn_num_input, ffn_num_hiddens,
                                        num_heads, dropout, i))
    self.dense = nn.Linear(num_hiddens, vocab_size)

  def init_state(self, enc_outputs, enc_valid_lens, *args):
    return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

  def forward(self, X, state):
    X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
    self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
    for i, blk in enumerate(self.blks):
      X, state = blk(X, state)
      # Decoder self-attention weights
      self._attention_weights[0][
        i] = blk.attention1.attention.attention_weights
      # Encoder-decoder attention weights
      self._attention_weights[1][
        i] = blk.attention2.attention.attention_weights
    return self.dense(X), state

  @property
  def attention_weights(self):
    return self._attention_weights

num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
  len(src_vocab), key_size, query_size, value_size, num_hiddens,
  norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
  num_layers, dropout)
decoder = TransformerDecoder(
  len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
  norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
  num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
  translation, dec_attention_weight_seq = d2l.predict_seq2seq(
    net, eng, src_vocab, tgt_vocab, num_steps, device, True)
  print(f'{eng} => {translation}, ',
        f'bleu {d2l.bleu(translation, fra, k=2):.3f}')