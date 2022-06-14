import os
import collections
import random
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from string import punctuation
from torch import nn
import math


def txt_prepocessing(txt_path):
    # 文本预处理
    test_path = txt_path
    txt_file = os.listdir(test_path)
    txt_file.sort()
    text = ''

    for i in txt_file:
        if i.endswith('.txt'):
            with open(os.path.join(test_path, i), 'r') as f:
                txt_f = f.readlines()
            if txt_f[-1].replace('\n', '').replace('-', '').replace('\n', '一').isnumeric():
                text += ''.join(txt_f[:-1])
            else:
                text += ''.join(txt_f)

    return text.replace('\n', '')


def tokenize(lines, token='word'):
    # 将文本行拆分为单词或字符词元
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误:未知词元类型:' + token)


def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens).most_common()

class Vocab:
    '''获得词字典'''
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        self._token_freqs = count_corpus(tokens)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

        # 已经排序self._token_freqs
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)
        return [self.__getitem__(token) for token in tokens]

    def __len__(self):
        return len(self.idx_to_token)


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # ⻓度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的⻓度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

class SeqDataLoader:
    def __init__(self, corpus, batch_size, num_steps):
        self.data_iter_fn = seq_data_iter_random
        self.vocab = Vocab(corpus)
        self.corpus = [self.vocab[char] for line in corpus for char in line]
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_corpus(path, batch_size=32, num_steps=35):
    text = ''
    for i in path:
        txt = txt_prepocessing(i)
        text += txt
        if text[-1] not in punctuation:
            text += '。'
    chars = tokenize(text, 'char')
    data_iter = SeqDataLoader(corpus=chars, batch_size=batch_size, num_steps=num_steps)
    return data_iter, data_iter.vocab


class RNNModel(nn.Module):
    '''循环神经网络'''
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的(之后将介绍)，num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


def predict(prefix, num_preds, net, vocab, device=torch.device('cuda')):
    def get_word_order_tensor(sign, device):
        return torch.LongTensor([sign]).reshape(1,1).to(device)

    state = net.begin_state(batch_size=1, device=device)

    #   state = torch.randn((net.num_directions * net.rnn.num_layers,
    #                              1, net.num_hiddens),
    #                             device=device)
    output = []

    for y in prefix[:-1]:
        word_num = get_word_order_tensor(vocab[y], device)
        _, state = net(word_num, state)
        output.append(int(word_num))
    word_num = get_word_order_tensor(vocab[prefix[-1]], device)
    output.append(int(word_num))

    num_pred = 0
    while num_pred<num_preds:
        word_num = get_word_order_tensor(output[-1], device)
        y, state = net(word_num, state)
        output.append(int(torch.argmax(y, axis=1).reshape(1)))
        if output[-1] == vocab['。']:
            num_pred += 1

    return ''.join([vocab.idx_to_token[i] for i in output]).replace('。', '。\n')


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_rnn(net, epochs, train_iter, loss, optim,
              state=None, grad_clip_theta=1, device=torch.device('cuda')):
    # 梯度裁剪
    loss = loss
    opt = optim
    epochs = epochs
    perplexity = []

    net.to(device=device)
    for epoch in range(epochs):
        epoch_loss = []
        for X, Y in train_iter:
            y = Y.T.reshape(-1)
            X, y = X.to(device), y.to(device)
            if state:
                state = net.begin_state(batch_size=X.shape[0], device=device)
            y_hat, state = net(X, state)
            # state 是叶子节点，不可以直接设置grad
            state = (state[0].detach(),)
            l = loss(y_hat, y.long()).mean()
            epoch_loss.append(l.item())

            opt.zero_grad()
            l.backward()
            grad_clipping(net, grad_clip_theta)
            opt.step()
        epoch_perplexity = np.exp(np.mean(epoch_loss))
        if (epoch+1)%50==0:
            print(f'epoch: {epoch+1}, perplexity:{epoch_perplexity:f}')
        perplexity.append(epoch_perplexity)

    plt.plot(perplexity, label='train', color='b', linestyle='solid')
    plt.xlabel('epoch')
    plt.ylabel('perplexity')
    plt.legend()
    plt.show()


# additive attention
class AdditiveAttention(nn.Module):
  def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
    super(AdditiveAttention, self).__init__(**kwargs)
    self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
    self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
    self.w_v = nn.Linear(num_hiddens, 1, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, queries, keys, values, valid_lens):
    queries, keys = self.W_q(queries), self.W_k(keys)
    features = queries.unsqueeze(2) + keys.unsqueeze(1)
    features = torch.tanh(features)
    scores = self.w_v(features).squeeze(-1)
    self.attention_weights = masked_softmax(scores, valid_lens)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    return torch.bmm(self.dropout(self.attention_weights), values)


def sequence_mask(X, valid_len, value=0):
  """在序列中屏蔽不相关的项"""
  maxlen = X.size(1)
  mask = torch.arange((maxlen), dtype=torch.float32,
          device=X.device)[None, :] < valid_len[:, None]
  X[~mask] = value
  return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
  """带遮蔽的softmax交叉熵损失函数"""
  # pred的形状：(batch_size,num_steps,vocab_size)
  # label的形状：(batch_size,num_steps)
  # valid_len的形状：(batch_size,)
  def forward(self, pred, label, valid_len):
    weights = torch.ones_like(label)
    weights = sequence_mask(weights, valid_len)
    self.reduction='none'
    unweighted_loss = super().forward(
                pred.permute(0, 2, 1), label)
    weighted_loss = (unweighted_loss * weights).mean(dim=1)
    return weighted_loss

def masked_softmax(X, valid_lens):
  """通过在最后⼀个轴上掩蔽元素来执⾏softmax操作"""
  # X:3D张量，valid_lens:1D或2D张量
  if valid_lens is None:
    return nn.functional.softmax(X, dim=-1)
  else:
    shape = X.shape
  if valid_lens.dim() == 1:
    valid_lens = torch.repeat_interleave(valid_lens, shape[1])
  else:
    valid_lens = valid_lens.reshape(-1)
  # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0
  X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
          value=-1e6)
  return nn.functional.softmax(X.reshape(shape), dim=-1)

class Seq2SeqEncoder(nn.Module):
  def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
               dropout=0, **kwargs):
    super().__init__(**kwargs)
    # embedding
    self.embedding = nn.Embedding(vocab_size, embed_size)
    self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

  def forward(self, X, *args):
    # 输出'X'的形状：(batch_size,num_steps,embed_size)
    X = self.embedding(X)
    X = X.permute(1, 0, 2)
    output, state = self.rnn(X)
    return output, state

class EncoderDecoder(nn.Module):
  """编码器-解码器架构的基类"""
  def __init__(self, encoder, decoder, **kwargs):
    super().__init__(**kwargs)
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, enc_X, dec_X, *args):
    enc_outputs = self.encoder(enc_X, *args)
    dec_state = self.decoder.init_state(enc_outputs, *args)
    return self.decoder(dec_X, dec_state)

def train_seq2seq(net, epochs, train_iter, loss, optim, tgt_vocab,
  grad_clip_theta=1, device=torch.device('cuda')):
  net.to(device)
  net.train()

  loss_fig = []
  for epoch in range(epochs):
    epoch_loss = []
    for batch in train_iter:
      optim.zero_grad()
      X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
      bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
      dec_input = torch.cat([bos, Y[:, :-1]], 1) # 强制教学
      Y_hat, _ = net(X, dec_input, X_valid_len)
      l = loss(Y_hat, Y, Y_valid_len)
      epoch_loss.append(l.mean().item())
      l.sum().backward()
      grad_clipping(net, grad_clip_theta)
      optim.step()
    epoch_perplexity = np.mean(epoch_loss)
    if (epoch+1)%50==0:
        print(f'epoch: {epoch+1}, loss:{epoch_perplexity:f}')
    loss_fig.append(epoch_perplexity)

  plt.plot(loss_fig, label='train', color='b', linestyle='solid')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  plt.show()

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device=torch.device('cuda'), save_attention_weights=False):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
                            src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor(
                [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使⽤具有预测最⾼可能性的词元，作为解码器在下⼀时间步的输⼊
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # ⼀旦序列结束词元被预测，输出序列的⽣成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k): #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def truncate_pad(line, num_steps, padding_token):
  """截断或填充⽂本序列"""
  if len(line) > num_steps:
    return line[:num_steps] # 截断
  return line + [padding_token] * (num_steps - len(line)) # 填充