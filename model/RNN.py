import os
import collections
import random
import torch
from string import punctuation
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


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
    pun = []
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


def train_rnn(net, epochs, train_iter, loss, optim,
              state=None, grad_clip_theta=1, device=torch.device('cuda')):
    # 梯度裁剪
    def grad_clipping(net, theta=grad_clip_theta):
        if isinstance(net, nn.Module):
            params = [p for p in net.parameters() if p.requires_grad]
        else:
            params = net.params
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm

    loss = loss
    opt = optim
    state = state
    epochs = epochs
    perplexity = []

    net.to(device=device)
    for epoch in range(epochs):
        epoch_loss = []
        for X, Y in train_iter:
            y = Y.T.reshape(-1)
            X, y = X.to(device), y.to(device)
            state = net.begin_state(batch_size=X.shape[0], device=device)
            y_hat, state = net(X, state)
            # state 是叶子节点，不可以直接设置grad
            state = (state[0].detach(),)
            l = loss(y_hat, y.long()).mean()
            epoch_loss.append(l.item())

            opt.zero_grad()
            l.backward()
            grad_clipping(net)
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
