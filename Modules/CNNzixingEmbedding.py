from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import opencc
from fastNLP import Vocabulary, logger
from fastNLP.embeddings import TokenEmbedding, StaticEmbedding
from fastNLP.embeddings.utils import get_embeddings

from Utils.paths import radical_path,ftradical_path,cj_path,wb_path


f_j=opencc.OpenCC('t2s')
#仓颉
cj_char_info = dict()
with open(cj_path,'r',encoding='utf-8') as cj_f:
    cj_lines = cj_f.readlines()
    for line in cj_lines:
        cj_char,cj_info=line.split('\t',1)
        cj_char=f_j.convert(cj_char)
        cj_char_info[cj_char]=cj_info.replace('\n', '').split('\t')
#现代五笔
wb_char_info = dict()
with open(wb_path,'r',encoding='utf-8') as wb_f:
    wb_lines = wb_f.readlines()
    for line in wb_lines:
        wb_char=line[0]
        wb_info=line[1:]
        wb_char_info[wb_char]=wb_info.replace('\n', '').split('\t')

def char2cj(c):
    if c in cj_char_info.keys():
        c_info = cj_char_info[c]
        # print('\t'.join([c] + c_info).strip(), file=new_file)
        return list(c_info[0])        #原：return list(c_info[3])
    return ['○']

def char2wb(c):
    if c in wb_char_info.keys():
        c_info = wb_char_info[c]
        # print('\t'.join([c] + c_info).strip(), file=new_file)
        return list(c_info[0])        #原：return list(c_info[3])
    return ['○']

def _construct_cj_vocab_from_vocab(char_vocab: Vocabulary, min_freq: int = 1, include_word_start_end=True):
    r"""
    给定一个char的vocabulary生成character的vocabulary.

    :param vocab: 从vocab
    :param min_freq:
    :param include_word_start_end: 是否需要包含特殊的<bow>和<eos>
    :return:
    """
    cj_vocab = Vocabulary(min_freq=min_freq)
    for char, index in char_vocab:
        if not char_vocab._is_word_no_create_entry(char):
            cj_vocab.add_word_lst(char2cj(char))
    if include_word_start_end:
        cj_vocab.add_word_lst(['<bow>', '<eow>'])
    return cj_vocab

def _construct_wb_vocab_from_vocab(char_vocab: Vocabulary, min_freq: int = 1, include_word_start_end=True):
    r"""
    给定一个char的vocabulary生成character的vocabulary.

    :param vocab: 从vocab
    :param min_freq:
    :param include_word_start_end: 是否需要包含特殊的<bow>和<eos>
    :return:
    """
    wb_vocab = Vocabulary(min_freq=min_freq)
    for char, index in char_vocab:
        if not char_vocab._is_word_no_create_entry(char):
            wb_vocab.add_word_lst(char2wb(char))
    if include_word_start_end:
        wb_vocab.add_word_lst(['<bow>', '<eow>'])
    return wb_vocab

class CNNzixingEmbedding(TokenEmbedding):
    def __init__(self, vocab: Vocabulary, embed_size: int = 50, char_emb_size: int = 50, char_dropout: float = 0,
                 dropout: float = 0, filter_nums: List[int] = (40, 30, 20), kernel_sizes: List[int] = (5, 3, 1),
                 pool_method: str = 'max', activation='relu', min_char_freq: int = 2, pre_train_char_embed: str = None,
                 requires_grad: bool = True, include_word_start_end: bool = True):

        super(CNNzixingEmbedding, self).__init__(vocab, word_dropout=char_dropout, dropout=dropout)

        for kernel in kernel_sizes:
            assert kernel % 2 == 1, "Only odd kernel is allowed."

        assert pool_method in ('max', 'avg')
        self.pool_method = pool_method
        # activation function
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                self.activation = F.relu
            elif activation.lower() == 'sigmoid':
                self.activation = F.sigmoid
            elif activation.lower() == 'tanh':
                self.activation = F.tanh
        elif activation is None:
            self.activation = lambda x: x
        elif callable(activation):
            self.activation = activation
        else:
            raise Exception(
                "Undefined activation function: choose from: [relu, tanh, sigmoid, or a callable function]")
        logger.info("Start constructing character vocabulary.")

        self.cj_vocab = _construct_cj_vocab_from_vocab(vocab, min_freq=min_char_freq,
                                                                 include_word_start_end=include_word_start_end)
        self.wb_vocab = _construct_wb_vocab_from_vocab(vocab, min_freq=min_char_freq,
                                                            include_word_start_end=include_word_start_end)
        # exit() # 查看radical表
        self.char_pad_index_cj = self.cj_vocab.padding_idx
        self.char_pad_index_wb = self.wb_vocab.padding_idx
        logger.info(f"In total, there are {len(self.cj_vocab)} distinct characters.----cj")
        logger.info(f"In total, there are {len(self.wb_vocab)} distinct characters.----wb")
        # 对vocab进行index
        max_cj_nums = max(map(lambda x: len(char2cj(x[0])), vocab))
        max_wb_nums = max(map(lambda x: len(char2wb(x[0])), vocab))
        if include_word_start_end:
            max_cj_nums += 2
            max_wb_nums += 2

        self.register_buffer('chars_to_cj_embedding', torch.full((len(vocab), max_cj_nums),
                                                                       fill_value=self.char_pad_index_cj,
                                                                       dtype=torch.long))
        self.register_buffer('chars_to_wb_embedding', torch.full((len(vocab), max_wb_nums),
                                                                       fill_value=self.char_pad_index_wb,
                                                                       dtype=torch.long))
        self.register_buffer('word_lengths_cj', torch.zeros(len(vocab)).long())
        self.register_buffer('word_lengths_wb', torch.zeros(len(vocab)).long())
        for word, index in vocab:
            # if index!=vocab.padding_idx:  # 如果是pad的话，直接就为pad_value了。修改为不区分pad, 这样所有的<pad>也是同一个embed
            word_cj = char2cj(word)
            if include_word_start_end:
                word_cj = ['<bow>'] + word_cj + ['<eow>']
            self.chars_to_cj_embedding[index, :len(word_cj)] = \
                torch.LongTensor([self.cj_vocab.to_index(c) for c in word_cj])
            self.word_lengths_cj[index] = len(word_cj)

            word_wb = char2wb(word)
            if include_word_start_end:
                word_wb = ['<bow>'] + word_wb + ['<eow>']
            self.chars_to_wb_embedding[index, :len(word_wb)] = \
                torch.LongTensor([self.cj_vocab.to_index(c) for c in word_wb])
            self.word_lengths_wb[index] = len(word_wb)

        # self.char_embedding = nn.Embedding(len(self.char_vocab), char_emb_size)
        self.char_cj_embedding = get_embeddings((len(self.cj_vocab), char_emb_size))
        self.char_wb_embedding = get_embeddings((len(self.wb_vocab), char_emb_size))
        # self.char_embedding = StaticEmbedding(self.radical_vocab,
        #                                       model_dir_or_name='/home/ws/data/gigaword_chn.all.a2b.uni.ite50.vec')

        self.convs_cj = nn.ModuleList([nn.Conv1d(
            self.char_cj_embedding.embedding_dim, filter_nums[i], kernel_size=kernel_sizes[i], bias=True,
            padding=kernel_sizes[i] // 2)
            for i in range(len(kernel_sizes))])
        self.convs_wb = nn.ModuleList([nn.Conv1d(
            self.char_wb_embedding.embedding_dim, filter_nums[i], kernel_size=kernel_sizes[i], bias=True,
            padding=kernel_sizes[i] // 2)
            for i in range(len(kernel_sizes))])
        self._embed_size = embed_size
        self.fc = nn.Linear(sum(filter_nums), embed_size)
        self.requires_grad = requires_grad
        self.w_cj = nn.Linear(self.embed_size,1)
        self.w_wb = nn.Linear(self.embed_size,1)

    def forward(self, words):
        r"""
        输入words的index后，生成对应的words的表示。

        :param words: [batch_size, max_len]
        :return: [batch_size, max_len, embed_size]
        """
        words = self.drop_word(words)
        batch_size, max_len = words.size()
        chars_cj = self.chars_to_cj_embedding[words]  # batch_size x max_len x max_word_len
        chars_wb = self.chars_to_wb_embedding[words]  # batch_size x max_len x max_word_len
        word_lengths_cj = self.word_lengths_cj[words]  # batch_size x max_len
        word_lengths_wb = self.word_lengths_wb[words]  # batch_size x max_len
        max_word_len_cj = word_lengths_cj.max()
        max_word_len_wb = word_lengths_wb.max()
        chars_cj = chars_cj[:, :, :max_word_len_cj]
        chars_wb = chars_wb[:, :, :max_word_len_wb]
        # 为1的地方为mask
        chars_cj_masks = chars_cj.eq(self.char_pad_index_cj)  # batch_size x max_len x max_word_len 如果为0, 说明是padding的位置了
        chars_wb_masks = chars_wb.eq(self.char_pad_index_wb)  # batch_size x max_len x max_word_len 如果为0, 说明是padding的位置了
        chars_cj = self.char_cj_embedding(chars_cj)  # batch_size x max_len x max_word_len x embed_size
        chars_wb = self.char_wb_embedding(chars_wb)  # batch_size x max_len x max_word_len x embed_size
        chars_cj = self.dropout(chars_cj)
        chars_wb = self.dropout(chars_wb)

        reshaped_chars_cj = chars_cj.reshape(batch_size * max_len, max_word_len_cj, -1)
        reshaped_chars_wb = chars_wb.reshape(batch_size * max_len, max_word_len_wb, -1)
        reshaped_chars_cj = reshaped_chars_cj.transpose(1, 2)  # B' x E x M
        reshaped_chars_wb = reshaped_chars_wb.transpose(1, 2)  # B' x E x M

        conv_chars_cj = [conv(reshaped_chars_cj).transpose(1, 2).reshape(batch_size, max_len, max_word_len_cj, -1)
                      for conv in self.convs_cj]
        conv_chars_wb = [conv(reshaped_chars_wb).transpose(1, 2).reshape(batch_size, max_len, max_word_len_wb, -1)
                      for conv in self.convs_wb]

        conv_chars_cj = torch.cat(conv_chars_cj, dim=-1).contiguous()  # B x max_len x max_word_len x sum(filters)
        conv_chars_wb = torch.cat(conv_chars_wb, dim=-1).contiguous()  # B x max_len x max_word_len x sum(filters)
        conv_chars_cj = self.activation(conv_chars_cj)
        conv_chars_wb = self.activation(conv_chars_wb)

        if self.pool_method == 'max':
            conv_chars_cj = conv_chars_cj.masked_fill(chars_cj_masks.unsqueeze(-1), float('-inf'))
            chars_cj, _ = torch.max(conv_chars_cj, dim=-2)  # batch_size x max_len x sum(filters)
            conv_chars_wb = conv_chars_wb.masked_fill(chars_wb_masks.unsqueeze(-1), float('-inf'))
            chars_wb, _ = torch.max(conv_chars_wb, dim=-2)  # batch_size x max_len x sum(filters)
        else:
            conv_chars_cj = conv_chars_cj.masked_fill(chars_cj_masks.unsqueeze(-1), 0)
            chars_cj = torch.sum(conv_chars_cj, dim=-2) / chars_cj_masks.eq(False).sum(dim=-1, keepdim=True).float()
            conv_chars_wb = conv_chars_wb.masked_fill(chars_wb_masks.unsqueeze(-1), 0)
            chars_wb = torch.sum(conv_chars_wb, dim=-2) / chars_wb_masks.eq(False).sum(dim=-1, keepdim=True).float()
        chars_cj = self.fc(chars_cj)
        chars_wb = self.fc(chars_wb)
        chars_cj = self.dropout(chars_cj)
        chars_wb = self.dropout(chars_wb)
        q_cj=self.w_cj(chars_cj)
        q_wb=self.w_wb(chars_wb)
        
        a_cj=torch.exp(q_cj)/(torch.exp(q_cj)+torch.exp(q_wb))
        a_wb=torch.exp(q_wb)/(torch.exp(q_cj)+torch.exp(q_wb))
        cj_wb_embed=a_cj*chars_cj+a_wb*chars_wb

        return cj_wb_embed