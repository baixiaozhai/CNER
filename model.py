import math

import torch
import numpy as np
from fastNLP import seq_len_to_mask
from torch import nn

from Modules.MyDropout import MyDropout
from Modules.TransformerEncoder import TransformerEncoder
from Utils.utils import get_crf_zero_init, get_embedding


class MECTNER(nn.Module):
    def __init__(self, lattice_embed, bigram_embed,cj_wb_embed, hidden_size,
                 k_proj, q_proj, v_proj, r_proj,
                 label_size, max_seq_len, dropout, dataset, ff_size):
        super().__init__()

        self.dataset = dataset

        self.lattice_embed = lattice_embed
        self.bigram_embed = bigram_embed
        # self.components_embed = components_embed
        self.cj_wb_embed = cj_wb_embed
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # 超参数
        self.rel_pos_init = 0
        self.learnable_position = False
        self.num_heads = 8
        self.layer_preprocess_sequence = ""
        self.layer_postprocess_sequence = "an"
        self.dropout = dropout
        self.scaled = False
        self.ff_size = ff_size
        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = True
        self.ff_activate = 'relu'
        self.use_pytorch_dropout = 0
        self.embed_dropout = MyDropout(self.dropout['embed'])
        self.gaz_dropout = MyDropout(self.dropout['gaz'])
        self.output_dropout = MyDropout(self.dropout['output'])

        pe = get_embedding(max_seq_len, self.hidden_size, rel_pos_init=self.rel_pos_init)
        self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
        self.pe_ss = self.pe
        self.pe_ee = self.pe

        self.lex_input_size = self.lattice_embed.embed_size
        self.bigram_size = self.bigram_embed.embed_size
        # self.components_embed_size = self.components_embed.embed_size
        self.cj_wb_embed_size = self.cj_wb_embed.embed_size
        self.char_input_size = self.lex_input_size + self.bigram_size   #原
        #self.char_input_size = self.lex_input_size + self.bigram_size +self.components_embed.embed_size #改

        self.char_proj = nn.Linear(self.char_input_size, self.hidden_size)
        self.lex_proj = nn.Linear(self.lex_input_size, self.hidden_size)
        # self.components_proj = nn.Linear(self.components_embed_size, self.hidden_size)
        self.cj_wb_proj = nn.Linear(self.cj_wb_embed_size, self.hidden_size)

        self.char_encoder = TransformerEncoder(self.hidden_size, self.num_heads,
                                               dataset=self.dataset,
                                               layer_preprocess_sequence=self.layer_preprocess_sequence,
                                               layer_postprocess_sequence=self.layer_postprocess_sequence,
                                               dropout=self.dropout,
                                               scaled=self.scaled,
                                               ff_size=self.ff_size,
                                               max_seq_len=self.max_seq_len,
                                               pe=self.pe,
                                               pe_ss=self.pe_ss,
                                               pe_ee=self.pe_ee,
                                               ff_activate=self.ff_activate,
                                               use_pytorch_dropout=self.use_pytorch_dropout)

        self.radical_encoder = TransformerEncoder(self.hidden_size, self.num_heads,
                                                  dataset=self.dataset,
                                                  layer_preprocess_sequence=self.layer_preprocess_sequence,
                                                  layer_postprocess_sequence=self.layer_postprocess_sequence,
                                                  dropout=self.dropout,
                                                  scaled=self.scaled,
                                                  ff_size=self.ff_size,
                                                  max_seq_len=self.max_seq_len,
                                                  pe=self.pe,
                                                  pe_ss=self.pe_ss,
                                                  pe_ee=self.pe_ee,
                                                  ff_activate=self.ff_activate,
                                                  use_pytorch_dropout=self.use_pytorch_dropout)

        self.output = nn.Linear(self.hidden_size*2, self.label_size)
        #self.output = nn.Linear(self.hidden_size , self.label_size) #改

        self.crf = get_crf_zero_init(self.label_size)

        self.w1 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.w2 = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.reset_parameters()
    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.w1.size(1))
        stdv2 = 1. / math.sqrt(self.w2.size(1))
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, lattice, bigrams,  seq_len, lex_num, pos_s, pos_e, target): #
        batch_size = lattice.size(0)
        max_seq_len_and_lex_num = lattice.size(1)
        max_seq_len = bigrams.size(1)

        raw_embed = self.lattice_embed(lattice)

        char_mask = seq_len_to_mask(seq_len, max_len=max_seq_len_and_lex_num).bool()
        char = lattice.masked_fill_(~char_mask, 0)
        #chaizi
        # components_embed = self.components_embed(char)
        # components_embed.masked_fill_(~(char_mask).unsqueeze(-1), 0)
        # components_embed = self.components_proj(components_embed)   #原

        #cj_wb
        cj_wb_embed = self.cj_wb_embed(char)
        cj_wb_embed.masked_fill_(~(char_mask).unsqueeze(-1),0)
        cj_wb_embed = self.cj_wb_proj(cj_wb_embed)

        bigrams_embed = self.bigram_embed(bigrams)
        bigrams_embed = torch.cat([bigrams_embed,
                                   torch.zeros(size=[batch_size, max_seq_len_and_lex_num - max_seq_len,
                                                     self.bigram_size]).to(bigrams_embed)], dim=1)
        raw_embed_char = torch.cat([raw_embed, bigrams_embed], dim=-1)
        #raw_embed_char = torch.cat([raw_embed_char,components_embed],dim=-1) #改

        raw_embed_char = self.embed_dropout(raw_embed_char)
        raw_embed = self.gaz_dropout(raw_embed)

        embed_char = self.char_proj(raw_embed_char)
        char_mask = seq_len_to_mask(seq_len, max_len=max_seq_len_and_lex_num).bool()
        embed_char.masked_fill_(~(char_mask.unsqueeze(-1)), 0)

        embed_lex = self.lex_proj(raw_embed)
        lex_mask = (seq_len_to_mask(seq_len + lex_num).bool() ^ char_mask.bool())
        embed_lex.masked_fill_(~lex_mask.unsqueeze(-1), 0)

        assert char_mask.size(1) == lex_mask.size(1)
        embedding=embed_char+embed_lex
        ##原
        char_encoded = self.char_encoder(embedding, embedding, embedding, seq_len,lex_num=lex_num, pos_s=pos_s,
                                         pos_e=pos_e)
        char_encoded = self.char_encoder(char_encoded, char_encoded, char_encoded, seq_len,lex_num=lex_num, pos_s=pos_s,
                                         pos_e=pos_e)
        # radical_encoded = self.radical_encoder(embedding, components_embed, components_embed, seq_len,
        #                                        lex_num=lex_num, pos_s=pos_s, pos_e=pos_e)

        # fusion = torch.cat([cj_wb_embed, char_encoded], dim=-1)
        #门控机制
        gated = char_encoded.matmul(self.w1.t()) + cj_wb_embed.matmul(self.w2.t()) + self.bias
        gate = torch.sigmoid(gated)
        fusion = torch.cat([char_encoded.mul(gate), cj_wb_embed.mul(1 - gate)], dim=2)
        #fusion = char_encoded.mul(gate) + cj_wb_embed.mul(1 - gate)
        
        output = self.output_dropout(fusion)
        output = output[:, :max_seq_len, :]
        pred = self.output(output)

        mask = seq_len_to_mask(seq_len).bool()

        if self.training:
            loss = self.crf(pred, target, mask).mean(dim=0)
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            result = {'pred': pred}

            return result
