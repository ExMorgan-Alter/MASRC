import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import vit

from model.base import HRE, Normalization,FeedForward, Embedding, AbsolutePositionalEncoder
from model.detector import ICCV2023, Detector, Bassl
from einops import rearrange


class MHRE_block(nn.Module):
    def __init__(self, in_ch, proj_ch, modal=2, heads=8, dropout=0.5):
        super().__init__()
        self.embedding = Embedding(in_ch*2, proj_ch-256)
        self.tpos = AbsolutePositionalEncoder(256, 14)
        self.attend = HRE(in_ch, heads=heads, modal=modal, seg_sz=14)

        # For Transformer
        dim_head = proj_ch//heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(proj_ch, proj_ch * 3, bias=False)

        self.mha_norm = Normalization(proj_ch, 'ln')

        self.ffc = FeedForward(proj_ch, int(proj_ch*1.5), dropout =dropout)

    def forward(self, x1, x2):
        hr = self.attend(x1, x2)
        x = torch.concat((x1, x2), dim=-1)
        pos_emb = self.tpos(x)
        x = self.embedding(x)
        x = torch.concat((x, pos_emb), dim=-1)

        # For transformer
        _x = x
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots) * hr[:, None]
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.mha_norm(x+_x)

        x = self.ffc(x)

        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

# from vit_pytorch.vit


class MHRE(nn.Module):
    def __init__(self, in_ch, proj_ch, heads=8, dropout=0.2):
        super().__init__()
        self.htrn = MHRE_block(in_ch, proj_ch, heads=heads, dropout=dropout)
        self.detector = ICCV2023(proj_ch)
        # self.detect = Detector(proj_ch, dropout, 20)
        # self.detect = Bassl(proj_ch)

    def forward(self, x1:torch.Tensor, x2:torch.Tensor):
        x = self.htrn(x1, x2)
        pred = self.detector(x)

        return pred





