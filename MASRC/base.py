import torch, einops, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class SelfAttention(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super(SelfAttention, self).__init__()

        self.q = nn.Linear(in_ch, out_ch, bias=False)
        self.k = nn.Linear(in_ch, out_ch, bias=False)
        self.v = nn.Linear(in_ch, out_ch, bias=False)
        self.scale = out_ch ** (-0.5)

        self.drop = nn.Dropout(dropout)

        self.initialize_weight(self.q)
        self.initialize_weight(self.k)
        self.initialize_weight(self.v)

    def forward(self, feat_q, feat_k=None, feat_v=None, mask=None):
        """
        :param feat_q: (batch, T, dim)
        :param feat_k: same
        :param feat_v: same
        :return:
        """
        if feat_k is None:
            feat_k, feat_v = feat_q[:], feat_q[:]

        batch, T, _ = feat_q.shape
        q = self.q(feat_q)
        k = self.k(feat_k)
        v = self.v(feat_v)

        sim = torch.matmul(q, k.transpose(2, 1)) * self.scale
        if mask is not None:
            sim = sim + mask
        sim = F.softmax(sim, dim=-1)
        v = torch.matmul(sim, v)

        v = self.drop(v)
        return v

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class GCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bias=False):
        super(GCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.linear = nn.Linear(self.in_ch, self.out_ch, bias=bias)

        self.initialize_weight(self.linear)

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

    def forward(self, feat, adj):
        """
        :param feat: (Batch, T, dim)
        :param adj:  (Batch, T, T)
        :return:
        """
        x = self.linear(feat)
        x = torch.matmul(adj, x)

        return x


class LGCN(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.2):
        super().__init__()
        self.Q = nn.Linear(in_ch, out_ch, bias=False)
        self.K = nn.Linear(in_ch, out_ch, bias=False)
        self.proj_att = nn.Linear(out_ch, 1, bias=False)
        self.drop = nn.Dropout(drop)

        self.initialize_weight(self.Q)
        self.initialize_weight(self.K)
        self.initialize_weight(self.proj_att)

    def forward(self, x, adj):
        """
        Learnable  Edge Weight GCN
        :param x: (B, T, D)
        :param adj: (B, T, T)
        :return:
        """
        mask = self.build_mask(adj)
        T = x.shape[1]
        feat_q = self.Q(x)
        feat_k = self.K(x)
        feat_q = feat_q[:, None].repeat(1, T, 1, 1)
        feat_k = feat_k[:, :, None].repeat(1, 1, T, 1)
        att = F.relu(feat_q + feat_k)
        att = self.drop(att)
        att = self.proj_att(att)
        att = torch.squeeze(att, dim=-1)

        att = F.softmax(att + mask, dim=-1)
        x = torch.matmul(att, x)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask


class ProcessSimilar(nn.Module):
    def __init__(self, in_channel=1):
        super(ProcessSimilar, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channel, 64, (3, 3), padding=1, padding_mode='replicate', bias=True)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), padding=1, padding_mode='replicate', bias=True)
        self.mpool = nn.MaxPool2d(2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), padding=1, padding_mode='replicate', bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), padding=1, padding_mode='replicate', bias=True)
        self.fconv = nn.Conv2d(128, 1, (1, 1), padding=0, bias=True)
        self.relu = nn.ReLU()

        self.initialize_weight(self.conv1_1)
        self.initialize_weight(self.conv1_2)
        self.initialize_weight(self.conv2_1)
        self.initialize_weight(self.conv2_2)
        self.initialize_weight(self.fconv)

    def forward(self, x):
        """
        :param x: (batch, T, T)
        :return:
            x:(batch, T/2, T/2)
        """
        x = x[:, None]

        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.mpool(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.fconv(x)
        x = torch.clamp(x, -1.0, 1.0)
        x = torch.squeeze(x, dim=1)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class CosSimilarity(nn.Module):
    def __init__(self, in_ch, out_ch, is_norm=False, is_scale=False):
        super(CosSimilarity, self).__init__()
        self.linear1 = nn.Linear(in_ch, out_ch, bias=False)
        self.linear2 = nn.Linear(in_ch, out_ch, bias=False)

        self.sqrt_dim = np.sqrt(out_ch)
        self.is_normlize = is_norm
        self.is_scale = is_scale

        self.initialize_weight(self.linear1)
        self.initialize_weight(self.linear2)

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

    def forward(self, feat1, feat2):
        """
        :param feat1: (batch, T, dim)
        :param feat2:
        :return:
        """
        batch, T, _ = feat2.shape
        feat1 = self.linear1(feat1)
        feat2 = self.linear2(feat2)
        if self.is_normlize:
            feat1 = F.normalize(feat1, dim=-1)
            feat2 = F.normalize(feat2, dim=-1)

        if self.is_scale:
            sim = torch.matmul(feat1, feat2.transpose(2, 1)) / self.sqrt_dim
        else:
            sim = torch.matmul(feat1, feat2.transpose(2, 1))

        return sim


class TemporalPosEmbed(nn.Module):
    def __init__(self, in_features=1, out_features=256, n_shot=20):
        super().__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

        self.pos_ids = torch.arange(n_shot, dtype=torch.float, device='cuda:0')[:, None]

    def forward(self, x):
        """
        :return:
        """
        B = x.shape[0]
        v1 = self.f(torch.matmul(self.pos_ids, self.w) + self.b)
        v2 = torch.matmul(self.pos_ids, self.w0) + self.b0
        v = torch.cat([v1, v2], -1)

        v = einops.repeat(v, 't n -> b t n', b=B)
        return v


class PosEmbed(nn.Module):
    def __init__(self, size, dim=256):
        super().__init__()
        self.size = size
        self.pe = nn.Embedding(size, dim)
        self.pos_ids = torch.arange(size, dtype=torch.long, device='cuda:0')

    def forward(self, x):
        pos_ids = einops.repeat(self.pos_ids, 'n -> b n', b=len(x))
        embeddings = torch.cat([x, self.pe(pos_ids)], dim=-1)
        return embeddings


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout =0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class Embedding(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.linear_1 = nn.Linear(in_ch, out_ch)
        self.linear_2 = nn.Linear(out_ch, out_ch)
        self.relu = nn.ReLU()

        self.initialize_weight(self.linear_1)
        self.initialize_weight(self.linear_2)

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)

        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super(AbsolutePositionalEncoder, self).__init__()
        self.position = torch.arange(max_position).unsqueeze(1)
        self.positional_encoding = torch.zeros(max_position, emb_dim, device="cuda:0")

        self.w = nn.Linear(emb_dim, emb_dim, bias=False)
        nn.init.xavier_uniform_(self.w.weight)

        _2i = torch.arange(0, emb_dim, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        self.positional_encoding[:, 0::2] = torch.sin(self.position / (10000 ** (_2i / emb_dim)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        self.positional_encoding[:, 1::2] = torch.cos(self.position / (10000 ** (_2i / emb_dim)))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        pos_emb = self.w(self.positional_encoding)
        return einops.repeat(pos_emb, 't n -> b t n', b=batch_size)


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()
        if normalization == 'batch':
            self.normalizer = nn.BatchNorm1d(embed_dim, affine=True)
        else:
            self.normalizer = nn.LayerNorm(embed_dim, eps=1e-6)

        self.type = normalization
        # self.T = timesz

        # Normalization by default initializes affine parameters
        # with bias 0 and weight unif(0,1) which is too large!
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.type == 'batch':
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        else:
            return self.normalizer(input)