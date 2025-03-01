import torch, einops, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from einops import rearrange
from typing import Optional


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
    def __init__(self, in_channel, out_channel, bias=False):
        super(GCNBlock, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
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


class ProcessSimilar(nn.Module):
    def __init__(self, in_channel=1):
        super(ProcessSimilar, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channel, 64, (3, 3), padding=1, padding_mode='replicate', bias=True)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), padding=1,  padding_mode='replicate', bias=True)
        self.mpool = nn.MaxPool2d(2, stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), padding=1,  padding_mode='replicate', bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), padding=1,  padding_mode='replicate', bias=True)
        self.fconv = nn.Conv2d(128, 1, (1, 1),  padding=0, bias=True)
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
        if len(x.shape)!=4:
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


class PyConv2d(nn.Module):
    def __init__(self, in_ch, out_ch:list, py_kernel:list, py_group:list, stride=1, bias=False):
        super().__init__()
        self.py_level = [None]* len(py_kernel)
        for i in range(len(py_kernel)):
            self.py_level[i] = nn.Conv2d(in_ch, out_ch[i], kernel_size=py_kernel[i],
                                         stride=stride, padding=py_kernel[i]//2, padding_mode='replicate',
                                         groups=py_group[i], bias=bias
                                         )
        self.py_level = nn.ModuleList(self.py_level)

    def forward(self, x):
        out = []
        for level in self.py_level:
            out.append(level(x))

        return torch.cat(out, 1)


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


class GAT(nn.Module):
    def __init__(self, in_ch, out_ch, mode='mlp'):
        super(GAT, self).__init__()
        self.mode = mode
        self.linear = nn.Linear(in_ch, out_ch, bias=False)
        self.drop = nn.Dropout(0.2)
        self.sqrt_dim = out_ch**(-0.5)
        if self.mode == 'mlp':
            # self.net = nn.Sequential(
            #     nn.Linear(in_ch*2, out_ch),
            #     nn.ReLU(),
            #     nn.Dropout(0.1),
            #     nn.Linear(out_ch, 1),
            # )
            self.net = nn.Sequential(
                nn.Conv2d(in_ch*2, out_ch, (1, 1), bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(out_ch, 1, (1, 1), bias=False)
            )

            for layer in self.net:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv1d):
                    self.initialize_weight(layer)
        if self.mode == 'v2':
            self.proj_1 = nn.Linear(in_ch, out_ch, bias=False)
            self.proj_2 = nn.Linear(in_ch, out_ch, bias=False)
            self.proj_att = nn.Linear(out_ch, 1, bias=False)
        if self.mode == 'self':
            self.Q = nn.Linear(in_ch, out_ch, bias=False)
            self.K = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, feat, adj):
        mask = self.build_mask(adj)
        T = feat.shape[1]
        # feat_proj = self.drop(self.linear(feat))
        if self.mode == 'mlp':
            feat_rep = feat[:, :, None].repeat(1, 1, T, 1)
            feat_rep = torch.cat((feat_rep, feat_rep.transpose(2, 1)), dim=-1)
            att = self.net(feat_rep.permute((0, -1, 1, 2)).contiguous())
            att = torch.squeeze(att, dim=1)
        if self.mode == 'v2':
            feat_q = self.proj_1(feat)
            feat_k = self.proj_2(feat)
            feat_q = feat_q[:, None].repeat(1, T, 1, 1)
            feat_k = feat_k[:, :, None].repeat(1, 1, T, 1)
            att = F.relu(feat_q + feat_k)
            att = self.drop(att)
            att = self.proj_att(att)
            att = torch.squeeze(att, dim=-1)
            # att = F.relu(att)
        if self.mode == 'self':
            q = self.Q(feat)
            k = self.K(feat)
            att = torch.matmul(q, k.transpose(2, 1)) * self.sqrt_dim

        att = F.softmax(att + mask, dim=-1)

        feat_att = torch.matmul(att, feat)
        return feat_att

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask


class TemporalCrossAttent(nn.Module):
    def __init__(self, q_in, k_in, v_in, hid_ch, T=20, win_size=3, is_scale=True, is_drop=False):
        super(TemporalCrossAttent, self).__init__()
        self.linear_q = nn.Conv1d(q_in, hid_ch, kernel_size=1, bias=False)
        self.linear_k = nn.Conv1d(k_in, hid_ch, kernel_size=1, bias=False)
        self.linear_v = nn.Conv1d(v_in, hid_ch, kernel_size=1, bias=False)

        self.is_drop = is_drop
        self.qkdrop = nn.Dropout(0.75)

        self.sqrt_dim = np.sqrt(hid_ch)
        self.is_scale = is_scale

        self.tmask = self.TB(T, win_size)
        self.drop = nn.Dropout(0.1)

        self.initialize_weight(self.linear_q)
        self.initialize_weight(self.linear_k)
        self.initialize_weight(self.linear_v)

    def forward(self, feat1, feat2):
        B, T, _ = feat1.shape

        mask = self.tmask.cuda(feat1.get_device())
        # mask = self.build_mask(adj)

        feat1 = feat1.permute(0, 2, 1).contiguous()
        feat2 = feat2.permute(0, 2, 1).contiguous()

        q = self.linear_q(feat1).permute(0, 2, 1).contiguous()
        k = self.linear_k(feat2).permute(0, 2, 1).contiguous()
        v = self.linear_v(feat2).permute(0, 2, 1).contiguous()

        if self.is_drop:
            q = self.qkdrop(q)
            k = self.qkdrop(k)

        if self.is_scale:
            sim = torch.matmul(q, k.transpose(2, 1)) / self.sqrt_dim

        sim = sim + mask
        sim = F.softmax(sim, dim=-1)
        v = torch.matmul(sim, v)

        v = self.drop(v)
        return v

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)

    def TB(self, T=20, win=5):
        tmask = torch.zeros((T, T), dtype=torch.float32)

        for i in range(T):
            if i - win < 0:
                tmask[i, :win] = 1
            elif i + win >= T:
                tmask[i, T - win:] = 1
            else:
                tmask[i - win // 2, i - win // 2:i + win // 2] = 1

        tmask = tmask.masked_fill_(tmask == 0, -float(1e22)).masked_fill_(tmask == 1, float(0))
        return tmask

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask == 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask


class RelateNet(nn.Module):
    def __init__(self, f_ch, g_ch, hid_ch):
        """
        :param f_ch:
        :param g_ch:
        :param hid_ch:
        """
        super(RelateNet, self).__init__()
        self.proj_f = nn.Linear(f_ch, hid_ch)
        self.proj_g = nn.Linear(g_ch, hid_ch)

        self.conv_f = nn.Conv2d(hid_ch, hid_ch, (1, 1), bias=True)
        self.conv_g = nn.Conv2d(hid_ch, hid_ch, (1, 1), bias=True)
        self.a = nn.Conv2d(hid_ch, 1, (1, 1), bias=False)

        self.relu = nn.ReLU()
        self.actv = nn.Sigmoid()

        self.initialize_weight(self.proj_g)
        self.initialize_weight(self.proj_f)
        self.initialize_weight(self.conv_f)
        self.initialize_weight(self.conv_g)
        self.initialize_weight(self.a)

    def forward(self, feat: torch.Tensor, gfeat: torch.Tensor, adj: torch.Tensor):
        """
        :param feat:
        :param adj:
        :return:
        """
        mask = self.build_mask(adj)
        proj_f = self.proj_f(feat)
        proj_g = self.proj_g(gfeat)

        # proposed
        S_f = proj_f[:, :, None] * proj_f[:, None]
        S_g = proj_g[:, :, None] * proj_g[:, None]
        S_f = F.normalize(S_f, dim=-1)
        S_g = F.normalize(S_g, dim=-1)

        S_f = torch.permute(S_f, (0, -1, 1, 2)).contiguous()
        S_g = torch.permute(S_g, (0, -1, 1, 2)).contiguous()

        S_fcov = self.conv_f(S_f)
        S_gcov = self.conv_g(S_g)
        S = S_fcov + S_gcov

        # # context-aware attention
        # g_att = self.actv(self.conv_ctx(S_g))
        # S = S_fcov + g_att * S_gcov

        att = self.a(S).squeeze(dim=1)
        att = F.softmax(att+mask, dim=-1)

        return att

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask == 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class TGAT(nn.Module):
    def __init__(self, f_ch, g_ch, hid_ch):
        """
        :param f_ch:
        :param g_ch:
        :param hid_ch:
        """
        super(TGAT, self).__init__()
        self.proj_f = nn.Linear(f_ch, hid_ch)
        self.proj_g = nn.Linear(g_ch, hid_ch)
        self.proj_att = nn.Linear(hid_ch, 1, bias=False)
        self.drop = nn.Dropout(0.5)

        self.conv_f = nn.Conv2d(1, 1, (1, 1), bias=True)
        self.conv_g = nn.Conv2d(1, 1, (1, 1), bias=True)
        # self.conv_ctx = nn.Conv2d(hid_ch, 1, (3, 3), (1, 1), (1, 1), bias=True)
        # # self.pool = nn.AvgPool2d((3, 3), (1, 1), (1, 1))
        self.a = nn.Conv2d(1, 1, (1, 1), bias=False)
        #
        self.relu = nn.ReLU()
        # self.actv = nn.Sigmoid()

        self.initialize_weight(self.proj_g)
        self.initialize_weight(self.proj_f)
        # self.initialize_weight(self.proj_att)
        self.initialize_weight(self.conv_f)
        self.initialize_weight(self.conv_g)
        self.initialize_weight(self.a)

    def forward(self, feat: torch.Tensor, gfeat: torch.Tensor, adj: torch.Tensor):
        """
        :param feat:
        :param adj:
        :return:
        """
        mask = self.build_mask(adj)
        T = feat.shape[1]

        feat_q = self.proj_f(feat)
        feat_k = self.proj_g(gfeat)
        # GAT
        # feat_q = feat_q[:, None].repeat(1, T, 1, 1)
        # feat_k = feat_k[:, :, None].repeat(1, 1, T, 1)
        # att = self.relu(feat_q + feat_k)
        # att = self.drop(att)
        # att = self.proj_att(att)
        # att = torch.squeeze(att, dim=-1)

        # similar value
        S_f = torch.matmul(feat_q, feat_q.transpose(2, 1))
        S_g = torch.matmul(feat_k, feat_k.transpose(2, 1))

        # S_f = torch.permute(S_f[:, None], (0, -1, 1, 2)).contiguous()
        # S_g = torch.permute(S_g[:, None], (0, -1, 1, 2)).contiguous()

        S_fcov = self.conv_f(S_f[:, None])
        S_gcov = self.conv_g(S_g[:, None])
        S = S_fcov + S_gcov

        att = self.a(S).squeeze(dim=1)
        att = F.softmax(att + mask, dim=-1)

        return att

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask == 0, -float(1e22)).masked_fill_(mask > 0, float(0))
        return mask

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


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


class TFeedForward(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel=3, drop=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, hid_ch, kernel_size=kernel, padding=1)
        self.conv2 = nn.Conv1d(hid_ch, in_ch, kernel_size=kernel, padding=1)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

        self.initialize_weight(self.conv1)
        self.initialize_weight(self.conv2)

    def forward(self, x:torch.Tensor):
        x = self.relu(self.conv1(x.transpose(-1, -2).contiguous()))
        x = self.drop(x)
        x = self.conv2(x)

        return x.transpose(-1, -2).contiguous()

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class TCN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, drop=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=int(kernel//2))
        self.norm = Normalization(out_ch, 'ln')
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

        self.initialize_weight(self.conv)

    def forward(self, x):
        x = x+self.conv(x.transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous()
        x = self.relu(x)
        x = self.drop(x)
        x = self.norm(x)
        return x

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


class RelativePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=20):
        super(RelativePositionalEncoder, self).__init__()
        self.max_position = max_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_position * 2 + 1, emb_dim))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, seq_len_q, seq_len_k):
        range_vec_q = torch.arange(seq_len_q)
        range_vec_k = torch.arange(seq_len_k)
        relative_matrix = range_vec_k[None, :] - range_vec_q[:, None]
        clipped_relative_matrix = torch.clamp(relative_matrix, -self.max_position, self.max_position)
        relative_position_matrix = clipped_relative_matrix + self.max_position
        embeddings = self.embeddings_table[relative_position_matrix]

        return embeddings


class T5RelativePositionalEncoder(nn.Module):
    def __init__(self, num_heads, max_position=512):
        super(T5RelativePositionalEncoder, self).__init__()
        self.max_position = max_position
        self.embeddings_table = nn.Embedding(max_position*max_position, num_heads)

    def forward(self, seq_len_q, seq_len_k):
        range_vec_q = torch.arange(seq_len_q)
        range_vec_k = torch.arange(seq_len_k)
        relative_position = range_vec_k[None, :] - range_vec_q[:, None]
        relative_position_clipped = torch.clamp(relative_position, -self.max_position, self.max_position)
        final_mat = relative_position_clipped + self.max_position
        embeddings = self.embeddings_table(final_mat)

        return embeddings


class MHAM(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim * 2, bias = False)
        self.initialize_weight(self.to_qk)

    def forward(self, x):
        qk = self.to_qk(x).chunk(2, dim = -1)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qk)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        return attn

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class HRE(nn.Module):
    def __init__(self, in_ch, heads=8, modal=2, seg_sz=20):
        super().__init__()
        dim_head = in_ch//heads
        self.mhts_1 = MHAM(in_ch, dim_head=dim_head, heads=heads, dropout=0.1)
        self.mhts_2 = MHAM(in_ch, dim_head=dim_head, heads=heads, dropout=0.1)
        self.cnn = nn.Conv2d(heads*modal, heads*modal, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.pos_emb = RelativePositionalEncoder(heads*modal, seg_sz)
        self.transform = SelfAttention(heads*modal, heads*modal)
        self.fuse = nn.Linear(modal*heads, 1)
        self.sigmoid = nn.Sigmoid()

        self.seg_sz = seg_sz

    def forward(self, x1, x2):
        B, T, _ = x1.shape
        pos_emb = self.pos_emb(self.seg_sz, self.seg_sz)[None]
        sim_1 = self.mhts_1(x1)
        sim_2 = self.mhts_2(x2)
        sim = torch.concat((sim_1, sim_2), dim=1)
        sim = self.cnn(sim) + pos_emb.permute(0, -1, 1, 2)

        sim = sim.permute(0, 2, 3, 1).contiguous()
        sim = sim.view(B, T*T, -1).contiguous()
        sim = self.transform(sim)
        sim = self.sigmoid(self.fuse(sim).view(-1, T, T))

        return sim


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


class MHA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            # if mask.shape[1] ==2:
            #     dots[:, :self.heads//2] += mask[:, 0][:, None]
            #     dots[:, self.heads // 2:] += mask[:, 1][:, None]
            # else:
            dots += mask
            attn = self.attend(dots)
        else:
            attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class MHABlock(nn.Module):
    def __init__(self, dim, heads = 8,  dropout = 0.):
        super().__init__()
        self.self = MHA(dim, heads, dim//heads, 0)
        self.out_ln = nn.Linear(dim, dim)
        self.out_norm = Normalization(dim, 'ln')
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        _x = x
        x = self.self(x, mask)
        x = self.out_drop(self.out_ln(x))
        x = self.out_norm(x + _x)

        return x


class GraphRes(nn.Module):
    def __init__(self, in_ch, out_ch, mode='gcn'):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, out_ch//2)
        if mode == 'gcn':
            self.gconv = GCNBlock(out_ch//2, out_ch//2)
        elif mode == 'self':
            self.gconv = GAT(out_ch//2, out_ch//2, mode='self')
        elif mode == 'v2':
            self.gconv = GAT(out_ch // 2, out_ch // 2, mode='v2')
        else:
            self.gconv = GAT(out_ch // 2, out_ch // 2, mode='mlp')
        self.lin2 = nn.Linear(out_ch//2, out_ch)

        self.pre_norm = Normalization(in_ch, 'ln')
        self.norm1 = Normalization(out_ch//2, 'ln')
        self.norm2 = Normalization(out_ch // 2, 'ln')

        self.init_parameters(self.lin1)
        self.init_parameters(self.lin2)

    def forward(self, x, adj):
        _x = x
        # x = F.relu(self.pre_norm(x))
        x = self.lin1(x)
        x = F.relu(self.norm1(x))
        x = self.gconv(x, adj)

        x = F.relu(self.norm2(x))
        x = self.lin2(x)

        return x + _x

    def init_parameters(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class SelfRes(nn.Module):
    def __init__(self, in_ch, out_ch, mode='gcn'):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, out_ch // 2)
        if mode == 'gcn':
            self.gconv = GCNBlock(out_ch // 2, out_ch // 2)
        elif mode == 'self':
            self.gconv = GAT(out_ch // 2, out_ch // 2, mode='self')
        elif mode == 'v2':
            self.gconv = GAT(out_ch // 2, out_ch // 2, mode='v2')
        else:
            self.gconv = GAT(out_ch // 2, out_ch // 2, mode='mlp')
        self.lin2 = nn.Linear(out_ch // 2, out_ch)

        self.pre_norm = Normalization(in_ch, 'ln')
        self.norm1 = Normalization(out_ch // 2, 'ln')
        self.norm2 = Normalization(out_ch // 2, 'ln')

        self.init_parameters(self.lin1)
        self.init_parameters(self.lin2)

    def forward(self, x, adj):
        _x = x
        # x = F.relu(self.pre_norm(x))
        x = self.lin1(x)
        x = F.relu(self.norm1(x))
        x = self.gconv(x, adj)

        x = F.relu(self.norm2(x))
        x = self.lin2(x)
        return x + _x

    def init_parameters(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 115,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()


    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None):
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)



