import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import GCNBlock, SelfAttention, \
     Normalization, GraphRes, MHABlock, SelfRes, GAT, FeedForward
from torch.autograd import Variable


class BackgroundNet(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.2):
        super().__init__()
        # v4
        # self.fore_inter = SelfAttention(in_ch, out_ch, dropout=drop)
        # self.fore_norm = nn.LayerNorm(out_ch, eps=1e-6)
        #
        # self.back = SelfAttention(out_ch, out_ch, dropout=drop)
        # self.back_norm = nn.LayerNorm(out_ch, eps=1e-6)
        #
        # self.relu = nn.ReLU()

        # v5
        self.gcn1 = SelfRes(in_ch, out_ch, mode='v2')
        self.drop = nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(out_ch, eps=1e-6)
        self.gcn2 = SelfRes(in_ch, out_ch, mode='v2')
        self.norm2 = nn.LayerNorm(out_ch, eps=1e-6)
        self.relu = nn.ReLU()

    def forward(self, x, graphs):
        """
        :param feat: (Batch, T, dim)
        :param graphs: (B, 3, T, T), 0:foreward_intra, 1: foreward_inter, 2: backward
        :return:
        """
        # # forward and backward v3
        # inter_mask = self.build_mask(graphs[:, 0])
        # back_mask = self.build_mask(graphs[:, 1])
        #
        # _x = x
        # x = self.relu(self.fore_inter(x, mask=inter_mask))
        # x = self.fore_norm(_x + x)
        #
        # _x = x
        # x = self.relu(self.back(x, mask=back_mask))
        # x = self.back_norm(x + _x)

        # v4
        inter_mask = graphs[:, 0]
        back_mask = graphs[:, 1]
        x = self.gcn1(x, inter_mask)
        x = self.gcn2(x, back_mask)
        x = self.drop(x)

        return x

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask


class TextNet(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.1):
        super().__init__()
        # self.att1 = SelfAttention(in_ch, out_ch, dropout=drop)
        self.gcn1 = GraphRes(in_ch, out_ch, mode='v2')
        self.norm1 = nn.LayerNorm(in_ch, eps=1e-6)

        self.gcn2 = GraphRes(in_ch, out_ch, mode='v2')
        self.norm2 = nn.LayerNorm(in_ch, eps=1e-6)
        self.drop = nn.Dropout(drop)

        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor, graph):
        x = self.gcn1(x, graph)
        x = self.gcn2(x, graph)
        x = self.drop(x)

        return x

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask


class ForegroundNet(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.1):
        super().__init__()
        self.gcn1 = GraphRes(in_ch, out_ch, mode='v2')
        # self.gcn1 = GCNBlock(in_ch, out_ch, bias=False)
        # self.gcn1 = GAT(in_ch, out_ch, 'mlp')
        self.norm1 = Normalization(out_ch, 'ln')
        self.drop = nn.Dropout(drop)
        # self.norm = nn.LayerNorm(out_ch, eps=1e-6)

        self.gcn2 = GraphRes(in_ch, out_ch, mode='v2')
        # self.gcn2 = GCNBlock(out_ch, out_ch, bias=False)
        # self.gcn2=GAT(in_ch, out_ch, 'mlp')
        self.norm2 = Normalization(out_ch, 'ln')

        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor, graph):
        # _x = x
        # x = self.relu(_x+self.gcn1(x, graph))
        # x = self.norm1(x)
        #
        # _x = x
        # x = self.relu(_x+self.gcn2(x, graph))

        x = self.gcn1(x, graph)
        x = self.gcn2(x, graph)
        x = self.drop(x)

        return x

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask





