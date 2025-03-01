import torch
import torch.nn as nn
import numpy as np
from model.base import TemporalCrossAttent, GCNBlock, \
    RelateNet, SelfAttention, TGAT, FeedForward, Normalization, MHA
from torch.autograd import Variable


class PCTCR(nn.Module):
    def __init__(self, cst_in, plc_in, win=4):
        """
        Temporal Cross Attention for interaction between place and cast
        :param cst_in:
        :param plc_in:
        :param win:
        """
        super(PCTCR, self).__init__()
        self.p2c = TemporalCrossAttent(cst_in, plc_in, plc_in, hid_ch=cst_in, win_size=win, is_scale=True)
        self.c2p = TemporalCrossAttent(plc_in, cst_in, cst_in, hid_ch=plc_in, win_size=win, is_scale=True, is_drop=True)

        self.cst_norm = nn.LayerNorm(cst_in)
        self.plc_norm = nn.LayerNorm(plc_in)

    def forward(self, feat_cst, feat_plc):
        feat_p2c = self.p2c(feat_cst, feat_plc)
        feat_c2p = self.c2p(feat_plc, feat_cst)

        feat_cst = self.cst_norm(feat_cst + feat_p2c)
        feat_plc = self.plc_norm(feat_plc + feat_c2p)

        return feat_cst, feat_plc


class F2BNet(nn.Module):
    def __init__(self, in_ch, hid_ch=128, drop=0.1):
        """
        Custom GraphNN for interaction between place and cast
        :param in_ch:
        """
        super().__init__()
        self.fgcn = GCNBlock(in_ch, in_ch, drop)
        self.fgcn_att = RelateNet(in_ch, in_ch, hid_ch)
        self.fgcn_norm = Normalization(in_ch, 'ln')

        self.bgcn = SelfAttention(in_ch, in_ch, dropout=drop)
        self.bgcn_norm = Normalization(in_ch, 'ln')
        self.relu = nn.ReLU()

    def forward(self, ftx, btx, graphs):
        """
        :param ftx:
        :param btx:
        :param graphs: 0: full connected fore-graph; 1: full connected back-graph
        :return:
        """
        _x = btx
        f2b_att = self.fgcn_att(ftx, btx, graphs[:, 0])
        x = self.relu(self.fgcn(btx, f2b_att))
        x = self.fgcn_norm(_x + x)

        _x = x
        mask = self.build_mask(graphs[:, 1])
        x = self.relu(self.bgcn(x, x, x, mask))
        x = self.bgcn_norm(_x + x)

        return x

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask


class B2FNet(nn.Module):
    def __init__(self, in_ch, hid_ch=128, drop=0.1):
        super().__init__()
        self.fgcn = SelfAttention(in_ch, in_ch, drop)
        self.fgcn_norm = Normalization(in_ch, 'ln')

        self.bgcn = GCNBlock(in_ch, in_ch, drop)
        self.bgcn_att = RelateNet(in_ch, in_ch, hid_ch)
        self.bgcn_norm = Normalization(in_ch, 'ln')

        self.relu = nn.ReLU()

    def forward(self, ftx, btx, graphs):
        _x = ftx
        mask = self.build_mask(graphs[:, 1])
        x = self.relu(self.fgcn(ftx, ftx, ftx, mask))
        x = self.fgcn_norm(_x + x)

        _x = x
        x_att = self.bgcn_att(ftx, btx, graphs[:, 0])
        x = self.relu(self.bgcn(x, x_att))
        # x = self.bgcn_norm(_x + x)

        return x

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask