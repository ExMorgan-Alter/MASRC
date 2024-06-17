import torch
import torch.nn as nn
from MASRC.base import GCNBlock, LGCN, Normalization
from torch.autograd import Variable


class PCGNet(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.2):
        super().__init__()
        # v4
        self.fore_inter = LGCN(in_ch, out_ch, dropout=drop)
        self.fore_norm = nn.LayerNorm(out_ch, eps=1e-6)

        self.back = LGCN(out_ch, out_ch, dropout=drop)
        self.back_norm = nn.LayerNorm(out_ch, eps=1e-6)

        self.relu = nn.ReLU()

    def forward(self, x, graphs):
        """
        :param feat: (Batch, T, dim)
        :param graphs: (B, 3, T, T), 0:foreward_intra, 1: foreward_inter, 2: backward
        :return:
        """
        # # forward and backward v3
        inter_mask = self.build_mask(graphs[:, 0])
        back_mask = self.build_mask(graphs[:, 1])

        _x = x
        x = self.relu(self.fore_inter(x, mask=inter_mask))
        x = self.fore_norm(_x + x)

        _x = x
        x = self.relu(self.back(x, mask=back_mask))
        x = self.back_norm(x + _x)

        return x

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask


class EJGNet(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.1):
        super().__init__()
        self.gcn1 = GCNBlock(in_ch, out_ch, bias=False)
        self.norm1 = Normalization(out_ch, 'ln')
        self.drop = nn.Dropout(drop)

        self.gcn2 = GCNBlock(out_ch, out_ch, bias=False)
        self.norm2 = Normalization(out_ch, 'ln')

        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor, graph):
        _x = x
        x = self.relu(_x+self.gcn1(x, graph))
        x = self.norm1(x)

        _x = x
        x = self.relu(_x+self.gcn2(x, graph))
        x = self.norm2(x)

        return x

    def build_mask(self, adj):
        mask = Variable(adj.clone(), requires_grad=False)
        mask = mask.masked_fill_(mask <= 0, -float(1e22)).masked_fill_(mask >0, float(0))
        return mask






