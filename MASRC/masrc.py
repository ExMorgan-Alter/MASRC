import torch
import torch.nn as nn
from MASRC.base import Embedding
from MASRC.detector import MCD
from MASRC.context import PCGNet, EJGNet


class FBNet(nn.Module):
    def __init__(self, in_ch, proj_ch, seg_sz=14, drop=0.2):
        super().__init__()
        self.ftx_emb = Embedding(in_ch, proj_ch)
        self.btx_emb = Embedding(in_ch, proj_ch)
        self.foreNet = EJGNet(proj_ch, proj_ch)
        self.backNet = PCGNet(proj_ch, proj_ch, drop=drop)
        self.detector = MCD(seg_sz)

    def forward(self, ftx, btx, graphs):
        """
        :param ftx: entity features, (B, T, D), B:batch, T:sliding window scale, D:feature dimension
        :param btx: place features, (B, T, D), same as above
        :param graphs: (B, 3, T, T), 0: entity jumping graph, 1-2: place continuity graph
        :return pred: (B,)
        :return:
        """
        ftx = self.ftx_emb(ftx)
        btx = self.btx_emb(btx)
        # MASR
        ftx = self.foreNet(ftx, graphs[:, 0])
        btx = self.backNet(btx, graphs[:, [1, 2]])
        # MCD
        pred = self.detector(ftx, btx)
        return pred


