import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import AbsolutePositionalEncoder, Embedding
from model.detector import Detector_v2
from model.context import BackgroundNet, ForegroundNet, TextNet


class FBNet(nn.Module):
    def __init__(self, in_ch, proj_ch, seg_sz=20, drop=0.2):
        super().__init__()
        self.apos = AbsolutePositionalEncoder(256, seg_sz)
        self.ftx_emb = Embedding(in_ch, proj_ch-256)
        self.btx_emb = Embedding(in_ch, proj_ch-256)

        self.foreNet = ForegroundNet(proj_ch, proj_ch)
        self.backNet = BackgroundNet(proj_ch, proj_ch, drop=drop)

        self.detector = Detector_v2(seg_sz)

    def forward(self, ftx, btx, graphs):
        ftx = self.ftx_emb(ftx)
        btx = self.btx_emb(btx)
        apos = self.apos(ftx)

        ftx = torch.concat((ftx, apos), dim=-1)
        btx = torch.concat((btx, apos), dim=-1)

        ftx = self.foreNet(ftx, graphs[:, 0])
        btx = self.backNet(btx, graphs[:, 1:])

        pred = self.detector(ftx, btx)

        return pred