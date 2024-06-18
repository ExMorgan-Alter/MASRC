import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import ProcessSimilar


class MCD(nn.Module):
    def __init__(self, seg=20):
        super().__init__()
        self.half = seg//2
        self.T = seg
        self.convnet = ProcessSimilar(1)
        self.cls = MLP(seg//4, 128, 1, 0.2)
        self.act = nn.Sigmoid()

    def forward(self, entity, place):
        """
        :param entity: (B, T, D)
        :param place: same as above
        :return:
        """
        T = entity.shape[1]

        # cosine similarity for single shot
        entity = F.normalize(entity, dim=-1)
        place = F.normalize(place, dim=-1)
        context = torch.concat((entity, place), dim=-1)
        prec, postc = torch.split(context, T // 2, dim=1)
        sim = 0.5*self.cosin_matrix(prec, postc)

        sim = self.convnet(sim)
        sim = torch.mean(sim, dim=-1)
        sim = self.cls(sim)
        sim = sim.squeeze()
        sim = self.act(sim)

        return sim

    def chamfer_similarity(self, mat, max_axis=-1, mean_axis=-1):
        """
        :param mat: (batch, T//2, T//2)
        :param max_axis:
        :param mean_axis:
        :return:
        """
        mat = torch.amax(mat, dim=max_axis)
        mat = torch.mean(mat, dim=mean_axis)

        return mat

    def cosin_matrix(self, feat1, feat2):
        """
        :param feat1: (batch, T//2, dim)
        :param feat2:
        :return:
        """
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)
        sim = torch.matmul(feat1, feat2.transpose(2, 1))

        return sim

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class MLP(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, drop=0.5):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(in_ch, hid_ch)
        self.linear_2 = nn.Linear(hid_ch, out_ch)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

        self.initialize_weight(self.linear_1)
        self.initialize_weight(self.linear_2)

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.drop(x)
        x = self.linear_2(x)

        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)





