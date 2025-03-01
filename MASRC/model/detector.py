import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base import ProcessSimilar


class Detector_v2(nn.Module):
    def __init__(self, seg=20):
        super().__init__()
        self.half = seg//2
        self.T = seg

        self.vconv = ProcessSimilar(1)
        self.cls = MLP((seg//4), 128, 1, 0.2)
        self.act = nn.Sigmoid()

    def forward(self, cast, place):
        """
        :param cast: (batch, T, dim)
        :param place: same
        :return:
        """
        batch, T, _ = cast.shape

        # cosine similarity for single shot
        precast, postcast = torch.split(cast, T // 2, dim=1)
        preplace, postplace = torch.split(place, T // 2, dim=1)

        sim_cast = self.cosin_matrix(precast, postcast)
        sim_place = self.cosin_matrix(preplace, postplace)
        sim_v = (sim_cast+sim_place)/2

        sim = self.vconv(sim_v).mean(dim=-1)
        sim = self.cls(sim)
        sim = sim.squeeze()
        sim = self.act(sim)

        return sim

    def pad_cosine(self, feat:torch.Tensor):
        """
        :param feat: (batch, T, dim)
        :return:
        """
        # mapping: y = half-1 + x
        left_pad = torch.clone(feat[:, :self.half-1])
        right_pad = torch.clone(feat[:, self.half:self.T])
        feat_pad = torch.concat((left_pad, feat, right_pad), dim=1)

        feat_pad = F.normalize(feat_pad, dim=-1)

        sims = self.cosin_matrix(feat_pad[:, :self.half], feat_pad[:, self.half:2*self.half])
        sims = sims[:, None]

        for i in range(self.half, 3*self.half-1):
            sim = self.cosin_matrix(feat_pad[:, i-self.half+1:i+1], feat_pad[:, i+1:i+1+self.half])
            sims = torch.concat((sims, sim[:, None]), dim=1)

        return sims

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

    def cosin_matrix(self, feat1, feat2, weight=None):
        """
        :param feat1: (batch, T//2, dim)
        :param feat2:
        :return:
        """
        # batch, T, _ = feat1.shape
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)

        sim = torch.matmul(feat1, feat2.transpose(2, 1))

        return sim

    def analayse(self, cast, place, scale, ratio):
        """
        Analyze outputs of different stages
        :param feat:
        :return:
        """
        batch, T, _ = cast.shape
        # Face Align
        precast, postcast = torch.split(cast, T // 2, dim=1)
        wo_cast = torch.matmul(precast, postcast.transpose(2, 1))

        crosscast = self.castAlign(precast, postcast, postcast)
        postcast = self.castNorm(crosscast + postcast)
        postcast = self.castLinear(postcast)

        # Place Align
        preplace, postplace = torch.split(place, T // 2, dim=1)
        wo_place = self.cosin_matrix(preplace, postplace)

        crossplace = self.placeAlign(preplace, postplace, postplace)
        postplace = self.placeNorm(crossplace + postplace)
        postplace = self.placeLinear(postplace)

        # cosine similarity
        sim_cast = self.cosin_matrix(precast, postcast)
        sim_place = self.cosin_matrix(preplace, postplace)
        sim_add = 0.5*sim_cast + 0.5*sim_place
        # mean-max filter
        sim_conv = self.convnet(sim_add)
        sim = torch.mean(sim_conv, dim=-1)
        pred = self.cls(sim)
        pred = pred.squeeze()
        pred = self.act(pred)
        return [wo_cast, wo_place], [sim_cast, sim_place], sim_conv, pred

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


class Shotcol(nn.Module):
    def __init__(self, in_ch=2560, hid_ch=2560, out_ch=1, win_size=3):
        super(Shotcol, self).__init__()
        self.linear_1 = nn.Linear(in_ch*win_size*2, hid_ch)
        self.linear_2 = nn.Linear(hid_ch, hid_ch//2)
        self.linear_3 = nn.Linear(hid_ch//2, out_ch)
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self.act = nn.Sigmoid()

        self.win = win_size

        self.initialize_weight(self.linear_1)
        self.initialize_weight(self.linear_2)
        self.initialize_weight(self.linear_3)

    def forward(self, x):
        batch = x.shape[0]
        half = x.shape[1]//2
        _, x, _ = torch.split(x, [half-self.win, 2*self.win, half-self.win], dim=1)
        x = x.view(batch, -1)
        x = self.relu(self.linear_1(x))
        x = self.drop(x)
        x = self.relu(self.linear_2(x))
        x = self.drop(x)
        x = self.act(self.linear_3(x))

        return x.squeeze()

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class TCD(nn.Module):
    def __init__(self, T, feat_dim):
        super(TCD, self).__init__()
        self.left_fc = nn.Conv1d(T//2, 1, 1, bias=False)
        self.right_fc = nn.Conv1d(T//2, 1, 1, bias=False)
        self.norm = nn.BatchNorm1d(feat_dim*2)
        self.cls = nn.Sequential(
            nn.Conv1d(feat_dim*2, feat_dim*4, 1),
            nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Conv1d(feat_dim*4, 1, 1)
        )
        self.act = nn.Sigmoid()

    def forward(self, feat):
        """
        :param feat: (batch, T, dim)
        :return:
        """
        batch, T, _ = feat.shape
        left_feat, right_feat = torch.split(feat, T//2, dim=1)

        left_feat = self.left_fc(left_feat)
        right_feat = self.right_fc(right_feat)

        feat_cat = torch.concat((left_feat, right_feat), dim=-1)
        feat_cat = self.norm(feat_cat.permute(dims=(0,2,1)).contiguous())
        score = self.cls(feat_cat)
        score = torch.squeeze(score)
        score = self.act(score)

        return score


class SCRL(nn.Module):
    def __init__(self, input_feature_dim=2560, fc_dim=1280, hidden_size=640,
                 input_drop_rate=0.2, lstm_drop_rate=0.2, fc_drop_rate=0.2, use_bn=True):
        super(SCRL, self).__init__()
        input_size = input_feature_dim
        output_size = fc_dim
        self.embed_sizes = input_feature_dim
        self.embed_fc = nn.Linear(input_size, output_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=lstm_drop_rate,
            bidirectional=True
        )
        # The probability is set to 0 by default
        self.input_dropout = nn.Dropout(p=input_drop_rate)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate)
        self.fc1 = nn.Linear(self.hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.act = nn.Sigmoid()
        self.use_bn = use_bn

        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(output_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x:torch.Tensor):
        """
        :param x: (batch, T, dim)
        :return:
        """
        T = x.shape[1]
        cind = T//2 - 1
        x = self.input_dropout(x)
        x = self.embed_fc(x)

        if self.use_bn:
            seq_len, C = x.shape[1:3]
            x = x.view(-1, C)
            x = self.bn1(x)
            x = x.view(-1, seq_len, C)
        x = self.fc_dropout(x)
        out, (_, _) = self.lstm(x, None)
        out = self.fc1(out)
        if self.use_bn:
            seq_len, C = out.shape[1:3]
            out = out.view(-1, C)
            out = self.bn2(out)
            out = out.view(-1, seq_len, C)
        out = self.fc_dropout(out)
        out = F.relu(out)
        out = self.act(self.fc2(out))
        return out[:, cind].squeeze()



class Bassl(nn.Module):
    def __init__(self, hid_dim=512, out_dim=1, bn=False):
        super().__init__()
        self.out_dim = out_dim
        self.fc1 = nn.LazyLinear(hid_dim)
        self.bn = nn.LazyBatchNorm1d() if bn else None
        self.relu = nn.ReLU()
        self.fc2 = nn.LazyLinear(out_dim)

    def forward(self, x):
        T = x.shape[1]
        shape = x.shape
        cind = shape[1]//2 -1
        x = x.reshape(-1, shape[-1])
        x = self.fc1(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.reshape(*shape[:-1], self.out_dim)
        return F.sigmoid(x[:, T//2].squeeze())


class ICCV2023(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.P = nn.parameter.Parameter(torch.randn(2, in_dim))
        self.q = nn.Linear(in_dim, in_dim, bias=False)
        self.k = nn.Linear(in_dim, in_dim, bias=False)

        self.mlp = MLP(in_dim*2, in_dim, 1)

    def forward(self, x:torch.Tensor):
        B = x.shape[0]
        # p_hat: (2, C)
        p_hat = self.P[None] * x.mean(dim=1, keepdim=True)
        p_hat = self.q(p_hat)
        x_k = self.k(x)
        # A: (2, T)
        p_hat = F.normalize(p_hat, dim=-1)
        x_k = F.normalize(x_k, dim=-1)
        A = torch.matmul(p_hat, x_k.transpose(-1, -2))
        # x_scn: (2, C)
        x_scn = torch.matmul(A, x)
        x_scn = x_scn.view(B, -1)

        pred = F.sigmoid(self.mlp(x_scn)).squeeze()
        return pred






