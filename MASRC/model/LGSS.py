import torch, vit_pytorch
import torch.nn as nn
import torch.nn.functional as F
from model.base import Embedding, AbsolutePositionalEncoder
from model.detector import Detector, SCRL, RecurrentDetect


class Cos(nn.Module):
    def __init__(self, channel):
        super(Cos, self).__init__()
        self.shot_num = 4
        self.channel = channel
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(self.shot_num//2, 1))

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        part1, part2 = torch.split(x, [self.shot_num//2]*2, dim=2)
        # batch_size*seq_len, 1, [self.shot_num//2], feat_dim
        part1 = self.conv1(part1).squeeze()
        part2 = self.conv1(part2).squeeze()
        x = F.cosine_similarity(part1, part2, dim=2)  # batch_size,channel
        return x


class BNet(nn.Module):
    def __init__(self, channel, shot_num):
        super(BNet, self).__init__()
        self.shot_num = shot_num
        self.channel = channel
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(self.shot_num, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))
        self.cos = Cos(channel)

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        context = x.reshape(x.shape[0]*x.shape[1], 1, -1, x.shape[-1])
        context = self.conv1(context)  # batch_size*seq_len,512,1,feat_dim
        context = self.max3d(context)  # batch_size*seq_len,1,1,feat_dim
        context = context.squeeze()
        sim = self.cos(x)
        bound = torch.cat((context, sim), dim=1)
        return bound


class LGSS(nn.Module):
    def __init__(self, in_dim=2048, proj_ch=1024, shot_num=4, seq_len=20):
        super().__init__()
        self.seq_len = seq_len
        self.shot_num = shot_num
        self.embedding = Embedding(in_dim, proj_ch)
        self.bnet = BNet(proj_ch, shot_num)

        self.lstm = nn.LSTM(input_size=proj_ch*2,
                            hidden_size=proj_ch//2,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

        self.fc1 = nn.Linear(proj_ch, 100)
        self.fc2 = nn.Linear(100, 1)
        # self.detect = Detector(proj_ch*2, seg=seq_len)

        self.ctxinx = self._tneigh_mask()[None]

    def forward(self, x):
        x = self.embedding(x)

        x = torch.einsum('btnd, btdc->btnc', (self.ctxinx, x.unsqueeze(dim=1)))
        x = self.bnet(x)
        x = x.view(-1, self.seq_len, x.shape[-1])

        # torch.Size([128, seq_len, 3*channel])
        self.lstm.flatten_parameters()
        out, (_, _) = self.lstm(x, None)

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = F.sigmoid(out[:, self.seq_len//2 - 1]).squeeze()
        # out = self.detect(x)
        return out

    def _tneigh_mask(self):
        T = self.seq_len
        tnei = self.shot_num//2
        mask = torch.zeros(T, self.shot_num, T, requires_grad=False, device="cuda:0")
        neigh_inx = {}
        for i in range(T):
            if i - tnei < 0:
                inx = [j for j in range(self.shot_num)]
            if i + 1 + tnei > T:
                inx = [j for j in range(T - 2 * tnei, T)]
            else:
                inx = [j for j in range(i - tnei+1, i + tnei + 1)]
            neigh_inx.update({i: inx})

        for i in neigh_inx.keys():
            for nn, j in enumerate(neigh_inx[i]):
                mask[i, nn, j] = 1
        return mask


class Transformer(nn.Module):
    def __init__(self, in_ch, proj_ch, seg_sz=20, drop=0.2):
        super().__init__()
        self.PE = AbsolutePositionalEncoder(256, seg_sz)
        self.embed = Embedding(in_ch, proj_ch)

        self.tf = vit_pytorch.vit.Transformer(
            dim=proj_ch,
            depth=2,
            heads=8,
            dim_head=proj_ch // 8,
            mlp_dim=int(proj_ch*2),
            dropout=drop,
        )

        # self.detect = Detector(in_ch*2)
        self.detect = RecurrentDetect(in_ch)

    def forward(self, x1, x2):
        x = torch.concat((x1, x2), dim=-1)
        # x = self.embed(x)
        # pos_emb = self.PE(x)
        # x = torch.concat((x, pos_emb), dim=-1)

        # ctx = self.tf(x)
        pred = self.detect(x)

        return pred

