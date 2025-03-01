import numpy as np
import pickle as pkl
import random
import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, samplelist:list):
        self.samplist = samplelist
        random.shuffle(self.samplist)

    def __getitem__(self, ind):
        return self.samplist[ind]

    def __len__(self):
        return len(self.samplist)

    def _read_pkl(self, path):
        with open(path, 'rb') as f:
            data = pkl.load(f)
        return data

    def _build_graph(self, links, seg_sz=20, mode='place', is_diag=True, is_norm=True):
        if mode == 'place':
            simgh = np.zeros((2, seg_sz, seg_sz))
            for star, link in links.items():
                if len(link) != 0:
                    simgh[0, star, link] = 1
                    simgh[1, link, star] = 1

        elif mode=='fore':
            simgh = np.zeros((1, seg_sz, seg_sz))
            for i, link in enumerate(links):
                if len(link) != 0:
                    simgh[0, i, link] = 1
        else:
            simgh = np.ones((2, seg_sz, seg_sz))
        if is_diag:
            simgh = np.eye(seg_sz)[None] + simgh
        if is_norm:
            simgh = self._norm_graph(simgh)
        return simgh

    def _norm_graph(self, adj):
        """
        :param adj: (X, T, T)
        :return:
        """
        adj_sum = np.sum(adj, axis=-1)
        adj_sum = np.sqrt(1/adj_sum)
        adj_norm = adj_sum[:, :, None] * adj_sum[:, None, :] * adj
        return adj_norm
