import numpy as np
import random
import torch
import os
import json as js
from load_data.BaseDataset import BaseDataset


class MovienetDataset(BaseDataset):
    def __init__(self, samplelist:list, seg_sz=20, mode='train'):
        super(MovienetDataset, self).__init__(samplelist)
        self.mode = mode
        self.seg_sz = seg_sz

    def __getitem__(self, ind):
        data = self._read_pkl(self.samplist[ind])
        sample_ftx = data['fore']
        sample_btx = data['back']
        if 'label' in data.keys():
            label = data['label']

        if label != 1 and label !=0:
            label=1

        alink = data['graph']
        hop = []
        hop.append(self._build_graph(alink[0], seg_sz=self.seg_sz, mode='fore'))
        hop.append(self._build_graph(alink[1], seg_sz=self.seg_sz, mode='place'))
        hop = np.concatenate(hop, axis=0)

        sample_ftx = torch.from_numpy(sample_ftx)
        sample_btx = torch.from_numpy(sample_btx)
        sample_ftx = sample_ftx.to(torch.float)
        sample_btx = sample_btx.to(torch.float)
        label = torch.from_numpy(np.array(label))
        label = label.to(torch.float)

        if self.mode == 'train':
            return sample_ftx, sample_btx, hop, label
        else:
            return self.samplist[ind], sample_ftx, sample_btx,  hop, label


def load_data(data_path, split_path, seg_sz, batch, mode='train'):
    with open(split_path, 'r') as f:
        data = js.load(f)
        random.shuffle(data['test'])
        # trainSet = data['train']  + random.choices(data['test'], k=int(len(data['test'])*0.4))
        trainSet = data['train'] + data['val']
        testSet = data['test']

    samplelist = os.listdir(data_path)
    if mode == 'train':
        trainlist = [data_path+'/'+sample for sample in samplelist if sample.split('_')[0] in trainSet]
        trainDataset = MovienetDataset(trainlist, seg_sz=seg_sz, mode=mode)
        dataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch,
                                                  shuffle=True, drop_last=True, num_workers=4)

    if mode == 'test':
        testlist = [data_path+'/'+sample for sample in samplelist if sample.split('_')[0] in testSet]
        testDataset = MovienetDataset(testlist, seg_sz=seg_sz, mode=mode)
        dataLoader = torch.utils.data.DataLoader(testDataset,  batch_size=batch,
                                                 shuffle=False, drop_last=False, num_workers=4)

    return dataLoader




