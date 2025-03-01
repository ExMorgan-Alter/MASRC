import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from load_data.supervise_movienet import load_data
from model.SGCN import  FBNet

from loss import bce
from warm_up import warmup_decay_cosine
from metric import metric

torch.cuda.set_device(0)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train_epoch(
        trainload,
        model,
        opti,
        lr_sh,
        gpu=0
):
    model.train()
    progress = tqdm(trainload)
    for i, sample in enumerate(progress):
        sample_ftx, sample_btx, graphs, label = sample

        sample_ftx = sample_ftx.cuda(gpu)
        sample_btx = sample_btx.cuda(gpu)
        graphs = trans_graph(graphs, gpu)
        label = label.cuda(gpu)

        pred = model(sample_ftx, sample_btx, graphs)
        loss = bce(pred, label)

        opti.zero_grad()
        loss.backward()
        opti.step()

        lr_sh.step()
        progress.set_postfix(loss=f'{loss.item():.8f}')

    return 1


def test_epoch(
        testload,
        model,
        need,
        gpu=0,
):
    predlist = []
    labelist = []
    pathlist = []
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate((tqdm(testload))):
           paths, sample_ftx, sample_btx, graphs, label = sample

           sample_ftx = sample_ftx.cuda(gpu)
           sample_btx = sample_btx.cuda(gpu)
           graphs = trans_graph(graphs, gpu)

           pred = model(sample_ftx, sample_btx, graphs)

           predlist.append(pred.data.cpu().numpy())
           labelist.append(label.data.cpu().numpy())
           pathlist.append(paths)
    met, moviePL = metric(pathlist, predlist, labelist, needs=need)

    return met


def main(
        sample_path,
        split_path=None,
        seg_sz=20,
        batch=64,
        epoch=10,
        gpu=0,
        model_path=None,
        save_path=None,
):
    trainload = load_data(sample_path, split_path, seg_sz, batch)
    testload = load_data(sample_path, split_path, seg_sz, batch, mode='test')

    model = FBNet(2048, 1024, seg_sz=seg_sz, drop=0.2)
    model.cuda(gpu)

    if model_path is None:
        for para in model.parameters():
            para.requires_grad = True

    max_miou = 0
    max_map = 0
    opti = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4)
    iter_num = len(trainload)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        opti,
        warmup_decay_cosine(iter_num, iter_num * (epoch - 1))
    )

    for i in range(epoch):
        train_epoch(trainload, model, opti, lr_scheduler, gpu)
        met = test_epoch(testload, model, ['map', 'miou', 'f1'], gpu=gpu)

        # save model
        if save_path is not None:
            max_miou = met['mIoU']
            max_map = met['mAP']
            save_checkpoint({
                'state_dict': model.state_dict(),
                'miou': max_miou,
                'map': max_map,
                'f1':met['F1'],
                'optim': opti.state_dict()
            },
                save_path + '/epoch_{}.pth.tar'.format(i+1)
            )

        print('{} epoch: mAP:{:.3f}, mIoU:{:.3f}'.format(i+1, met['mAP'], met['mIoU']))
        print('{} epoch: F1:{:.3f}'.format(i + 1, met['F1']))

    return 1


def cal_params(model):
    params = model.state_dict()
    n_para = 0

    for k in params.keys():
        tensor = params[k]
        w = 1
        for i in range(len(tensor.shape)):
            w *= tensor.shape[i]
        n_para += w

    return n_para/1e6


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# ------------- Utilize -------------  #
def trans_graph(graph, gpu):
    graph = graph.to(torch.float)
    graph = graph.cuda(gpu)
    graph = Variable(graph, requires_grad=False)
    return graph


if __name__=='__main__':
    data_path = r'movienet_14_super'
    split_path = r'split318.json'
    save_path = None
    model_path = None

    movie_PL = main(data_path, split_path, seg_sz=14, batch=512, epoch=9, save_path=save_path, model_path=model_path)


