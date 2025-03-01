import numpy as np
import pickle as pkl
import os
from tqdm import tqdm
import ruptures as rup
import random

# ----------- Pseudo Labels ----------- #

def MDTW(feats:np.ndarray):
    """
    :param feats: (T, D)
    :return:
    """
    T = feats.shape[0]
    sim_bank = np.zeros(T)
    sim_bank[0] = -100
    sim_bank[-1] = -100
    simat = similarity(feats, feats)

    for i in range(1, T-1):
        sim_pre = simat[0, 1:i+1]
        sim_aft = simat[-1, i+1:-1]

        sim = sim_pre.sum() + sim_aft.sum()
        sim_bank[i] = sim

    pse_label = np.argmax(sim_bank)
    return pse_label


def CPD(fore:dict, back:dict):
    fore = dict2ndarry(fore)
    back = dict2ndarry(back)
    T = fore.shape[0]
    feat = np.concatenate((fore, back), axis=-1)
    feat = normalized(feat)

    algo = rup.BottomUp(model='rbf', min_size=1, jump=1).fit(feat)
    cps = algo.predict(n_bkps=int(T*0.09))

    cand = np.array(cps[:-1]) - 1

    return cand


def dict2ndarry(feat:dict, dim=2048):
    nshot = len(list(feat.keys()))
    feat_nd = np.zeros((nshot, dim))
    for i in range(nshot):
        if dim ==1:
            feat_nd[i] = feat[i]
        else:
            feat_nd[i] = feat[f'{i:04d}']
    return feat_nd


# ----------- Basic Funtion ----------- #
def read_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data


def write_pkl(path: str, data: dict):
    with open(path, 'wb') as f:
        pkl.dump(data, f)
    return 1


def acc_hop(mask, graph):
    is_1 = mask.sum()
    g_mask = np.sum((graph > 0) * mask)
    acc = g_mask/is_1

    return acc


def area_hop(mask, graph):
    mask = 1- mask
    g_mask = np.sum((graph > 0) * mask)
    acc = g_mask/mask.sum()
    return acc


def read_label(path):
    """
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        labelDict = {}
        while 1:
            line = f.readline()
            if not line:
                break
            line = line.split('\n')[0]
            shot_id, label = line.split(' ')
            labelDict.update({int(shot_id): int(label)})
    return labelDict

def build_graph(links:dict, seg_sz):
    simgh = np.zeros((seg_sz, seg_sz))
    for star, link in links.items():
        if len(link[0]) != 0:
            simgh[star, link[0]] = 1
            simgh[link[0], star] = 1
        simgh[star, star] = 1

    return simgh


def Qfunc(sim, cps):
    t = sim.shape[0]
    SS = np.sum(sim)
    Q = 0

    for i in range(cps.shape[0]-1):
        if i == 0:
            start = 0
            end = cps[0]-1
        elif i == cps.shape[0] - 2:
            start = cps[i-1]
            end = t-1
        else:
            start = cps[i-1]
            end = cps[i]-1
        yy = np.sum(sim[start:end+1, start:end+1])
        yS = np.sum(sim[start:end+1])
        Q += (yy/SS) - (yS/SS)**2

    return Q


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def similarity(feat1, feat2, tmask=None):
    feat1 = normalized(feat1)
    feat2 = normalized(feat2)

    mat = np.matmul(feat1, feat2.transpose(1, 0))
    mask = np.eye(feat1.shape[0])
    mat = mat - mask

    if tmask is not None:
        mat = mat * tmask

    return mat


def sampleDtx(fore:dict, back:dict, center_id, seg_sz, fore_dim=2048, back_dim=2048):
    """
    Collecting shots centred in centre_id in the window at scale of seg_sz
    :param fore:
    :param back:
    :param center_id:
    :param seg_sz: is a even number
    :return:
        ftx: (seg_sz, ??)
        btx: same
    """
    max_id = len(fore.keys())
    half = seg_sz // 2
    ctx_id = np.arange(center_id - half + 1, center_id + half + 1)
    ctx_id = np.clip(ctx_id, 0, max_id - 1)

    if fore_dim == 2048:
        ftx = np.zeros((seg_sz, fore_dim))
    else:
        ftx = np.zeros((seg_sz, 5, fore_dim//5))
    btx = np.zeros((seg_sz, back_dim))
    for i, shot_id in enumerate(ctx_id):
        ftx[i] = fore[f'{shot_id:04d}'][None]
        btx[i] = back[f'{shot_id:04d}'][None]
    if fore_dim != 2048:
        ftx = np.reshape(ftx, (seg_sz, fore_dim))
    return ftx, btx


def find_extreme(seq):
    """
    find maximum in shot scale sequence
    :param deq:
    :return:
        center_index: list
    """
    center_index = [0]
    for i in range(1, seq.shape[0]-1):
        if seq[i-1]<seq[i] and seq[i]>seq[i+1]:
            center_index.append(i)
    if 2*seq[-1] > (seq[-2]+seq[-3]):
        center_index.append(seq.shape[0]-1)

    if len(center_index) == 0:
        center_index.append(0)
        center_index.append(seq.shape[0]//2 - 1)
        center_index.append(seq.shape[0]-1)

    return np.array(center_index)


def gen_pair(T, exnode):
    """
    Generate a diction, key is time index, value is the interval that the time index belongs to
    :param T:  the length of input
    :param exnode: extra scales
    :return:
    """
    exnode = exnode.copy()
    pairs = {}
    interval = []
    max_node = max(exnode)

    # if the end shot is the extra shot
    if max_node != T-1:
        exnode.append(T-1)
    if 0 not in exnode:
        exnode = [0]+exnode

    # generate extra segments
    for i in range(len(exnode)-1):
        interval.append([exnode[i], exnode[i+1]])

    for i in range(T):
        for intv in interval:
            st, ed = intv
            if i not in exnode and st < i < ed:
                pairs.update({i: [st, ed]})
                break
            if i == ed:
                pairs.update({i: [st]})
                continue
            if i == st:
                if i != 0:
                    pairs.update({i:[ed]})
                else:
                    pairs.update({i:[ed]})
                break

    return pairs


def gen_labelName(path):
    files = os.listdir(path)
    names = [file.split('.')[0] for file in files]

    return names


# ----------- Foreground Links ----------- #
def topKNN(feat, top=3):
    """
    :param feat: (T, dim)
    :param top:
    :return:
        simat: (T, top)
    """
    # KNN
    simat = similarity(feat, feat)
    simat = np.argsort(simat, axis=-1)[:, -1:-(top+1):-1]

    return simat


# ----------- Place Links ----------- #

def threshold_link(simat, thre=0.3):
    is_thre = simat > thre
    # simat_f = simat * is_thre
    return is_thre*1


def best_cps(feat, step=5):
    """
    :param feat:
    :param step:
    :return:
        bsd: list
    """
    simat = similarity(feat, feat)
    mean_th = np.mean(simat)
    std_th = np.std(simat)
    interval = std_th*2/step

    max_q = -100
    bst = []
    for thre in np.arange(mean_th-3*std_th, mean_th+std_th, interval):
        simatf = threshold_link(feat, thre)
        cps = find_extreme(simatf.sum(-1))
        q_score = Qfunc(simat, cps)
        if q_score>max_q:
            bst = cps
            max_q = q_score
    return list(bst)


def fast_segment(feat, is_center=False):
    half = feat.shape[0] // 2
    if is_center:
        c_lft = np.argmax(similarity(feat[:half], feat[:half]).sum(-1))
        c_rgt = half + np.argmax(similarity(feat[half:], feat[half:]).sum(-1))
        bound = c_lft + MDTW(feat[c_lft:c_rgt + 1])
    else:
        c = np.argmax(similarity(feat, feat).sum(-1))
        if c > half:
            bound = MDTW(feat[:c + 1])
        else:
            bound = MDTW(feat[c:])

    # gen pairs
    peaks = [0, bound, feat.shape[0]]
    clust = {}
    for i in range(len(peaks)-1):
        seg = feat[peaks[i]:peaks[i+1]]
        cc = peaks[i]+np.argmax(similarity(seg, seg).sum(-1))
        clust[cc] = list(range(peaks[i], peaks[i+1]))

    return clust


def cps(feat, ratio=-1):
    """
    :param feat:
    :param ratio:
    :return:
        bsd: list
    """
    simat = similarity(feat, feat)
    mean_th = np.mean(simat)
    std_th = np.std(simat)
    simatf = threshold_link(feat, mean_th + ratio*std_th)
    bst = find_extreme(simatf.sum(-1))
    return list(bst)


def segmentKNN(feat, cps_mode='self'):
    """
    :param feat:
    :return:
        clustee: diction, key is star shot index, value consists of two part: node index that is attributed to
        the star shot; node index that climax is belonged to.
    """
    T = feat.shape[0]
    if cps_mode == 'self':
        cind = best_cps(feat)
    else:
        n_cps = random.sample([2,3,4,5], 1)[0]
        cind = random.sample(range(1, T), n_cps)

    pair = gen_pair(T, cind)
    dist = similarity(feat, feat)

    # clust is a list saving the star shot index that each shot is attributed to
    clust = []
    for i in range(T):
        intv = pair[i]
        if len(intv) == 1:
            clust.append(intv[0])
        else:
            if dist[i, intv[0]] > dist[i, intv[1]]:
                clust.append(intv[0])
            else:
                clust.append(intv[1])

    clustee = {}
    for i, c in enumerate(clust):
        # i^th shot is non-star shot
        if i not in cind:
            if c not in clustee.keys():
                clustee.update({c: [[i], []]})
            else:
                clustee[c][0].append(i)
        else:
            if i not in clustee.keys():
                clustee.update({i: [[], [c]]})
            else:
                clustee[i][1].append(c)

    return clustee

# ----------- Generate FPgraph ----------- #


def gen_movienet(fore_path, back_path, lb_path, seg_sz, topk=4, have_graph=False, graph_path=None, save_path=None):
    fore_feat = read_pkl(fore_path)
    back_feat = read_pkl(back_path)
    mnames = gen_labelName(lb_path)
    if have_graph:
        graphs = read_pkl(graph_path)

    for name in tqdm(mnames):
        if name in fore_feat.keys():
            fore_mv = fore_feat[name]
            back_mv = back_feat[name]
            label_m = read_label(lb_path + '/' + name + '.txt')
            n_shot = len(label_m.keys())

            for c_id in range(n_shot):
                ftx, btx = sampleDtx(fore_mv, back_mv, c_id, seg_sz, fore_dim=2048, back_dim=2048)
                if not have_graph:
                    plink = fast_segment(btx, True)
                    flink = topKNN(ftx, topk)
                else:
                    plink = graphs[name][c_id]['place']
                    flink = graphs[name][c_id]['entity']

                save_ctx = save_path + '/{}_shot{}.pkl'.format(name, c_id)
                sample = {'fore': ftx, 'back':btx, 'graph':[flink, plink], 'label':label_m[c_id]}
                write_pkl(save_ctx, sample)

    return 1


if __name__=='__main__':
    fore_path = r'ImageNet_shot.pkl'
    back_path = r'Places_shot.pkl'
    label_path = r'label318'
    save_path = r'movienet_14_super'
    graph_path = r''

    # # if you have not downloaded graphs from quark
    # gen_movienet(fore_path, back_path, label_path, seg_sz=14, topk=4, save_path=save_path)
    #
    # # if you have downloaded graphs from quark
    # gen_movienet(fore_path, back_path, label_path, seg_sz=14, topk=4, have_graph=True, graph_path=graph_path, save_path=save_path)







