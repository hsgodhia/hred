import torch
import copy
import pickle
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np

use_cuda = torch.cuda.is_available()


def custom_collate_fn(batch):
    # input is a list of dialogturn objects
    bt_siz = len(batch)
    # sequence length only affects the memory requirement, otherwise longer is better
    pad_idx, max_seq_len = 10003, 160

    u1_batch, u2_batch, u3_batch = [], [], []
    u1_lens, u2_lens, u3_lens = np.zeros(bt_siz, dtype=int), np.zeros(bt_siz, dtype=int), np.zeros(bt_siz, dtype=int)

    # these store the max sequence lengths for the batch
    l_u1, l_u2, l_u3 = 0, 0, 0
    for i, (d, cl_u1, cl_u2, cl_u3) in enumerate(batch):
        cl_u1 = min(cl_u1, max_seq_len)
        cl_u2 = min(cl_u2, max_seq_len)
        cl_u3 = min(cl_u3, max_seq_len)

        if cl_u1 > l_u1:
            l_u1 = cl_u1
        u1_batch.append(torch.LongTensor(d.u1))
        u1_lens[i] = cl_u1

        if cl_u2 > l_u2:
            l_u2 = cl_u2
        u2_batch.append(torch.LongTensor(d.u2))
        u2_lens[i] = cl_u2

        if cl_u3 > l_u3:
            l_u3 = cl_u3
        u3_batch.append(torch.LongTensor(d.u3))
        u3_lens[i] = cl_u3

    t1, t2, t3 = u1_batch, u2_batch, u3_batch

    u1_batch = Variable(torch.ones(bt_siz, l_u1).long() * pad_idx)
    u2_batch = Variable(torch.ones(bt_siz, l_u2).long() * pad_idx)
    u3_batch = Variable(torch.ones(bt_siz, l_u3).long() * pad_idx)
    end_tok = torch.LongTensor([2])

    for i in range(bt_siz):
        seq1, cur1_l = t1[i], t1[i].size(0)
        if cur1_l <= l_u1:
            u1_batch[i, :cur1_l].data.copy_(seq1[:cur1_l])
        else:
            u1_batch[i, :].data.copy_(torch.cat((seq1[:l_u1-1], end_tok), 0))

        seq2, cur2_l = t2[i], t2[i].size(0)
        if cur2_l <= l_u2:
            u2_batch[i, :cur2_l].data.copy_(seq2[:cur2_l])
        else:
            u2_batch[i, :].data.copy_(torch.cat((seq2[:l_u2-1], end_tok), 0))

        seq3, cur3_l = t3[i], t3[i].size(0)
        if cur3_l <= l_u3:
            u3_batch[i, :cur3_l].data.copy_(seq3[:cur3_l])
        else:
            u3_batch[i, :].data.copy_(torch.cat((seq3[:l_u3-1], end_tok), 0))

    sort1, sort2, sort3 = np.argsort(u1_lens*-1), np.argsort(u2_lens*-1), np.argsort(u3_lens*-1)
    # cant call use_cuda here because this function block is used in threading calls

    return u1_batch[sort1, :], u1_lens[sort1], u2_batch[sort2, :], u2_lens[sort2], u3_batch[sort3, :], u3_lens[sort3]


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


def cmp_dialog(d1, d2):
    if len(d1) < len(d2):
        return -1
    elif len(d1) > len(d2):
        return 1
    else:
        return 0


class DialogTurn:
    def __init__(self, item):
        self.u1, self.u2, self.u3 = [], [], []
        cur_list, i = [], 0
        for d in item:
            cur_list.append(d)
            if d == 2:
                if i == 0:
                    self.u1 = copy.copy(cur_list)
                    cur_list[:] = []
                elif i == 1:
                    self.u2 = copy.copy(cur_list)
                    cur_list[:] = []
                else:
                    self.u3 = copy.copy(cur_list)
                    cur_list[:] = []
                i += 1

    def __len__(self):
        return len(self.u1) + len(self.u2) + len(self.u3)

    def __repr__(self):
        return str(self.u1 + self.u2 + self.u3)


class MovieTriples(Dataset):
    def __init__(self, data_type, length=None):
        if data_type == 'train':
            _file = '/home/harshals/hed-dlg/Data/MovieTriples/Training.triples.pkl'
        elif data_type == 'valid':
            _file = '/home/harshals/hed-dlg/Data/MovieTriples/Validation.triples.pkl'
        elif data_type == 'test':
            _file = '/home/harshals/hed-dlg/Data/MovieTriples/Test.triples.pkl'
        self.utterance_data = []

        with open(_file, 'rb') as fp:
            data = pickle.load(fp)
            for d in data:
                self.utterance_data.append(DialogTurn(d))
        # it helps in optimization that the batch be diverse, definitely helps!
        # self.utterance_data.sort(key=cmp_to_key(cmp_dialog))
        if length:
            self.utterance_data = self.utterance_data[2000:2000 + length]

    def __len__(self):
        return len(self.utterance_data)

    def __getitem__(self, idx):
        dialog = self.utterance_data[idx]
        return dialog, len(dialog.u1), len(dialog.u2), len(dialog.u3)


def tensor_to_sent(x, inv_dict, greedy=False):
    sents = []
    inv_dict[10003] = '<pad>'
    for li in x:
        if not greedy:
            scr = li[1]
            seq = li[0]
        else:
            scr = 0
            seq = li
        sent = []
        for i in seq:
            sent.append(inv_dict[i])
            if i == 2:
                break
        sents.append((" ".join(sent), scr))
    return sents
