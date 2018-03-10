import argparse
import pickle
import time

import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from modules import *
from util import *

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
np.random.seed(123)
if use_cuda:
    torch.cuda.manual_seed(123)


def cmp_dialog(d1, d2):
    if len(d1) < len(d2):
        return -1
    elif len(d2) > len(d1):
        return 1
    else:
        return 0


class DialogTurn():
    def __init__(self, item):
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
    def __init__(self, data_type, len=None):

        if data_type == 'train':
            _file = '/home/harshal/code/research/hred/data/MovieTriples_Dataset/Training.triples.pkl'
        elif data_type == 'valid':
            _file = '/home/harshal/code/research/hred/data/MovieTriples_Dataset/Validation.triples.pkl'
        elif data_type == 'test':
            _file = '/home/harshal/code/research/hred/data/MovieTriples_Dataset/Test.triples.pkl'
        self.utterance_data = []

        with open(_file, 'rb') as fp:
            data = pickle.load(fp)
            for d in data:
                self.utterance_data.append(DialogTurn(d))
        self.utterance_data.sort(cmp=cmp_dialog)
        if len:
            self.utterance_data = self.utterance_data[20000:20000+len]

    def __len__(self):
        return len(self.utterance_data)

    def __getitem__(self, idx):
        dialog = self.utterance_data[idx]
        return dialog, len(dialog.u1), len(dialog.u2), len(dialog.u3)


def custom_collate_fn(batch):
    bt_siz = len(batch)
    u1_batch, u2_batch, u3_batch = [], [], []
    u1_lens, u2_lens, u3_lens = np.zeros(bt_siz, dtype=int), np.zeros(bt_siz, dtype=int), np.zeros(bt_siz, dtype=int)

    l_u1, l_u2, l_u3 = 0, 0, 0
    for i, (d, cl_u1, cl_u2, cl_u3) in enumerate(batch):
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
    u1_batch = Variable(torch.ones(bt_siz, l_u1).long() * 10003)
    u2_batch = Variable(torch.ones(bt_siz, l_u2).long() * 10003)
    u3_batch = Variable(torch.ones(bt_siz, l_u3).long() * 10003)

    for i in range(bt_siz):
        seq1 = t1[i]
        u1_batch[i, :seq1.size(0)].data.copy_(seq1)
        seq2 = t2[i]
        u2_batch[i, :seq2.size(0)].data.copy_(seq2)
        seq3 = t3[i]
        u3_batch[i, :seq3.size(0)].data.copy_(seq3)

    sort1, sort2, sort3 = np.argsort(u1_lens*-1), np.argsort(u2_lens*-1), np.argsort(u3_lens*-1)

    return u1_batch[sort1, :], u1_lens[sort1].tolist(), u2_batch[sort2, :], u2_lens[sort2].tolist(), u3_batch[sort3, :], u3_lens[sort3].tolist()


def calc_valid_loss(base_enc, ses_enc, dec):
    base_enc.eval()
    ses_enc.eval()
    dec.eval()
    # dec.set_teacher_forcing(False)

    bt_siz, valid_dataset = 32, MovieTriples('valid', 32)
    valid_dataloader = DataLoader(valid_dataset, batch_size=bt_siz, shuffle=False, num_workers=2,
                                  collate_fn=custom_collate_fn)

    valid_loss = 0
    for i_batch, sample_batch in enumerate(valid_dataloader):
        u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], sample_batch[3], \
                                                sample_batch[4], sample_batch[5]

        o1, o2 = base_enc(u1, u1_lens), base_enc(u2, u2_lens)
        qu_seq = torch.cat((o1, o2), 1)
        # if we need to decode the intermediate queries we may need the hidden states
        final_session_o = ses_enc(qu_seq)
        loss = dec(final_session_o, u3, u3_lens)
        valid_loss += loss.data[0]

    base_enc.train()
    ses_enc.train()
    dec.train()
    # dec.set_teacher_forcing(True)

    return valid_loss/(1 + i_batch)


def train(options, base_enc, ses_enc, dec):
    base_enc.train()
    ses_enc.train()
    dec.train()

    all_params = list(base_enc.parameters()) + list(ses_enc.parameters()) + list(dec.parameters())
    # init parameters
    for name, param in base_enc.named_parameters():
        if name.startswith('rnn') and len(param.size()) >= 2:
            init.orthogonal(param)
        else:
            init.normal(param, 0, 0.01)

    bt_siz, train_dataset = 32, MovieTriples('train', 1000)
    print('training set size', len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=bt_siz, shuffle=False, num_workers=2,
                                  collate_fn=custom_collate_fn)
    optimizer = optim.Adam(all_params)
    for i in range(options.e):
        tr_loss = 0
        strt = time.time()
        for i_batch, sample_batch in enumerate(train_dataloader):
            # u1_batch, u1_lens, u2_batch, u2_lens, u3_batch, u3_lens
            u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], sample_batch[3], sample_batch[4], sample_batch[5]
            o1, o2 = base_enc(u1, u1_lens), base_enc(u2, u2_lens)
            qu_seq = torch.cat((o1, o2), 1)
            # if we need to decode the intermediate queries we may need the hidden states
            final_session_o = ses_enc(qu_seq)

            loss = dec(final_session_o, u3, u3_lens)
            tr_loss += loss.data[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i_batch % 100 == 0:
                print('done', i_batch)

        vl_loss = calc_valid_loss(base_enc, ses_enc, dec)
        print("Training loss {} Valid loss {} ".format(tr_loss/(1 + i_batch), vl_loss))
        print("epoch {} took {}".format(i+1, (time.time() - strt)/3600.0))
        if i % 2 == 0 or i == options.e-1:
            torch.save(base_enc.state_dict(), 'enc_mdl.pth')
            torch.save(ses_enc.state_dict(), 'ses_mdl.pth')
            torch.save(dec.state_dict(), 'dec_mdl.pth')
            torch.save(optimizer.state_dict(), 'opti_st.pth')


def main():
    # we use a common dict for all test, train and validation
    _dict_file = '/home/harshal/code/research/hred/data/MovieTriples_Dataset/Training.dict.pkl'
    with open(_dict_file, 'rb') as fp2:
        dict_data = pickle.load(fp2)
    # dictionary data is like ('</s>', 2, 588827, 785135)
    # so i believe that the first is the ids are assigned by frequency
    # thinking to use a counter collection out here maybe
    inv_dict = {}
    dict = {}
    for x in dict_data:
        tok, f, _, _ = x
        dict[tok] = f
        inv_dict[f] = tok

    parser = argparse.ArgumentParser(description='HRED parameter options')
    parser.add_argument('-e', dest='e', type=int, default=15, help='number of epochs')
    options = parser.parse_args()

    base_enc = BaseEncoder(10004, 300, 1000, 1, False)
    ses_enc = SessionEncoder(1500, 1000, 1, False)
    dec = Decoder(10004, 300, 1500, 1000, 1, False, True)
    if use_cuda:
        base_enc.cuda()
        ses_enc.cuda()
        dec.cuda()

    # train(options, base_enc, ses_enc, dec)
    bt_siz, test_dataset = 1, MovieTriples(data_type='train', len=5)
    test_dataloader = DataLoader(test_dataset, batch_size=bt_siz, shuffle=False, num_workers=2,
                                  collate_fn=custom_collate_fn)

    inference_beam(test_dataloader, base_enc, ses_enc, dec, inv_dict)


main()