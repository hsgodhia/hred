import argparse
import time
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader

from modules import *
from util import *

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
np.random.seed(123)
if use_cuda:
    torch.cuda.manual_seed(123)


def custom_collate_fn(batch):
    # todo default truncates sequence till 80 words and <pad> is 10003
    bt_siz = len(batch)
    pad_idx, max_seq_len = 10003, 120

    u1_batch, u2_batch, u3_batch = [], [], []
    u1_lens, u2_lens, u3_lens = np.zeros(bt_siz, dtype=int), np.zeros(bt_siz, dtype=int), np.zeros(bt_siz, dtype=int)

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

    for i in range(bt_siz):
        seq1 = t1[i]
        u1_batch[i, :seq1.size(0)].data.copy_(seq1[:l_u1])
        seq2 = t2[i]
        u2_batch[i, :seq2.size(0)].data.copy_(seq2[:l_u2])
        seq3 = t3[i]
        u3_batch[i, :seq3.size(0)].data.copy_(seq3[:l_u3])

    sort1, sort2, sort3 = np.argsort(u1_lens*-1), np.argsort(u2_lens*-1), np.argsort(u3_lens*-1)

    return u1_batch[sort1, :], u1_lens[sort1].tolist(), u2_batch[sort2, :], u2_lens[sort2].tolist(), u3_batch[sort3, :], u3_lens[sort3].tolist()


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

    bt_siz, train_dataset, valid_dataset = 32, MovieTriples('train', 1000), MovieTriples('train', 32)
    train_dataloader = DataLoader(train_dataset, batch_size=bt_siz, shuffle=False, num_workers=2,
                                  collate_fn=custom_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=bt_siz, shuffle=False, num_workers=2,
                                  collate_fn=custom_collate_fn)

    print("Training set {} Validation set {}".format(len(train_dataset), len(valid_dataset)))

    optimizer = optim.Adam(all_params)
    criteria = nn.CrossEntropyLoss(ignore_index=10003, size_average=False)

    if use_cuda:
        criteria.cuda()

    for i in range(options.epoch):
        tr_loss = 0
        strt = time.time()
        for i_batch, sample_batch in enumerate(train_dataloader):
            u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], \
                                                    sample_batch[3], sample_batch[4], sample_batch[5]
            o1, o2 = base_enc(u1, u1_lens), base_enc(u2, u2_lens)
            qu_seq = torch.cat((o1, o2), 1)
            final_session_o = ses_enc(qu_seq)
            preds = dec(final_session_o, u3, u3_lens)
            preds = preds.view(-1, preds.size(2))
            # of size (N, SEQLEN, DIM)
            if use_cuda:
                u3 = u3.cuda()

            u3 = u3.view(-1)
            loss = criteria(preds, u3)

            loss = loss / u3.ne(10003).long().sum().data[0]
            tr_loss += loss.data[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm(all_params, 1.0)
            if i_batch % 100 == 0:
                print('done', i_batch)

        vl_loss = calc_valid_loss(valid_dataloader, criteria, base_enc, ses_enc, dec)
        print("Training loss {} Valid loss {} ".format(tr_loss/(1 + i_batch), vl_loss))
        print("epoch {} took {}".format(i+1, (time.time() - strt)/3600.0))
        if i % 2 == 0 or i == options.epoch -1:
            torch.save(base_enc.state_dict(), 'enc_mdl.pth')
            torch.save(ses_enc.state_dict(), 'ses_mdl.pth')
            torch.save(dec.state_dict(), 'dec_mdl.pth')
            torch.save(optimizer.state_dict(), 'opti_st.pth')


def main():
    print('torch version {}'.format(torch.__version__))

    # we use a common dict for all test, train and validation
    _dict_file = '/home/harshal/code/research/hred/data/MovieTriples_Dataset/Training.dict.pkl'
    with open(_dict_file, 'rb') as fp2:
        dict_data = pickle.load(fp2)
    # dictionary data is like ('</s>', 2, 588827, 785135)
    # so i believe that the first is the ids are assigned by frequency
    # thinking to use a counter collection out here maybe
    inv_dict = {}
    for x in dict_data:
        tok, f, _, _ = x
        inv_dict[f] = tok

    parser = argparse.ArgumentParser(description='HRED parameter options')
    parser.add_argument('-e', dest='epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('-tc', dest='teacher', type=bool, default=False, help='default teacher forcing')
    parser.add_argument('-bi', dest='bidi', type=bool, default=False, help='bidirectional enc/decs')
    parser.add_argument('-nl', dest='num_lyr', type=int, default=1, help='number of enc/dec layers(same for both)')
    options = parser.parse_args()
    print(options)

    base_enc = BaseEncoder(10004, 300, 1000, options.num_lyr, options.bidi)
    ses_enc = SessionEncoder(1500, 1000, options.num_lyr, options.bidi)
    dec = Decoder(10004, 300, 1500, 1000, options.num_lyr, options.bidi, options.teacher)
    if use_cuda:
        base_enc.cuda()
        ses_enc.cuda()
        dec.cuda()

    train(options, base_enc, ses_enc, dec)
    # chooses 10 examples only
    bt_siz, test_dataset = 1, MovieTriples('test', 10)
    test_dataloader = DataLoader(test_dataset, bt_siz, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    inference_beam(test_dataloader, base_enc, ses_enc, dec, inv_dict)


main()