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

    train_dataset, valid_dataset = MovieTriples('train', 1000), MovieTriples('train', 32)
    train_dataloader = DataLoader(train_dataset, batch_size=options.bt_siz, shuffle=False, num_workers=2,
                                  collate_fn=custom_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=options.bt_siz, shuffle=False, num_workers=2,
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
            preds = dec(final_session_o, u3, u3_lens)  # of size (N, SEQLEN, DIM)
            preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))

            u3 = u3[:, 1:].contiguous().view(-1)
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


# sample a sentence from the test set by using beam search
def inference_beam(dataloader, base_enc, ses_enc, dec, inv_dict, beam=5):
    saved_state = torch.load("enc_mdl.pth")
    base_enc.load_state_dict(saved_state)

    saved_state = torch.load("ses_mdl.pth")
    ses_enc.load_state_dict(saved_state)

    saved_state = torch.load("dec_mdl.pth")
    dec.load_state_dict(saved_state)

    base_enc.eval()
    ses_enc.eval()
    dec.eval()

    for i_batch, sample_batch in enumerate(dataloader):
        u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], sample_batch[3], \
                                                sample_batch[4], sample_batch[5]
        o1, o2 = base_enc(u1, u1_lens), base_enc(u2, u2_lens)
        qu_seq = torch.cat((o1, o2), 1)

        # if we need to decode the intermediate queries we may need the hidden states
        final_session_o = ses_enc(qu_seq)

        # forward(self, ses_encoding, x=None, x_lens=None, beam=5 ):
        sent = dec(final_session_o, None, None)
        # print(sent)
        print(tensor_to_sent(sent, inv_dict))
        # greedy true for below because only beam generates a tuple of sequence and probability
        print("Ground truth {} \n".format(tensor_to_sent(u3.data.cpu().numpy(), inv_dict, True)))


def calc_valid_loss(data_loader, criteria, base_enc, ses_enc, dec):
    base_enc.eval()
    ses_enc.eval()
    dec.eval()

    valid_loss = 0
    for i_batch, sample_batch in enumerate(data_loader):
        u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], sample_batch[3], \
                                                sample_batch[4], sample_batch[5]

        o1, o2 = base_enc(u1, u1_lens), base_enc(u2, u2_lens)
        qu_seq = torch.cat((o1, o2), 1)
        final_session_o = ses_enc(qu_seq)

        preds = dec(final_session_o, u3, u3_lens)
        preds = preds.view(-1, preds.size(2))
        # of size (N, SEQLEN, DIM)
        u3 = u3.view(-1)
        loss = criteria(preds, u3)

        loss = loss / u3.ne(10003).long().sum().data[0]
        valid_loss += loss.data[0]

    base_enc.train()
    ses_enc.train()
    dec.train()

    return valid_loss/(1 + i_batch)


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
    parser.add_argument('-tc', dest='teacher', action='store_true', default=False, help='default teacher forcing')
    parser.add_argument('-bi', dest='bidi', action='store_true', default=False, help='bidirectional enc/decs')
    parser.add_argument('-nl', dest='num_lyr', type=int, default=1, help='number of enc/dec layers(same for both)')
    parser.add_argument('-bs', dest='bt_siz', type=int, default=80, help='batch size')
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
    bt_siz, test_dataset = 1, MovieTriples('train', 10)
    test_dataloader = DataLoader(test_dataset, bt_siz, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    inference_beam(test_dataloader, base_enc, ses_enc, dec, inv_dict)


main()