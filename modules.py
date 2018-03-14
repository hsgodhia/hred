import torch.nn as nn, torch, numpy as np, copy, pdb
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()


def max_out(x):
    # make sure s2 is even and that the input is 2 dimension
    if len(x.size()) == 2:
        s1, s2 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2 // 2, 2)
        x, _ = torch.max(x, 2)

    elif len(x.size()) == 3:
        s1, s2, s3 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2, s3 // 2, 2)
        x, _ = torch.max(x, 3)

    return x


# encode each sentence utterance into a single vector
class BaseEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, num_lyr, bidi):
        super(BaseEncoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = num_lyr
        self.drop = nn.Dropout(0.3)
        self.direction = 2 if bidi else 1
        # by default they requires grad is true
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=10003, sparse=False)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size,
                          num_layers=num_lyr, bidirectional=bidi, batch_first=True)

    def forward(self, x, x_lens):
        bt_siz = x.size(0)
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size), requires_grad=False)
        if use_cuda:
            x = x.cuda()
            h_0 = h_0.cuda()
        x_emb = self.embed(x)
        x_emb = self.drop(x_emb)
        x_emb = torch.nn.utils.rnn.pack_padded_sequence(x_emb, x_lens, batch_first=True)
        _, x_hid = self.rnn(x_emb, h_0)
        # move the batch to the front of the tensor
        x_hid = x_hid.view(x.size(0), -1, self.hid_size)
        return x_hid


# encode the hidden states of a number of utterances
class SessionEncoder(nn.Module):
    def __init__(self, hid_size, inp_size, num_lyr, bidi):
        super(SessionEncoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = num_lyr
        self.direction = 2 if bidi else 1
        self.rnn = nn.GRU(hidden_size=hid_size, input_size=inp_size,
                          num_layers=num_lyr, bidirectional=bidi, batch_first=True)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, x.size(0), self.hid_size), requires_grad=False)
        if use_cuda:
            h_0 = h_0.cuda()
        # output, h_n for output batch is already dim 0
        _, h_n = self.rnn(x, h_0)
        # move the batch to the front of the tensor
        h_n = h_n.view(x.size(0), -1, self.hid_size)
        return h_n


# decode the hidden state
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, ses_hid_size, hid_size, num_lyr=1, bidi=False, teacher=True):
        super(Decoder, self).__init__()
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.num_lyr = num_lyr
        self.drop = nn.Dropout(0.3)
        self.tanh = nn.Tanh()
        self.in_embed = nn.Embedding(vocab_size, emb_size, padding_idx=10003, sparse=False)
        self.rnn = nn.GRU(hidden_size=2*hid_size, input_size=emb_size,
                          num_layers=num_lyr, bidirectional=False, batch_first=True)

        self.lin1 = nn.Linear(ses_hid_size, hid_size)
        self.lin2 = nn.Linear(hid_size, emb_size)
        self.out_embed = nn.Linear(emb_size, vocab_size, False)
        self.log_soft2 = nn.LogSoftmax(dim=2)
        self.direction = 2 if bidi else 1
        self.teacher_forcing = teacher

    def do_decode(self, siz, seq_len, ses_encoding, target=None):
        preds = []
        tok = Variable(torch.ones(siz, 1).long(), requires_grad=False)
        hid_n = ses_encoding
        if use_cuda:
            tok = tok.cuda()
        for i in range(seq_len):
            if target is not None:
                tok = target.select(1, i)
                tok = tok.unsqueeze(1)

            tok_vec = self.in_embed(tok)
            tok_vec = self.drop(tok_vec)
            hid_o, hid_n = self.rnn(tok_vec, torch.cat((hid_n, ses_encoding), 2))
            hid_o = max_out(hid_o)
            hid_n = max_out(hid_n)
            hid_o = self.lin2(hid_o) + tok_vec
            hid_o = self.out_embed(hid_o)
            preds.append(hid_o)

            op = self.log_soft2(hid_o)
            _, max_ind = torch.max(op, dim=2)
            tok = max_ind.clone()

        dec_o = torch.cat(preds, 1)
        return dec_o

    def forward(self, ses_encoding, x=None, x_lens=None, beam=5):
        ses_encoding = self.tanh(self.lin1(ses_encoding))
        # indicator that we are doing inference
        if x is None:
            n_candidates = []
            candidates = [([1], 0)]
            gen_len = 1
            while gen_len <= 10:
                for c in candidates:
                    seq, score = c[0], c[1]
                    _target = Variable(torch.LongTensor([seq]), requires_grad=False)
                    dec_o = self.do_decode(1, len(seq), ses_encoding, _target)
                    op = dec_o[:, -1, :]
                    for i in range(op.size(1)):
                        n_candidates.append((seq + [i], score + op.data[0, i]))
                # hack to exponent sequence length by alpha-0.7
                n_candidates.sort(key=lambda temp: temp[1] / (1.0*len(temp[0])**0.7), reverse=True)
                candidates = copy.copy(n_candidates[:beam])
                n_candidates[:] = []
                gen_len += 1

            return candidates
        else:
            siz, seq_len = x.size(0), x.size(1)
            # start of sentence is the first tok
            if use_cuda:
                x = x.cuda()

            mask = x < 10003
            mask = mask.float()

            ses_encoding = ses_encoding.view(self.num_lyr*self.direction, siz, self.hid_size)
            if not self.teacher_forcing:
                dec_o = self.do_decode(siz, seq_len, ses_encoding)
            else:
                x_emb = self.in_embed(x)
                x_emb_pack = torch.nn.utils.rnn.pack_padded_sequence(x_emb, x_lens, batch_first=True)
                dec_o, dec_ts = self.rnn(x_emb_pack, torch.cat((ses_encoding, ses_encoding), 2))
                # dec_o is of size (batch, seq_len, hidden_size * num_directions)
                dec_o, _ = torch.nn.utils.rnn.pad_packed_sequence(dec_o, batch_first=True)
                dec_o = max_out(dec_o.contiguous())
                dec_o = dec_o * mask.unsqueeze(2)
                dec_o = self.lin2(dec_o) + x_emb  # padding index is embedded to 0 doesn't spoil the addition
                dec_o = self.out_embed(dec_o)

            return dec_o

    def set_teacher_forcing(self, val):
        self.teacher_forcing = val
