import torch.nn as nn, torch, numpy as np
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()


# encode each sentence utterance into a single vector
class BaseEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, num_lyr, bidi):
        super(BaseEncoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = num_lyr
        self.direction = 2 if bidi else 1
        # by default they requires grad is true
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(dropout=0.2, bias=False, input_size=emb_size, hidden_size=hid_size,
                          num_layers=num_lyr, bidirectional=bidi, batch_first=True)

    def forward(self, x, x_lens):
        bt_siz = x.data.size(0)
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size))
        if use_cuda:
            x = x.cuda()
            h_0 = h_0.cuda()
        x_emb = self.embed(x)
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
        self.rnn = nn.GRU(dropout=0.2, hidden_size=hid_size, input_size=inp_size,
                          num_layers=num_lyr, bidirectional=bidi, batch_first=True)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, x.size(0), self.hid_size))
        if use_cuda:
            h_0 = h_0.cuda()

        # output, h_n for output batch is already dim 0
        _, o = self.rnn(x, h_0)
        # move the batch to the front of the tensor
        o = o.view(x.size(0), -1, self.hid_size)
        return o


# decode the hidden state
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, ses_hid_size, hid_size, num_lyr=1, bidi=False, teacher=True):
        super(Decoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = num_lyr
        self.embed = nn.Embedding(vocab_size, emb_size)  # currently the output embedding doesn't share weight
        self.direction = 2 if bidi else 1
        self.lin1 = nn.Linear(ses_hid_size, hid_size)
        self.tanh = nn.Tanh()
        self.rnn = nn.GRU(dropout=0.2, hidden_size=hid_size, input_size=emb_size,
                          num_layers=num_lyr, bidirectional=False, batch_first=True)
        self.lin2 = nn.Linear(hid_size, vocab_size)
        self.log_soft = nn.LogSoftmax(dim=2)
        self.loss_cri = nn.NLLLoss()
        self.teacher_forcing = teacher

    def forward(self, ses_encoding, x=None, x_lens=None):
        ses_encoding = self.tanh(self.lin1(ses_encoding))

        if x is None:
            # we use this for inference when the only input is the ses_encoding
            siz = ses_encoding.size(0)
            # in each iteration the generated sentence's length increases by 1 so loop terminates
            gen_len, sent, hid_n = 0, np.zeros(siz, 50), ses_encoding
            # the start token has index 1

            tok = Variable(torch.ones(siz, 1).long())
            sent[:, 0] = tok.data[:, 0].numpy()
            if use_cuda:
                tok = tok.cuda()

            while True:
                if gen_len > 50 or tok.data[0, 0] == 2:
                    break
                tok_vec = self.embed(tok)
                hid_n, _ = self.rnn(tok_vec, hid_n)
                op = self.lin2(hid_n)
                op = self.log_soft(op)
                op = op.squeeze(1)
                tok_val, tok = torch.max(op, dim=1, keepdim=True)
                sent[:, gen_len] = tok.data[:, 0].numpy()
                gen_len += 1
            return sent
        else:
            loss = 0
            if use_cuda:
                x = x.cuda()
            siz, seq_len = x.size(0), x.size(1)
            mask = x < 10003
            mask = mask.float()

            x_emb = self.embed(x)
            x_emb = torch.nn.utils.rnn.pack_padded_sequence(x_emb, x_lens, batch_first=True)
            ses_encoding = ses_encoding.view(self.num_lyr*self.direction, siz, self.hid_size)

            if not self.teacher_forcing:
                # start of sentence is the first tok
                tok = Variable(self.embed.weight.data[1, :].repeat(siz, 1))
                tok = tok.unsqueeze(1)
                hid_n = ses_encoding

                for i in range(seq_len):
                    hid_o, hid_n = self.rnn(tok, hid_n)
                    # hid_o (seq_len, batch, hidden_size * num_directions) batch _first affects
                    # hid_n (num_layers * num_directions, batch, hidden_size)  batch_first doesn't affect
                    # h_0 (num_layers * num_directions, batch, hidden_size) batch_first doesn't affect
                    op = self.lin2(hid_o)
                    op = self.log_soft(op)
                    op = op.squeeze(1)
                    op = op * mask[:, i].unsqueeze(1)
                    if i+1 < seq_len:
                        loss += self.loss_cri(op, x[:, i+1])
                        _, tok = torch.max(op, dim=1, keepdim=True)
                        tok = self.embed(tok)
            else:
                # I'm directly doing teacher forcing here by feeding the true sequence via embedding layer
                dec_o, dec_ts = self.rnn(x_emb, ses_encoding)
                # dec_o is of size (batch, seq_len, hidden_size * num_directions)

                # got a input is not contiguous error above
                dec_o, _ = torch.nn.utils.rnn.pad_packed_sequence(dec_o, batch_first=True)
                dec_o = self.lin2(dec_o)
                dec_o = self.log_soft(dec_o)
                dec_o = dec_o * mask.unsqueeze(2)
                # here the dimension is N*SEQ_LEN*VOCAB_SIZE
                for i in range(seq_len):
                    loss += self.loss_cri(dec_o[:, i, :], x[:, i])

            return loss

    def set_teacher_forcing(self, val):
        self.teacher_forcing = val
