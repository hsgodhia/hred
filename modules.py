import torch.nn as nn, torch, copy, tqdm, math
from torch.autograd import Variable
import torch.nn.functional as F
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
    def __init__(self, vocab_size, emb_size, hid_size, options):
        super(BaseEncoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = options.num_lyr
        self.drop = nn.Dropout(0.3)
        self.direction = 2 if options.bidi else 1
        # by default they requires grad is true
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=10003, sparse=False)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size,
                          num_layers=self.num_lyr, bidirectional=options.bidi, batch_first=True, dropout=0.3)

    def forward(self, inp):
        x, x_lens = inp[0], inp[1]
        bt_siz, seq_len = x.size(0), x.size(1)
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size), requires_grad=False)
        if use_cuda:
            x = x.cuda()
            h_0 = h_0.cuda()
        x_emb = self.embed(x)
        x_emb = self.drop(x_emb)
        x_emb = torch.nn.utils.rnn.pack_padded_sequence(x_emb, x_lens, batch_first=True)
        x_o, x_hid = self.rnn(x_emb, h_0)
        x_o, _ = torch.nn.utils.rnn.pad_packed_sequence(x_o, batch_first=True)
        return x_o[:, -1, :].unsqueeze(1)


# encode the hidden states of a number of utterances
class SessionEncoder(nn.Module):
    def __init__(self, hid_size, inp_size, options):
        super(SessionEncoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = options.num_lyr
        self.direction = 2 if options.bidi else 1
        self.rnn = nn.GRU(hidden_size=hid_size, input_size=inp_size,
                          num_layers=options.num_lyr, bidirectional=options.bidi, batch_first=True)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, x.size(0), self.hid_size), requires_grad=False)
        if use_cuda:
            h_0 = h_0.cuda()
        # output, h_n for output batch is already dim 0
        _, h_n = self.rnn(x, h_0)
        # move the batch to the front of the tensor
        # return h_o if you want to decode intermediate queries as well
        h_n = h_n.view(x.size(0), -1, self.hid_size)
        return h_n


# decode the hidden state
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, ses_hid_size, hid_size, options):
        super(Decoder, self).__init__()
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.num_lyr = options.num_lyr
        self.drop = nn.Dropout(0.3)
        self.shared_weight = options.shrd_dec_emb
        self.tanh = nn.Tanh()
        self.in_embed = nn.Embedding(vocab_size, emb_size, padding_idx=10003, sparse=False)
        if not self.shared_weight:
            self.out_embed = nn.Linear(emb_size, vocab_size, bias=False)
        
        self.rnn = nn.GRU(hidden_size=2*hid_size, input_size=emb_size,
                          num_layers=self.num_lyr, bidirectional=False, batch_first=True, dropout=0.3)
        self.lm = nn.GRU(input_size=self.emb_size, hidden_size=self.hid_size, num_layers=self.num_lyr, batch_first=True, dropout=0.3)

        self.lin1 = nn.Linear(ses_hid_size, hid_size)
        self.lin2 = nn.Linear(2*self.hid_size, emb_size, False)
        self.lin3 = nn.Linear(self.hid_size, emb_size, False)
        
        self.direction = 2 if options.bidi else 1
        self.teacher_forcing = options.teacher
        self.diversity_rate = 1
        self.antilm_param = 20
        self.lambda_param = 0.4

    def do_decode(self, siz, seq_len, ses_encoding, target=None):
        ses_encoding = ses_encoding.view(self.direction * self.num_lyr, siz, self.hid_size)
        hid_n, preds, lm_preds = ses_encoding, [], []
        inp_tok = Variable(torch.ones(siz, 1).long(), requires_grad=False)
        lm_hid = Variable(torch.zeros(self.direction * self.num_lyr, siz, self.hid_size), requires_grad=False)
        
        if use_cuda:
            lm_hid = lm_hid.cuda()
            inp_tok = inp_tok.cuda()
            if target is not None:
                target = target.cuda()
                
        for i in range(seq_len):
            if self.teacher_forcing:
                inp_tok = target.select(1, i)
                inp_tok = inp_tok.unsqueeze(1)

            inp_tok_vec = self.in_embed(inp_tok)
            inp_tok_vec = self.drop(inp_tok_vec)
            hid_o, hid_n = self.rnn(inp_tok_vec, torch.cat((hid_n, ses_encoding), 2))
            lm_o, lm_hid = self.lm(inp_tok_vec, lm_hid)
            
            hid_n = hid_n[:, :, :self.hid_size]
            hid_o = self.lin2(hid_o)
            lm_o = self.lin3(lm_o)
            
            if self.shared_weight:
                hid_o = F.linear(hid_o, self.in_embed.weight)
                lm_o = F.linear(lm_o, self.in_embed.weight)
            else:
                hid_o = self.out_embed(hid_o)
                lm_o = self.out_embed(lm_o)
            
            lm_preds.append(lm_o)
            preds.append(hid_o)
            
            if not self.teacher_forcing:
                final_hid_o = hid_o
                op = final_hid_o[:, :, :-1]
                max_val, inp_tok = torch.max(op, dim=2)
                # now inp_tok will be val between 0 and 10002 ignoring padding_idx                
                # here we do greedy decoding
                # so we can ignore the last symbol which is a padding token
                # technically we don't need a softmax here as we just want to choose the max token, max score will result in max softmax.Duh! 
            
        dec_o = torch.cat(preds, 1)
        dec_lmo = torch.cat(lm_preds, 1)
        return dec_o, dec_lmo

    def forward(self, input):
        if len(input) == 1:
            ses_encoding = input
            x, x_lens = None, None
            beam = 5
        elif len(input) == 3:
            ses_encoding, x, x_lens = input
            beam = 5
        else:
            ses_encoding, x, x_lens, beam = input
            
        ses_encoding = self.tanh(self.lin1(ses_encoding))
        # indicator that we are doing inference
        if x is None:
            n_candidates, final_candids = [], []
            candidates = [([1], 0, 0)]
            gen_len, max_gen_len = 1, 20
            pbar = tqdm.tqdm(total=max_gen_len)
            while gen_len <= max_gen_len:
                for c in candidates:
                    seq, pts_score, pt_score = c[0], c[1], c[2]
                    _target = Variable(torch.LongTensor([seq]), requires_grad=False)
                    dec_o, dec_lm = self.do_decode(1, len(seq), ses_encoding, _target)
                    op = F.softmax(dec_o, 2)
                    lm_op = F.softmax(dec_lm, 2)

                    if lm_op.size(1) >= self.antilm_param:
                        lm_op[0, -1, :].data.mul_(0)

                    op = op[:, -1, :]
                    lm_op = lm_op[:, -1, :]
                    topval, topind = op.topk(beam, 1)

                    for i in range(beam):
                        ctok, cval = topind.data[0, i], topval.data[0, i]
                        utval = lm_op.data[0, ctok]
                        if ctok == 2:
                            list_to_append = final_candids
                            # prune it and for comparsion in final sequences
                        else:
                            list_to_append = n_candidates
                        # todo we don't include <s> score but include </s>
                        list_to_append.append((seq + [ctok], pts_score + math.log(cval) - self.diversity_rate*(i+1), pt_score + math.log(utval) if utval > 0 else 0))
                # hack to exponent sequence length by alpha-0.7
                n_candidates.sort(key=lambda temp: temp[1]/len(temp[0]), reverse=True)
                candidates = copy.copy(n_candidates[:beam])
                n_candidates[:] = []
                gen_len += 1
                pbar.update(1)
            pbar.close()
            final_candids = final_candids + candidates
            final_candids.sort(key=lambda temp: temp[1]/len(temp[0]), reverse=True)
            return final_candids[:beam]
        else:
            if use_cuda:
                x = x.cuda()
            siz, seq_len = x.size(0), x.size(1)
            ses_encoding = ses_encoding.view(self.num_lyr*self.direction, siz, self.hid_size)
            dec_o, dec_lm = self.do_decode(siz, seq_len, ses_encoding, x)
            return dec_o, dec_lm

    def set_teacher_forcing(self, val):
        self.teacher_forcing = val
