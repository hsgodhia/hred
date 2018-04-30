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


class Seq2Seq(nn.Module):
    def __init__(self, options):
        super(Seq2Seq, self).__init__()
        self.base_enc = BaseEncoder(options.vocab_size, options.emb_size, options.ut_hid_size, options)
        self.ses_enc = SessionEncoder(options.ses_hid_size, options.ut_hid_size, options)
        self.dec = Decoder(options)
        
    def forward(self, sample_batch):
        u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], \
        sample_batch[3], sample_batch[4], sample_batch[5]
        if use_cuda:
            u1 = u1.cuda()
            u2 = u2.cuda()
            u3 = u3.cuda()
        o1, o2 = self.base_enc((u1, u1_lens)), self.base_enc((u2, u2_lens))
        qu_seq = torch.cat((o1, o2), 1)
        final_session_o = self.ses_enc(qu_seq)
        preds, lmpreds = self.dec((final_session_o, u3, u3_lens))
        
        return preds, lmpreds
    
    
# encode each sentence utterance into a single vector
class BaseEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, options):
        super(BaseEncoder, self).__init__()
        self.hid_size = hid_size
        self.num_lyr = options.num_lyr
        self.drop = nn.Dropout(options.drp)
        self.direction = 2 if options.bidi else 1
        # by default they requires grad is true
        self.embed = nn.Embedding(vocab_size, emb_size, padding_idx=10003, sparse=False)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hid_size,
                          num_layers=self.num_lyr, bidirectional=options.bidi, batch_first=True, dropout=options.drp)

    def forward(self, inp):
        x, x_lens = inp[0], inp[1]
        bt_siz, seq_len = x.size(0), x.size(1)
        h_0 = Variable(torch.zeros(self.direction * self.num_lyr, bt_siz, self.hid_size))
        if use_cuda:
            h_0 = h_0.cuda()
        x_emb = self.embed(x)
        x_emb = self.drop(x_emb)
        x_emb = torch.nn.utils.rnn.pack_padded_sequence(x_emb, x_lens, batch_first=True)
        x_o, x_hid = self.rnn(x_emb, h_0)
        # assuming dimension 0, 1 is for layer 1 and 2, 3 for layer 2
        if self.direction == 2:
            x_hids = []
            for i in range(self.num_lyr):
                x_hid_temp, _ = torch.max(x_hid[2*i:2*i + 2, :, :], 0, keepdim=True)
                x_hids.append(x_hid_temp)
            x_hid = torch.cat(x_hids, 0)
        # x_o, _ = torch.nn.utils.rnn.pad_packed_sequence(x_o, batch_first=True)
        # using x_o and returning x_o[:, -1, :].unsqueeze(1) is wrong coz its all 0s careful! it doesn't adjust for variable timesteps
        
        x_hid = x_hid[self.num_lyr-1, :, :].unsqueeze(0)
        # take the last layer of the encoder GRU
        x_hid = x_hid.transpose(0, 1)
        
        return x_hid


# encode the hidden states of a number of utterances
class SessionEncoder(nn.Module):
    def __init__(self, hid_size, inp_size, options):
        super(SessionEncoder, self).__init__()
        self.hid_size = hid_size
        self.rnn = nn.GRU(hidden_size=hid_size, input_size=inp_size,
                          num_layers=1, bidirectional=False, batch_first=True, dropout=options.drp)

    def forward(self, x):
        h_0 = Variable(torch.zeros(1, x.size(0), self.hid_size))
        if use_cuda:
            h_0 = h_0.cuda()
        # output, h_n for output batch is already dim 0
        h_o, h_n = self.rnn(x, h_0)
        h_n = h_n.view(x.size(0), -1, self.hid_size)
        return h_n


# decode the hidden state
class Decoder(nn.Module):
    def __init__(self, options):
        super(Decoder, self).__init__()
        self.emb_size = options.emb_size
        self.hid_size = options.dec_hid_size
        self.num_lyr = 1
        self.teacher_forcing = options.teacher
        self.train_lm = options.lm
        
        self.drop = nn.Dropout(options.drp)
        self.tanh = nn.Tanh()
        self.shared_weight = options.shrd_dec_emb
        
        self.embed_in = nn.Embedding(options.vocab_size, self.emb_size, padding_idx=10003, sparse=False)
        if not self.shared_weight:
            self.embed_out = nn.Linear(self.emb_size, options.vocab_size, bias=False)
        
        self.rnn = nn.GRU(hidden_size=self.hid_size,input_size=self.emb_size,num_layers=self.num_lyr,batch_first=True,dropout=options.drp)
        
        self.ses_to_dec = nn.Linear(options.ses_hid_size, self.hid_size)
        self.dec_inf = nn.Linear(self.hid_size, self.emb_size*2, False)
        self.ses_inf = nn.Linear(options.ses_hid_size, self.emb_size*2, False)
        self.emb_inf = nn.Linear(self.emb_size, self.emb_size*2, True)
        self.tc_ratio = 1.0
        
        if options.lm:
            self.lm = nn.GRU(input_size=self.emb_size, hidden_size=self.hid_size, num_layers=self.num_lyr, batch_first=True, dropout=options.drp, bidirectional=False)
            self.lin3 = nn.Linear(self.hid_size, self.emb_size, False)

    def do_decode_tc(self, ses_encoding, target, target_lens):
        target_emb = self.embed_in(target)
        target_emb = self.drop(target_emb)
        # below will be used later as a crude approximation of an LM
        emb_inf_vec = self.emb_inf(target_emb)
        
        target_emb = torch.nn.utils.rnn.pack_padded_sequence(target_emb, target_lens, batch_first=True)
        
        init_hidn = self.tanh(self.ses_to_dec(ses_encoding))
        init_hidn = init_hidn.view(self.num_lyr, target.size(0), self.hid_size)
        
        hid_o, hid_n = self.rnn(target_emb, init_hidn)
        hid_o, _ = torch.nn.utils.rnn.pad_packed_sequence(hid_o, batch_first=True)        
        # linear layers not compatible with PackedSequence need to unpack, will be 0s at padded timesteps!
        
        dec_hid_vec = self.dec_inf(hid_o)
        ses_inf_vec = self.ses_inf(ses_encoding)
        total_hid_o = dec_hid_vec + ses_inf_vec + emb_inf_vec
        
        hid_o_mx = max_out(total_hid_o)        
        hid_o_mx = F.linear(hid_o_mx, self.embed_in.weight) if self.shared_weight else self.embed_out(hid_o_mx)
        
        if self.train_lm:
            siz = target.size(0)
            
            lm_hid0 = Variable(torch.zeros(self.num_lyr, siz, self.hid_size))
            if use_cuda:
                lm_hid0 = lm_hid0.cuda()

            lm_o, lm_hid = self.lm(target_emb, lm_hid0)
            lm_o, _ = torch.nn.utils.rnn.pad_packed_sequence(lm_o, batch_first=True)
            lm_o = self.lin3(lm_o)
            lm_o = F.linear(lm_o, self.embed_in.weight) if self.shared_weight else self.embed_out(lm_o)
            return hid_o_mx, lm_o
        else:
            return hid_o_mx, None
        
        
    def do_decode(self, siz, seq_len, ses_encoding, target):
        ses_inf_vec = self.ses_inf(ses_encoding)
        ses_encoding = self.tanh(self.ses_to_dec(ses_encoding))
        hid_n, preds, lm_preds = ses_encoding, [], []
        
        hid_n = hid_n.view(self.num_lyr, siz, self.hid_size)
        inp_tok = Variable(torch.ones(siz, 1).long())
        lm_hid = Variable(torch.zeros(self.num_lyr, siz, self.hid_size))
        if use_cuda:
            lm_hid = lm_hid.cuda()
            inp_tok = inp_tok.cuda()

        
        for i in range(seq_len):
            # initially tc_ratio is 1 but then slowly decays to 0 (to match inference time)
            if torch.randn(1)[0] < self.tc_ratio:
                inp_tok = target[:, i].unsqueeze(1)
            
            inp_tok_vec = self.embed_in(inp_tok)
            emb_inf_vec = self.emb_inf(inp_tok_vec)
            
            inp_tok_vec = self.drop(inp_tok_vec)
            
            hid_o, hid_n = self.rnn(inp_tok_vec, hid_n)
            dec_hid_vec = self.dec_inf(hid_o)
            
            total_hid_o = dec_hid_vec + ses_inf_vec + emb_inf_vec
            hid_o_mx = max_out(total_hid_o)

            hid_o_mx = F.linear(hid_o_mx, self.embed_in.weight) if self.shared_weight else self.embed_out(hid_o_mx)
            preds.append(hid_o_mx)
            
            if self.train_lm:
                lm_o, lm_hid = self.lm(inp_tok_vec, lm_hid)
                lm_o = self.lin3(lm_o)
                lm_o = F.linear(lm_o, self.embed_in.weight) if self.shared_weight else self.embed_out(lm_o)
                lm_preds.append(lm_o)
            
            op = hid_o[:, :, :-1]
            op = F.log_softmax(op, 2, 5)
            max_val, inp_tok = torch.max(op, dim=2)
            # now inp_tok will be val between 0 and 10002 ignoring padding_idx                
            # here we do greedy decoding
            # so we can ignore the last symbol which is a padding token
            # technically we don't need a softmax here as we just want to choose the max token, max score will result in max softmax.Duh! 
            
        dec_o = torch.cat(preds, 1)
        dec_lmo = torch.cat(lm_preds, 1) if self.train_lm else None
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
            
        if use_cuda:
            x = x.cuda()
        siz, seq_len = x.size(0), x.size(1)
        
        if self.teacher_forcing:
            dec_o, dec_lm = self.do_decode_tc(ses_encoding, x, x_lens)
        else:
            dec_o, dec_lm = self.do_decode(siz, seq_len, ses_encoding, x)
            
        return dec_o, dec_lm

    def set_teacher_forcing(self, val):
        self.teacher_forcing = val
    
    def get_teacher_forcing(self):
        return self.teacher_forcing
    
    def set_tc_ratio(self, new_val):
        self.tc_ratio = new_val
    
    def get_tc_ratio(self):
        return self.tc_ratio
