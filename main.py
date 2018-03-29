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


def init_param(model):
    for name, param in model.named_parameters():
        # skip over the embeddings so that the padding index ones are 0
        if name.startswith('embed'):
            continue
        elif (name.startswith('rnn') or name.startswith('lm')) and len(param.size()) >= 2:
            init.orthogonal(param)
        else:
            init.normal(param, 0, 0.01)

def clip_gnorm(model):
    for name, p  in model.named_parameters():
        if name.startswith('lm') or name.startswith('lin3'):
            continue
        param_norm = p.grad.data.norm()
        if param_norm > 1:
            p.grad.data.mul_(1/param_norm)
                    
def train(options, base_enc, ses_enc, dec):
    base_enc.train()
    ses_enc.train()
    dec.train()

    all_params = list(base_enc.parameters()) + list(ses_enc.parameters()) + list(dec.parameters())
    optimizer = optim.Adam(all_params, options.lr)
    if options.btstrp:
        load_model_state(base_enc, options.btstrp + "_enc_mdl.pth")
        load_model_state(ses_enc, options.btstrp + "_ses_mdl.pth")
        load_model_state(dec, options.btstrp + "_dec_mdl.pth")
        load_model_state(optimizer, options.btstrp + "_opti_st.pth")
    else:
        init_param(base_enc)
        init_param(ses_enc)
        init_param(dec)

    train_dataset, valid_dataset = MovieTriples('train'), MovieTriples('valid')
    train_dataloader = DataLoader(train_dataset, batch_size=options.bt_siz, shuffle=True, num_workers=2,
                                  collate_fn=custom_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=options.bt_siz, shuffle=True, num_workers=2,
                                  collate_fn=custom_collate_fn)

    print("Training set {} Validation set {}".format(len(train_dataset), len(valid_dataset)))

    
    criteria = nn.CrossEntropyLoss(ignore_index=10003, size_average=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    if use_cuda:
        criteria.cuda()
    
    for i in range(options.epoch):
        scheduler.step()
        tr_loss, tlm_loss, num_words = 0, 0, 0
        strt = time.time()
        for i_batch, sample_batch in enumerate(tqdm(train_dataloader)):
            u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], \
                                                    sample_batch[3], sample_batch[4], sample_batch[5]
            o1, o2 = base_enc((u1, u1_lens)), base_enc((u2, u2_lens))
            qu_seq = torch.cat((o1, o2), 1)
            final_session_o = ses_enc(qu_seq)
            if use_cuda:
                u3 = u3.cuda()
            preds, lmpreds = dec([final_session_o, u3, u3_lens])  # of size (N, SEQLEN, DIM)
            lmpreds = lmpreds[:, :-1, :].contiguous().view(-1, lmpreds.size(2))
            preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
            u3 = u3[:, 1:].contiguous().view(-1)

            loss = criteria(preds, u3)
            lm_loss = criteria(lmpreds, u3)
            target_toks = u3.ne(10003).long().sum().data[0]
            num_words += target_toks
            tr_loss += loss.data[0]
            tlm_loss += lm_loss.data[0]
            
            loss = loss/target_toks
            lm_loss = lm_loss/target_toks
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            lm_loss.backward()
            
            clip_gnorm(base_enc)
            clip_gnorm(ses_enc)
            clip_gnorm(dec)
            
            optimizer.step()

        vl_loss = calc_valid_loss(valid_dataloader, criteria, base_enc, ses_enc, dec)
        print("Training loss {} lm loss {} Valid loss {}".format(tr_loss/num_words, tlm_loss/num_words, vl_loss))
        print("epoch {} took {} miss".format(i+1, (time.time() - strt)/60.0))
        if i % 2 == 0 or i == options.epoch -1:
            torch.save(base_enc.state_dict(), options.name + '_enc_mdl.pth')
            torch.save(ses_enc.state_dict(), options.name + '_ses_mdl.pth')
            torch.save(dec.state_dict(), options.name + '_dec_mdl.pth')
            torch.save(optimizer.state_dict(), options.name + '_opti_st.pth')


def load_model_state(mdl, fl):
    saved_state = torch.load(fl)
    mdl.load_state_dict(saved_state)


# sample a sentence from the test set by using beam search
def inference_beam(dataloader, base_enc, ses_enc, dec, inv_dict, options):
    load_model_state(base_enc, options.name + "_enc_mdl.pth")
    load_model_state(ses_enc, options.name + "_ses_mdl.pth")
    load_model_state(dec, options.name + "_dec_mdl.pth")

    base_enc.eval()
    ses_enc.eval()
    dec.eval()
    dec.set_teacher_forcing(False)

    for i_batch, sample_batch in enumerate(dataloader):
        u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], sample_batch[3], \
                                                sample_batch[4], sample_batch[5]
            
        
        o1, o2 = base_enc((u1, u1_lens)), base_enc((u2, u2_lens))
        qu_seq = torch.cat((o1, o2), 1)

        # if we need to decode the intermediate queries we may need the hidden states
        final_session_o = ses_enc(qu_seq)

        # forward(self, ses_encoding, x=None, x_lens=None, beam=5 ):
        sent = dec((final_session_o, None, None, options.beam))
        # print(sent)
        print(tensor_to_sent(sent, inv_dict))
        # greedy true for below because only beam generates a tuple of sequence and probability
        print("Ground truth {} \n".format(tensor_to_sent(u3.data.cpu().numpy(), inv_dict, True)))


def calc_valid_loss(data_loader, criteria, base_enc, ses_enc, dec):
    base_enc.eval()
    ses_enc.eval()
    dec.eval()
    cur_tc = dec.get_teacher_forcing()
    dec.set_teacher_forcing(False)
    
    valid_loss, num_words = 0, 0
    for i_batch, sample_batch in enumerate(data_loader):
        u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], sample_batch[3], \
                                                sample_batch[4], sample_batch[5]
        if use_cuda:
            u3 = u3.cuda()

        o1, o2 = base_enc((u1, u1_lens)), base_enc((u2, u2_lens))
        qu_seq = torch.cat((o1, o2), 1)
        final_session_o = ses_enc(qu_seq)

        preds, lmpreds = dec((final_session_o, u3, u3_lens))
        preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
        u3 = u3[:, 1:].contiguous().view(-1)
        
        # do not include the lM loss, exp(loss) is perplexity
        loss = criteria(preds, u3)
        num_words += u3.ne(10003).long().sum().data[0]
        valid_loss += loss.data[0]

    base_enc.train()
    ses_enc.train()
    dec.train()
    dec.set_teacher_forcing(cur_tc)
    
    return valid_loss/num_words


def data_to_seq():
    # we use a common dict for all test, train and validation
    _dict_file = '/home/harshals/hed-dlg/Data/MovieTriples/Training.dict.pkl'
    with open(_dict_file, 'rb') as fp2:
        dict_data = pickle.load(fp2)
    # dictionary data is like ('</s>', 2, 588827, 785135)
    # so i believe that the first is the ids are assigned by frequency
    # thinking to use a counter collection out here maybe
    inv_dict, vocab_dict = {}, {}
    for x in dict_data:
        tok, f, _, _ = x
        inv_dict[f] = tok
        vocab_dict[tok] = f
    _file = '/data2/chatbot_eval_issues/results/AMT_NCM_Test_NCM_Joao/neural_conv_model_eval_source.txt'
    with open(_file, 'r') as fp:
        all_seqs = []
        for lin in fp.readlines():
            seq = list()
            seq.append(1)
            for wrd in lin.split(" "):
                if wrd not in vocab_dict:
                    seq.append(0)
                else:
                    seq_id = vocab_dict[wrd]
                    seq.append(seq_id)
            seq.append(2)
        all_seqs.append(seq)

    with open('CustomTest.pkl', 'wb') as handle:
        pickle.dump(all_seqs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    print('torch version {}'.format(torch.__version__))
    _dict_file = '/home/harshals/hed-dlg/Data/MovieTriples/Training.dict.pkl'
    # we use a common dict for all test, train and validation
    
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
    parser.add_argument('-n', dest='name', help='enter suffix for model files', required=True)
    parser.add_argument('-e', dest='epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('-tc', dest='teacher', action='store_true', default=False, help='default teacher forcing')
    parser.add_argument('-bi', dest='bidi', action='store_true', default=False, help='bidirectional enc/decs')
    parser.add_argument('-test', dest='test', action='store_true', default=False, help='only test or inference')
    parser.add_argument('-shrd_dec_emb', dest='shrd_dec_emb', action='store_true', default=False, help='shared embedding in/out for decoder')
    parser.add_argument('-btstrp', dest='btstrp', default=None, help='bootstrap/load parameters give name')
    parser.add_argument('-nl', dest='num_lyr', type=int, default=1, help='number of enc/dec layers(same for both)')
    parser.add_argument('-lr', dest='lr', type=float, default=0.001, help='learning rate for optimizer')
    parser.add_argument('-bs', dest='bt_siz', type=int, default=100, help='batch size')
    parser.add_argument('-bms', dest='beam', type=int, default=1, help='beam size for decoding')

    options = parser.parse_args()
    print(options)

    base_enc = BaseEncoder(10004, 300, 1000, options)
    ses_enc = SessionEncoder(1500, 1000, options)
    dec = Decoder(10004, 300, 1500, 1000, options)
    if use_cuda:
        base_enc.cuda()
        ses_enc.cuda()
        dec.cuda()

    if not options.test:
        train(options, base_enc, ses_enc, dec)
    # chooses 10 examples only
    bt_siz, test_dataset = 1, MovieTriples('test', 100)
    test_dataloader = DataLoader(test_dataset, bt_siz, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    inference_beam(test_dataloader, base_enc, ses_enc, dec, inv_dict, options)


main()