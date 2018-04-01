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
        if 'embed' in name:
            continue
        elif ('rnn' in name or 'lm' in name) and len(param.size()) >= 2:
            init.orthogonal(param)
        else:
            init.normal(param, 0, 0.01)

def clip_gnorm(model):
    for name, p  in model.named_parameters():
        param_norm = p.grad.data.norm()
        if param_norm > 1:
            p.grad.data.mul_(1/param_norm)
                    
def train(options, model):
    model.train()
    optimizer = optim.Adam(model.parameters(), options.lr)
    if options.btstrp:
        load_model_state(model, options.btstrp + "_mdl.pth")
        load_model_state(optimizer, options.btstrp + "_opti_st.pth")
    else:
        init_param(model)

    train_dataset, valid_dataset = MovieTriples('train'), MovieTriples('valid')
    train_dataloader = DataLoader(train_dataset, batch_size=options.bt_siz, shuffle=True, num_workers=2,
                                  collate_fn=custom_collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=options.bt_siz, shuffle=True, num_workers=2,
                                  collate_fn=custom_collate_fn)

    print("Training set {} Validation set {}".format(len(train_dataset), len(valid_dataset)))

    
    criteria = nn.CrossEntropyLoss(ignore_index=10003, size_average=False)
    if use_cuda:
        criteria.cuda()
    
    for i in range(options.epoch):
        tr_loss, tlm_loss, num_words = 0, 0, 0
        strt = time.time()
        for i_batch, sample_batch in enumerate(tqdm(train_dataloader)):
            preds, lmpreds = model(sample_batch)
            u3 = sample_batch[4]
            if use_cuda:
                u3 = u3.cuda()
                
            preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
            u3 = u3[:, 1:].contiguous().view(-1)

            loss = criteria(preds, u3)
            target_toks = u3.ne(10003).long().sum().data[0]
            
            num_words += target_toks
            tr_loss += loss.data[0]
            loss = loss/target_toks
            
            if options.lm:
                lmpreds = lmpreds[:, :-1, :].contiguous().view(-1, lmpreds.size(2))
                lm_loss = criteria(lmpreds, u3)
                tlm_loss += lm_loss.data[0]
                lm_loss = lm_loss/target_toks
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            if options.lm:
                lm_loss.backward()
            clip_gnorm(model)
            optimizer.step()

        vl_loss = calc_valid_loss(valid_dataloader, criteria, model)
        print("Training loss {} lm loss {} Valid loss {}".format(tr_loss/num_words, tlm_loss/num_words, vl_loss))
        print("epoch {} took {} miss".format(i+1, (time.time() - strt)/60.0))
        if i % 2 == 0 or i == options.epoch -1:
            torch.save(model.state_dict(), options.name + '_mdl.pth')
            torch.save(optimizer.state_dict(), options.name + '_opti_st.pth')


def load_model_state(mdl, fl):
    saved_state = torch.load(fl)
    mdl.load_state_dict(saved_state)
    
    
def generate(model, ses_encoding, beam, diversity_rate):
    n_candidates, final_candids = [], []
    candidates = [([1], 0, 0)]
    gen_len, max_gen_len = 1, 20
    pbar = tqdm(total=max_gen_len)


    # we provide the top k options/target defined each time
    while gen_len <= max_gen_len:
        for c in candidates:
            seq, pts_score, pt_score = c[0], c[1], c[2]
            _target = Variable(torch.LongTensor([seq]), volatile=True)
            dec_o, _ = model.dec([ses_encoding, _target, [len(seq)]])
            dec_o = dec_o[:, :, :-1]

            op = F.log_softmax(dec_o, 2, 5)
            op = op[:, -1, :]
            topval, topind = op.topk(beam, 1)

            for i in range(beam):
                ctok, cval = topind.data[0, i], topval.data[0, i]
                if ctok == 2:
                    list_to_append = final_candids
                else:
                    list_to_append = n_candidates

                list_to_append.append((seq + [ctok], pts_score + cval - diversity_rate*(i+1), 0))

        n_candidates.sort(key=lambda temp: temp[1]/len(temp[0])**0.7, reverse=True)
        candidates = copy.copy(n_candidates[:beam])
        n_candidates[:] = []
        gen_len += 1
        pbar.update(1)

    pbar.close()
    final_candids = final_candids + candidates
    final_candids.sort(key=lambda temp: temp[1]/len(temp[0])**0.7, reverse=True)

    return final_candids[:beam]    

# sample a sentence from the test set by using beam search
def inference_beam(dataloader, model, inv_dict, options):
    cur_tc = model.dec.get_teacher_forcing()
    model.dec.set_teacher_forcing(True)
    
    load_model_state(model, options.name + "_mdl.pth")
    model.eval()
    diversity_rate = 1
    antilm_param = 20
    lambda_param = 0.4

    for i_batch, sample_batch in enumerate(dataloader):
        u1, u1_lens, u2, u2_lens, u3, u3_lens = sample_batch[0], sample_batch[1], sample_batch[2], sample_batch[3], \
                                                sample_batch[4], sample_batch[5]
            
        if use_cuda:
            u1 = u1.cuda()
            u2 = u2.cuda()
        o1, o2 = model.base_enc((u1, u1_lens)), model.base_enc((u2, u2_lens))
        qu_seq = torch.cat((o1, o2), 1)
        # if we need to decode the intermediate queries we may need the hidden states
        final_session_o = model.ses_enc(qu_seq)
        # forward(self, ses_encoding, x=None, x_lens=None, beam=5 ):
        sent = generate(model, final_session_o, options.beam, diversity_rate)
        # print(sent)
        print(tensor_to_sent(sent, inv_dict))
        # greedy true for below because only beam generates a tuple of sequence and probability
        print("Ground truth {} \n".format(tensor_to_sent(u3.data.cpu().numpy(), inv_dict, True)))

    model.dec.set_teacher_forcing(cur_tc)

def calc_valid_loss(data_loader, criteria, model):
    model.eval()
    cur_tc = model.dec.get_teacher_forcing()
    model.dec.set_teacher_forcing(True)
    # we want to find the perplexity or likelihood of the provided sequence
    
    valid_loss, num_words = 0, 0
    for i_batch, sample_batch in enumerate(data_loader):
        preds, lmpreds = model(sample_batch)
        u3 = sample_batch[4]
        if use_cuda:
            u3 = u3.cuda()
        preds = preds[:, :-1, :].contiguous().view(-1, preds.size(2))
        u3 = u3[:, 1:].contiguous().view(-1)
        # do not include the lM loss, exp(loss) is perplexity
        loss = criteria(preds, u3)
        num_words += u3.ne(10003).long().sum().data[0]
        valid_loss += loss.data[0]

    model.train()
    model.dec.set_teacher_forcing(cur_tc)
    
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
    parser.add_argument('-lm', dest='lm', action='store_true', default=False, help='enable a RNN language model joint training as well')
    parser.add_argument('-drp', dest='drp', type=float, default=0.3, help='dropout probability used all throughout')
    parser.add_argument('-nl', dest='num_lyr', type=int, default=1, help='number of enc/dec layers(same for both)')
    parser.add_argument('-lr', dest='lr', type=float, default=0.001, help='learning rate for optimizer')
    parser.add_argument('-bs', dest='bt_siz', type=int, default=100, help='batch size')
    parser.add_argument('-bms', dest='beam', type=int, default=1, help='beam size for decoding')

    options = parser.parse_args()
    print(options)

    model = Seq2Seq(options)
    if use_cuda:
        model.cuda()

    if not options.test:
        train(options, model)
    # chooses 10 examples only
    bt_siz, test_dataset = 1, MovieTriples('test', 100)
    test_dataloader = DataLoader(test_dataset, bt_siz, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    inference_beam(test_dataloader, model, inv_dict, options)


main()
