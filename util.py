import torch
import heapq


def tensor_to_sent(x, inv_dict):
    sent = []
    for i in x:
        sent.append(inv_dict[i])

    return " ".join(sent)


# sample a sentence from the test set by using beam search
# todo currently does greedy modify to do beam
def inference_beam(dataloader, base_enc, ses_enc, dec, inv_dict, width=5):
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

        # forward(self, ses_encoding, greedy=True, beam=5, x=None, x_lens=None):
        sent = dec(final_session_o, greedy=False)
        # print(tensor_to_sent(sent, inv_dict))
        print(sent)
