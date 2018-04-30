##### An implementation of the paper  [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](https://arxiv.org/abs/1507.04808) and  [Mutual Information and Diverse Decoding Improve Neural Machine Translation](https://arxiv.org/abs/1601.00372)

#### Results

The model is *able to replicate the results* of the paper. 

| Model           | Test Perplexity | Training Loss | #of epochs | Diversity ratio |
|-----------------|-----------------|---------------|------------|-----------------|
| HRED            | 35.128          | 3.334         | 8          | NA              |
| HRED*+Bi+LM     | 35.694          | 3.811         | 7          | 18.609%         |
| HRED*+Bi+LM     | 33.458          | 3.334         | 25         | 12.908%         |

Model 1
`python3.6 main.py -n full_final2 -tc -bms 20 -bs 100 -e 80 -seshid 300 -uthid 300 -drp 0.4 -lr 0.0005`

Model 2 (curriculum learning with inverse sigmoid teacher forcing ratio decay)

`python3.6 main.py -n curlrn -bi -lm -nl 2 -lr 0.0003 -e 10 -seshid 300 -uthid 300`

Model 3 (100% teacher forcing)
`python3.6 main.py -n onlytc -nl 2 -bi -lm -drp 0.4 -e 25 -seshid 300 -uthid 300 -lr 0.0001 -bs 100 -tc`

 - We notice over fitting on the validation loss (patience 3) from epoch 8 onwards for the first, second model and from epoch 24 (smaller learning rate) for the 1st one 
 - Training time is about 15 mins(30 mins w/o teacher forcing) per epoch on a Tesla Geforce GTX Titan X consuming about 11GB of GPU RAM
 - Beam search decoding with size 50 is used, *MMI anti-LM* is used for ranking the results
 - Test set results (w/ ground truth for teacher forcing 100%) available [here](https://github.com/hsgodhia/hred/blob/master/onlytc_result.txt)
 - Test set results (w/ ground truth for curriculum learning) available [here](https://github.com/hsgodhia/hred/blob/master/curlrn_result.txt)
 - For inference use the flags `-test -mmi -bms 50`


#### Notes
 - Greedy decoding is used during training time if teacher forcing is disabled (by default we train with tc)
 - During inference time the MMI-antiLM score is computed as per equation (15) in Jiwei Li (Diversity Promoting Paper)
 - an LM loss is by default included (through an additional plain RNN) and this jointly trained with the other parameters, although not much difference in results are obtained and to speedup I often disable it
 - When processing the data, diverse sequence lengths in a given batch leads to better optimization so no sorting of training data
 - Validation/test perplexity is calculated using teacher forcing as we want to capture the true work log likelihood
 - Inference or generation is using beam search
 - Note with curriculum learning we get more diversity during generation or inference time (almost  
 
#### Train

`python3.6 main.py -tc -e 100 -n full_tc -bms 20 -bs 80`

A brief list of options is given below, for a longer list please see main.py file
- tc says that use teacher forcing for the entire training procedure
- bms is the beam size for decoing used only during inference time, during training if teacher forcing is disabled greedy decoding is used
- n is a required parameter that gives a name to the model files
- bs is the batch size
- e is the number of epochs 
- test boolean switch says to run only inference mode
- btstrp flag gives a name of a pre-trained model which is used for parameter initializations instead of the default of gaussian mean 0 and standard deviationn 0.01
 
#### Sanity check:-

 - If you load a small training set, like 1000 training and 100 valid as here `train_dataset, valid_dataset = MovieTriples('train', 1000), MovieTriples('valid', 100)` and train to overfit the model converges to 0.5 training loss in 50 epochs with training command `python3.6 main.py -n sample -tc -bms 20 -bs 100 -e 50 -seshid 300 -uthid 300 -drp 0.4`
 
 - Some samples generated at inference as compared to ground truth (on the test set) are
   ```
   [("i don ' t know . ", -11.935150146484375), ("  i ' m in from new york . i came to see <person> .  ", -20.482309341430664), ("  i ' ll take you there , sir .  ", -16.400659561157227), ("  i ' m sorry , but no one by that name lives here .  ", -22.178613662719727), ("  i know it ' s none of my business --  ", -18.43322467803955), ("  i don ' t think you win elections by telling <number> percent of the people that they are .  ", -27.444936752319336), ("  you ' re going to break up with <person> , aren ' t you ?  ", -23.688961029052734), ("  i ' m afraid not .  ", -14.662097930908203), ("  i don ' t know , do you ? <continued_utterance> it ' s a <person> .  ", -25.888113021850586), ("  i ' ll be right back .  ", -15.958183288574219)]Ground truth [("  what ' s bugging her ?  ", 0)] 
   ```