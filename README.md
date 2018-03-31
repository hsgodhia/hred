### Usage

----
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

#### Notes
 - Greedy decoding is used during training time if teacher forcing is disabled
 - During inference time the MMI-antiLM score is computed as per equation (15) in Jiwei Li (Diversity Promoting Paper)
 - an LM loss is by default included (through an additional plain RNN) and this jointly trained with the other parameters 
 - note: when processing the data, diverse sequence lengths lead to better optimization
#### Ablation study:-
 
 - *Activation function*: On a small data set I found that not using any activation is better than using a tanh function which is even better than using a maxout. 
 - *Dimension of embedding*: I observed that a dimension of 300 was better than 100 and 500 both 
 - *Decoder hidden layer size*: I observed that 1000 was better than 2000
 
#### Sanity check:-

 - If you load a small training set, like 1000 training and 100 valid as here `train_dataset, valid_dataset = MovieTriples('train', 1000), MovieTriples('valid', 100)` and train to overfit the model converges to 0.5 training loss in 50 epochs with training command `python3.6 main.py -n sample -bms 10 -e 50 -tc -lm`
 - Some samples generated at inference as compared to ground truth (on the test set) are
  - ```
  [("<s> i don ' t know . </s>", -11.935150146484375), ("<s> i ' m in from new york . i came to see <person> . </s>", -20.482309341430664), ("<s> i ' ll take you there , sir . </s>", -16.400659561157227), ("<s> i ' m sorry , but no one by that name lives here . </s>", -22.178613662719727), ("<s> i know it ' s none of my business -- </s>", -18.43322467803955), ("<s> i don ' t think you win elections by telling <number> percent of the people that they are . </s>", -27.444936752319336), ("<s> you ' re going to break up with <person> , aren ' t you ? </s>", -23.688961029052734), ("<s> i ' m afraid not . </s>", -14.662097930908203), ("<s> i don ' t know , do you ? <continued_utterance> it ' s a <person> . </s>", -25.888113021850586), ("<s> i ' ll be right back . </s>", -15.958183288574219)]
Ground truth [("<s> what ' s bugging her ? </s>", 0)] 
```
 