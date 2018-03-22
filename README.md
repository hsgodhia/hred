### Usage

----
#### Train

`python3.6 main.py -tc -e 100 -n full_tc -bms 20 -bs 80`

- tc says that use teacher forcing for the entire training procedure
- bms is the beam size for decoing used only during inference time, during training if teacher forcing is disabled greedy decoding is used
- n is a required parameter that gives a name to the model files
- bs is the batch size
- e is the number of epochs 
- test boolean switch says to run only inference mode
- btstrp flag gives a name of a pre-trained model which is used for parameter initializations instead of the default of gaussian mean 0 and standard deviationn 0.01

#### Ablation study:-
 
 - *Activation function*: On a small data set I found that not using any activation is better than using a tanh function which is even better than using a maxout. 
 - *Dimension of embedding*: I observed that a dimension of 300 was better than 100 and 500 both 
 - *Decoder hidden layer size*: I observed that 1000 was better than 2000
 