# LSTM-based-language-model

#### Description 
A language model implementation in Tensorflow using Long-Short-Term-Memory (LSTM) network. 

#### Data

- Training/Evaluation data available at [1]
- Word2Vec data available at [2]
- Testing data available at [3]

#### Validation

In runs_results directory can be found the results of the continuation model on the [1]/sentences.continuation file and the perplexity of the model on [3] in the following cases:
- using randomly generated word embeddings and a hidden state size of 512 in perplexity_random_w2v_hidden_layer_512.txt file
- using pretrained word embeddings and a hidden state size of 512 in perplexity_pretrained_w2v_hidden_layer_512.txt file
- using pretrained word embeddings and a hidden state size of 1024 in perplexity_pretrained_w2v_hidden_layer_1024.txt file

[1] https://polybox.ethz.ch/index.php/s/qUc2NvUh2eONfEB
[2] https://polybox.ethz.ch/index.php/s/cpicEJeC2G4tq9U
[3] https://polybox.ethz.ch/index.php/s/HJUnOuIj3K4FEdT