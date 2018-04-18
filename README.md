# LSTM-based-language-model

#### Description 
A language model implementation in Tensorflow using Long-Short-Term-Memory (LSTM) network. 

#### Data

- Training/Evaluation data available at: https://polybox.ethz.ch/index.php/s/qUc2NvUh2eONfEB
- Word2Vec data available at: https://polybox.ethz.ch/index.php/s/cpicEJeC2G4tq9U
- Testing data available at: https://polybox.ethz.ch/index.php/s/HJUnOuIj3K4FEdT
#### Output examples 



#### Install @Leonard
- module load python_gpu/3.6.1 cuda/9.0.176 cudnn/7.0
- pip install --user pipenv
- pipenv install numpy
- pipenv install tensorflow-gpu==1.6
- pipenv install gensim
- pipenv shell #To activate this project's virtualenv.  Use 'exit' to leave.


#### Validation

In runs_results can be found the results of the continuation model on the sentences.continuation file and the perplexity of the model in the following cases:
- using randomly generated word embeddings and a hidden state size of 512 in perplexity_random_w2v_hidden_layer_512.txt file
- using pretrained word embeddings and a hidden state size of 512 in perplexity_pretrained_w2v_hidden_layer_512.txt file
- using pretrained word embeddings and a hidden state size of 1024 in perplexity_pretrained_w2v_hidden_layer_1024.txt file

