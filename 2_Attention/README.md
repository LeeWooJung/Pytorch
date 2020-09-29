Practice for Attention Mechanism using Pytorch
==============================================

This model(Seq2Seq with various version of attentions) is to translate from English to Korean.  
This model uses tokenizer as follows.
* English tokenizer: spacy
* Korean tokenizer: mecab  
  
There are many arguments you can use.
* min_freq: Minimum frequency the words can be involved in Vocab[default: 5].
* seed: Random seed[default: 1024].
* batch_size: Size of batch[default: 256].
* dropout: The probability of dropout[default: 0.3].
* enc_emb_dim: Dimension of embedded source word in Encoder[default: 128].
* dec_emb_dim: Dimension of embedded target word in Decoder[default: 128].
* hidden_dim: Dimension of hidden state in Encoder & Decoder[default: 128].
* n_layers: Number of layers of botch Encoder & Decoder[default: 2].
* n_epochs: Number of epochs[default: 25].
* clip: Gradient clip[default: 1.0].
* model: Trained model name[default: SeqSeq-attention.pt].
* train: Check if you want to train model[action: store_true].
* test: Check if you want to test model[action: store_true].


Requirements
=============================================
* dataset
Since this model uses torchtext.data.TabularDataset in utils.py, you need to have .csv or .tsv type of data  
I use translation dataset from http://semanticweb.kaist.ac.kr/home/index.php/KAIST_Corpus  

* utils.py  
You need to download 'en' using "python -m spacy download en" so that you can implement spacy_en.tokenizer  
  
Version Control
=============================================

* python: 3.6.10
* pytorch: 1.5.1
* torchvision: 0.6.0a0+35d732a
* torchtext: 0.6.0
* numpy: 1.18.5
* konlpy: 0.5.2
* spacy: 2.3.2
