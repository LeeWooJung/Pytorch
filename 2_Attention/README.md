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

Model
=============================================
  
* Model architecture
[blank]  
* Attention
[blank]  
* Test Example
Sampled test examples of trained model with default arguments are as follows.  
  
```
Source sentence: i will go with you to the bridge , but not a step beyond.  
Target sentence: 내 가 너 를 데리 고 그 다리 까지 는 갈 수 있 지만 절대로 더 이상 은 갈 수 없 다 .  
Predicted sentence: 내 가 너 를 데리 고 그 다리 까지 는 갈 수 있 지만 절대로 더 이상 은 갈 수 없 다.  
```
  
```
Source sentence: i protest against beding treated in this way .  
Target sentence: 나 는 이러 한 대우 를 받 는 것 을 항의 했 다 .  
Predicted sentence: 나 는 이러 한 대우 를 받 는 것 을 숫자 했 다 .  
```
  
```
Source sentence: the shock <unk> his mind .  
Target sentence: 이 타격 은 그 를 <unk> 하 게 했 다 .  
Predicted sentence: 이 타격 은 그 를 <unk> 하 게 했 다 .  
```
  
```
Source sentence: we acknowledge receipt of your favour of the <unk> july , and thank you for the order you have given us .  
Target sentence: 귀사 의 7월 10 일 편지 를 받 았 다 . 이번 주문 에 대해 우리 회사 는 감사 를 표시 한다 .  
Predicted sentence: 귀사 의 갔 로 로 일 편지 를 받 았 다 . 중국 . 에 로 우리 회사 본 감사 를 갔 님 .
```
  
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
