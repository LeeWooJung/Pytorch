Word2Vec implementation using Pytorch
================================================

* I wrote my own word2vec for English model(python-Pytorch) by referrring to the word2vec model code of the site below.  
  - https://github.com/theeluwin/pytorch-sgns

* Also, I wrote word2vec for Korean model.  
  - I get Korean corpus dataset from Wikepedia. You can download the dataset from "https://dumps.wikimedia.org/kowiki/latest/"
  - You can download Wikiextractor which cleans text from the Wikipedia dump file, from https://github.com/attardi/wikiextractor.git  
  - I recommend you download the wikiextractor using pip, and read carefully README.md file in that page  
  - Then, try to compose the all files(in text/AA, text/AB, etc.)  
  - For example,  
	cat ./text/AA/wiki* > wikiAA.txt  
	cat wiki* > korean_corpus.txt  
  - After then, run Korean_word2vec.py file.
  - This file make Tokenized sentences, Word2Idx, Idx2Word files, etc.
  - Also, this file execute the word2vec model.


* You can see Word2Vec paper at here:  
  - https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf


Version Control
==================================================

* torch: 1.5.1  
* torchvision: 0.6.0a0+35d732a  
* numpy: 1.18.5 
* tqdm: 4.47.0  
* konlpy: 0.5.2
