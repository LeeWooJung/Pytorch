from konlpy.tag import Mecab
import spacy

def eng_tokenizer(text):

	spacy_en = spacy.load('en')
	return [tok.text for tok in spacy_en.tokenizer(text)][::-1]

def kor_tokenizer(text):#, src = True):

	mecab = Mecab()
	return [tok for tok in mecab.morphs(text)]
