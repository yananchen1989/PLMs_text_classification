import sys,os,logging,glob,pickle,random
import numpy as np
import torch
import pandas as pd 

from transformers import AutoTokenizer, AutoModelWithLMHead
from flair.data import Sentence
from flair.models import SequenceTagger
# from nltk.corpus import stopwords
# stopwords = stopwords.words('english')
from load_data import *
# distilroberta-base
class fillInmask():
    def __init__(self, model_name='distilbert-base-uncased', mask_ratio=0.35, ner_set=True):
        self.model_name = model_name
        self.mask_ratio = mask_ratio
        self.ner_set = ner_set
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load tagger
        self.tagger = SequenceTagger.load("flair/ner-english-fast")
        self.load_model()
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelWithLMHead.from_pretrained(self.model_name)
        self.model.to(self.device)

    def get_ners(self, text):
        # make example sentence
        sentence = Sentence(text)
        # predict NER tags
        self.tagger.predict(sentence)
        ners = list(set([ii['text'] for ii in sentence.to_dict(tag_type='ner')['entities']]))
        ratio = 1
        ners_to_masked = random.sample(ners, int(len(ners) * ratio ))
        return ners_to_masked

    def get_random_span(self, text):
        tokens = list(set(text.split(' ')))
        random_tokens = random.sample(tokens, int(len(tokens)* self.mask_ratio ) )
        return random_tokens

    def augment(self, text ):
        ner_replace = {}
        if self.ner_set:
            ners_to_masked = self.get_ners(text)
        else:
            ners_to_masked = self.get_random_span(text)
            #print(ners_to_masked)
        for ner in ners_to_masked:
            if len(ner)<=2 or ner.lower() in stopwords:
                continue
            text_masked = text.replace(ner, self.tokenizer.mask_token)

            input_encode = self.tokenizer.encode(text_masked, return_tensors="pt", truncation=True).to(self.device) 
            mask_token_index = torch.where(input_encode == self.tokenizer.mask_token_id)[1]
            if mask_token_index.shape[0] == 0:
                continue
            token_logits = self.model(input_encode).logits
            mask_token_logits = token_logits[0, mask_token_index, :]

            top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

            ner_replace[ner] = self.tokenizer.decode(top_5_tokens[0])
            #print('{} to be replaced by filled in ==>{}'.format(ner, ner_replace[ner]) )

        for ner, alter in ner_replace.items():
            text = text.replace(ner, alter)
        return text


# unit test
'''
augmentor = fillInmask(ner_set=False)
ds = load_data(dataset='bbc', samplecnt=1000)
content_fillin = ds.df_train['content'].map(lambda x: augmentor.augment(x)).tolist()
    
for content in ds.df_train['content'].tolist():
    content_ = augmentor.augment(content)



ners_to_masked = augmentor.get_random_span(content)
    #print(ners_to_masked)
for ner in ners_to_masked:
    if len(ner)<=2 or ner.lower() in stopwords:
        continue
    text_masked = content.replace(ner, augmentor.tokenizer.mask_token)

    input_encode = augmentor.tokenizer.encode(text_masked, return_tensors="pt", truncation=True).to(augmentor.device) 
    mask_token_index = torch.where(input_encode == augmentor.tokenizer.mask_token_id)[1]
    if mask_token_index.shape[0] == 0:
        continue
    token_logits = augmentor.model(input_encode).logits
    mask_token_logits = token_logits[0, mask_token_index, :]

    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
'''



'''
content = """
"As the 2004 Summer Olympics officially get underway Friday with an international broadcast of the opening ceremonies, health experts expect the Athens games to inspire couch potatoes to become more active. But, experts caution, amateurs, particularly sedentary ones, should not jump into a new sports activity without sufficient preparation."
"""
nlp = pipeline("fill-mask" , model = 'distilbert-base-uncased')
content_mask = content.replace('sports', nlp.tokenizer.mask_token)

nlp(text_masked)

'''



'''
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

label_list = [
     "O",       # Outside of a named entity
     "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
     "I-MISC",  # Miscellaneous entity
     "B-PER",   # Beginning of a person's name right after another person's name
     "I-PER",   # Person's name
     "B-ORG",   # Beginning of an organisation right after another organisation
     "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
     "I-LOC"    # Location
 ]

tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="pt")

outputs = model(inputs).logits
predictions = torch.argmax(outputs, dim=2)

ORGs, LOCs, PERs, MISCs = [], [], [], []

wner = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].detach().numpy())]
for w, ner in wner:
    if ner.endswith('-MISC'):
        MISCs.append(w)
    if ner.endswith('-PER'):
        PERs.append(w)
    if ner.endswith('-ORG'):
        ORGs.append(w)
    if ner.endswith('-LOC'):
        LOCs.append(w)
    print(w, ner )

NERs = []

for ii in (ORGs, LOCs, PERs, MISCs):
    for j in list(set(tokenizer.decode(tokenizer.convert_tokens_to_ids(ii)).split(' '))):
        if not j or '#' in j:
            continue
        NERs.append(j)
'''









