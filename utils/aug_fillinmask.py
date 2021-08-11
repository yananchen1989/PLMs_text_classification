import sys,os,logging,glob,pickle,random
import numpy as np
import torch,spacy
import pandas as pd 
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead
from utils.load_data import *
#from flair.data import Sentence
#from flair.models import SequenceTagger
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
#from load_data import *
# distilroberta-base
class fillInmask():
    def __init__(self):
        #self.model_name = model_name
        #self.mask_ratio = mask_ratio
        #self.ner_set = ner_set
        
        #self.device = torch.device(device)
        # load tagger
        # if torch.__version__.startswith('1.8'):
        #     self.tagger = SequenceTagger.load("flair/ner-english-large")
        # else:
        #     self.tagger = SequenceTagger.load("flair/ner-english-fast")
        #self.tagger = SequenceTagger.load("ner-large")
        # python -m spacy download en_core_web_sm
        self.ner_model = spacy.load('en_core_web_sm')
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased',cache_dir="./cache",local_files_only=True)
        self.model = AutoModelWithLMHead.from_pretrained('distilbert-base-uncased',cache_dir="./cache",local_files_only=True)

        #if torch.cuda.is_available():
        self.nlp = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer, device=0)



    #def get_ners(self, text):
        # make example sentence
        #sentence = Sentence(text)
        # predict NER tags
        #self.tagger.predict(sentence)
        #ners = list(set([ii['text'] for ii in sentence.to_dict(tag_type='ner')['entities']]))
        
        #return doc.ents

    # def get_random_span(self, text):
    #     tokens = list(set(text.split(' ')))
    #     random_tokens = random.sample(tokens, int(len(tokens)* self.mask_ratio ) )
    #     return random_tokens

    def augment(self, sent ):
        doc = self.ner_model(sent)
        ners_to_masked = list(set([ii.text for ii in doc.ents]))
        for ner in ners_to_masked:
            if len(ner)<=2 or ner.lower() in stopwords or ner not in sent:
                continue
            text_masked = sent.replace(ner, self.nlp.tokenizer.mask_token, 1) 
            fillin_results = self.nlp(text_masked)
            fillin_ners = [i['token_str'] for i in fillin_results]
            sent = text_masked.replace(self.nlp.tokenizer.mask_token, fillin_ners[0])
        return sent
        # for ner in ners_to_masked:
        #     if len(ner)<=2 or ner.lower() in stopwords or ner not in text:
        #         continue

        #     text_masked = text.replace(ner, self.nlp.tokenizer.mask_token, 1)

        #     text = self.nlp(text_masked)[0]['sequence']
            # except:
            #     #print("fillinerror==>", text_masked)
            #     continue

            #text = text_masked.replace( self.nlp.tokenizer.mask_token, pred_tokens[0]['token_str'])

            # input_encode = self.tokenizer.encode(text_masked, return_tensors="pt", truncation=True).to(self.device) 
            # mask_token_index = torch.where(input_encode == self.tokenizer.mask_token_id)[1]
            # if mask_token_index.shape[0] == 0:
            #     continue
            # token_logits = self.model(input_encode).logits
            # mask_token_logits = token_logits[0, mask_token_index, :]

            # top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

            # ner_replace[ner] = self.tokenizer.decode(top_5_tokens[0])
            #print('{} to be replaced by filled in ==>{}'.format(ner, ner_replace[ner]) )

        # for ner, alter in ner_replace.items():
        #     text = text.replace(ner, alter)
        


# unit test



'''
nlp = pipeline("fill-mask" , model = 'distilbert-base-uncased')
content_mask = content.replace('White House', nlp.tokenizer.mask_token, 1)
nlp(content_mask)

[{'sequence': 'bush campaign trail stops at george washington - enmeshed in his re - election campaign, president bush rarely spends a full day at the white house. but even when he does, presidential politics and campaign pitches remain close at hand...',
  'score': 0.21938802301883698,
  'token': 2577,
  'token_str': 'george'},
 {'sequence': 'bush campaign trail stops at mount washington - enmeshed in his re - election campaign, president bush rarely spends a full day at the white house. but even when he does, presidential politics and campaign pitches remain close at hand...',
  'score': 0.2040329873561859,
  'token': 4057,
  'token_str': 'mount'},
 {'sequence': 'bush campaign trail stops at downtown washington - enmeshed in his re - election campaign, president bush rarely spends a full day at the white house. but even when he does, presidential politics and campaign pitches remain close at hand...',
  'score': 0.1190764307975769,
  'token': 5116,
  'token_str': 'downtown'},
 {'sequence': 'bush campaign trail stops at fort washington - enmeshed in his re - election campaign, president bush rarely spends a full day at the white house. but even when he does, presidential politics and campaign pitches remain close at hand...',
  'score': 0.05357954278588295,
  'token': 3481,
  'token_str': 'fort'},
 {'sequence': 'bush campaign trail stops at capitol washington - enmeshed in his re - election campaign, president bush rarely spends a full day at the white house. but even when he does, presidential politics and campaign pitches remain close at hand...',
  'score': 0.04831065610051155,
  'token': 9424,
  'token_str': 'capitol'}]
'''



'''
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("flair/ner-english-large")
model = AutoModelForTokenClassification.from_pretrained("flair/ner-english-large")

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









