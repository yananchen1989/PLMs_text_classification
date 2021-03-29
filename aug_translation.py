import torch
import sys,os,logging,argparse,gc
import numpy as np
import pandas as pd 

from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class backTranslate():
    def __init__(self, lang='de'):
        self.lang = lang
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.lang not in ['ru','de','zh','fr']:
            raise KeyError("language not supported!")
        self.load_model()

    def load_model(self):
        # if self.lang in ['ru', 'de']:
        #     self.tokenizer_forward = FSMTTokenizer.from_pretrained("facebook/wmt19-en-{}".format(self.lang))
        #     self.model_forward = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-{}".format(self.lang))
        #     self.tokenizer_backward = FSMTTokenizer.from_pretrained("facebook/wmt19-{}-en".format(self.lang))
        #     self.model_backward = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-{}-en".format(self.lang))
        # #elif self.lang in ['zh','fr']:
        self.tokenizer_backward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-{}-en".format(self.lang))
        self.model_backward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-{}-en".format(self.lang))
        self.tokenizer_forward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(self.lang))
        self.model_forward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(self.lang))
        
        self.model_forward.to(self.device)
        self.model_backward.to(self.device)

    # def translate_batch(self, contents, tokenizer, model):
    #     # Tokenize the text
    #     batch = tokenizer.prepare_seq2seq_batch(src_texts=contents, return_tensors="pt").to(device) 
    #     # Perform the translation and decode the output
    #     translation = model.generate(**batch)
    #     target_contents = tokenizer.batch_decode(translation, skip_special_tokens=True)
    #     return target_contents

    def translate_map_forward(self, content):
        input_ids = self.tokenizer_forward.encode(content, return_tensors="pt").to(self.device) 
        outputs = self.model_forward.generate(input_ids)
        decoded = self.tokenizer_forward.decode(outputs[0], skip_special_tokens=True)
        return decoded

    def translate_map_backward(self, content):
        input_ids = self.tokenizer_backward.encode(content, return_tensors="pt").to(self.device) 
        outputs = self.model_backward.generate(input_ids)
        decoded = self.tokenizer_backward.decode(outputs[0], skip_special_tokens=True)
        return decoded
    def augment(self, content):
        content_ = self.translate_map_forward(content)
        content__ = self.translate_map_backward(content_)
        return content__




# unit test
'''
augmentor = backTranslate(lang='zh')

content = "Microsoft has said it will replace more than 14 million power cables for its Xbox consoles due to safety concerns."

content = """
In a statement, it added: "In almost all instances, any damage caused by these failures was contained within the console itself or \
limited to the tip of the power cord at the back of the console." But in seven cases, \
customers reported sustaining a minor burn to their hand. In 23 cases, \
customers reported smoke damage, or minor damage to a carpet or entertainment centre. \
"This is a preventative step we're choosing to take despite the rarity of these incidents,"\
 said Robbie Bach, senior vice president, Microsoft home and entertainment division. \
 "We regret the inconvenience, but believe offering consumers a free replacement \
 cord is the responsible thing to do." Consumers can order a new cable from the Xbox \
 website or by telephoning 0800 028 9276 in the UK. Microsoft said customers \
 would get replacement cords within two to four weeks from the time of order. \
 It advised users to turn off their Xboxes when not in use. \
 A follow-up to Xbox is expected to released at the end of this year or the beginning of 2006.
"""

content = "Consumers can order a new cable from the Xbox website or by telephoning 0800 028 9276 in the UK. Microsoft said customers would get replacement cords within two to four weeks from the time of order. It advised users to turn off their Xboxes when not in use. A follow-up to Xbox is expected to released at the end of this year or the beginning of 2006."

augmentor.augment(content)
'''

'''
bacth_size = 128
generated_contents = []
ix = 0
while ix < df_train.shape[0]:
    contents = df_train[ix:ix+bacth_size]['content'].tolist()
    target_contents = translate_batch(contents, tokenizer_forward, model_forward)
    back_contents = translate_batch(target_contents, tokenizer_backward, model_backward)
    generated_contents.extend(back_contents)
    ix += bacth_size
    print(ix)
    gc.collect()

assert len(generated_contents) == df_train.shape[0]
'''







