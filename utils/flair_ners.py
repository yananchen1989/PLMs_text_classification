import pandas as pd
import time
from flair.data import Sentence
from flair.models import SequenceTagger

import torch, flair
flair.device = torch.device('cpu')
#flair.device = torch.device('cuda:0')
#  note that get_ners can only run one single GPU !!! 
tagger = SequenceTagger.load("flair/ner-english-fast")


text = '''
Reuters/Vincest West Lucasfilm is eyeing a date for an Obi-Wan Kenobi solo.\nLucasfilm is already eyeing a production date for an Obi-Wan Kenobi solo film.\nThere are plans to create a solo debut for Obi-Wan Kenobi from the "Star Wars" franchise, and according to Omega Underground, Lucasfilm Productions is setting a film date in January 2019. The set location is expected to be at Pinewood Studios in the United Kingdom.\nMeanwhile, the anticipated Obi-Wan debut also has a working title, which is "Joshua Tree." The working title offers hints that the film will feature scenes when Obi-Wan was watching over a young Luke Skywalker in Tatooine.\nAccording to Screenrant, the story of Obi-Wan in his solo film could take place between the saga\'s episode 3, "Star Wars: Revenge of the Sith," and episode 4, "A New Hope." However, there might be difficulty in finding the perfect timeline for the Obi-Wan solo.\nObi-Wan\'s adventures have been featured in many other platforms aside from the "Star Wars" saga. His character and adventures already appeared in the animated "Star Wars: The Clone Wars," as well as the "Star Wars Rebels" TV series. Hopefully the upcoming Obi-Wan spin-off won\'t collide with other stories from the "Star Wars" franchise.\nThere are speculations that Stephen Daldry, director of "The Hours" and "The Reader" is currently in talks to spearhead the upcoming Obi-Wan solo film.\nAlthough it is yet to be confirmed, fans are already expecting Ewan McGregor to fulfil the role of Obi-Wan in his solo debut. There are talks that McGregor was not given the opportunity to embody the character of Obi-Wan in the prequel trilogy of "Star Wars," and that reprising the role in a solo would give him that chance.\nAside from that, fans won\'t be happy if Lucasfilm hires another actor to play Obi-Wan, since he is the original portrayer.\nThe director and screenwriter for the Obi-Wan Kenobi solo film are yet to be confirmed.
'''

def get_ners(text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    ners = list(set([ii['text'] for ii in sentence.to_dict(tag_type='ner')['entities']]))
    return ners







'''
from transformers import pipeline
# t5
from transformers import T5Tokenizer, AutoModelWithLMHead
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
print(tokenizer_t5)

t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)

# ft_model_path = 'ft_model_{}_{}'.format('t5', 'ep')
# checkpoint_files = glob.glob(ft_model_path+"/checkpoint_loss_*")
# list.sort(checkpoint_files)
# t5 = AutoModelWithLMHead.from_pretrained(checkpoint_files[0])  

gen_nlp  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=len(gpus)-1)
sep = tokenizer_t5.eos_token

# gpt2
from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#gpt2 = GPT2LMHeadModel.from_pretrained('ft_model_ft_ep')

gpt2.trainable = False
gpt2.config.pad_token_id = 50256
gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=0, return_full_text=False)

sep = tokenizer_gpt2.sep_token
'''



