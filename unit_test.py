
from load_data import *
from transformers import pipeline
import random,torch
ds = load_data(dataset='nyt', samplecnt=-1)


nlp = pipeline("fill-mask" , model = 'distilbert-base-cased', device=0)


text_mask = text.replace('Fassa Bortolo', nlp.tokenizer.mask_token, 1)

pred_tokens = nlp(text_mask)

text_mask.replace( nlp.tokenizer.mask_token, pred_tokens[0]['token_str'])


ds.df_train.sample(10000)['content'].map(lambda x: augmentor.augment(x)) 

augmentor = fillInmask(ner_set= 0 )






from transformers import pipeline

from transformers import AutoModelWithLMHead, AutoTokenizer
lang = 'zh'
model = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(lang))
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(lang))
nlp = pipeline("translation_en_to_{}".format(lang), model=model, tokenizer=tokenizer, device=0)

results = nlp(ds.df.sample(1000)['content'].tolist(), max_length=128, do_sample=False)
print(results[0]['translation_text'])



nlp = pipeline("ner", model="dslim/bert-base-NER")
content = random.sample(contents,1)[0]
print(content)
nlp(content)
















