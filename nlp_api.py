
from load_data import *
from transformers import pipeline
import random,torch
ds = load_data(dataset='ag', samplecnt=-1)
contents = ds.df.sample(10000)['content'].tolist()
content = contents[1]
print(content)


nlp  = pipeline("text-generation", model='gpt2', device=0, return_full_text=False)
results = nlp(contents, max_length=250, do_sample=True, top_p=0.9, top_k=0, \
           repetition_penalty=1, num_return_sequences=64)




nlp = pipeline("fill-mask" , model = 'distilbert-base-uncased', device=0)

for content in contents:
    tokens = content.split(' ')
    content_mask = content.replace(random.sample(tokens,1)[0], nlp.tokenizer.mask_token, 1)
    nlp(content_mask)




nlp = pipeline("translation" , model = "Helsinki-NLP/opus-mt-en-fr")
nlp(content)

nlp = pipeline("ner", model="dslim/bert-base-NER")


