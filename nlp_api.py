
from load_data import *
from transformers import pipeline
import random,torch
ds = load_data(dataset='ag', samplecnt=-1)
contents = ds.df.sample(10000)['content'].tolist()
content = contents[1]
print(content)


nlp  = pipeline("text-generation", model='gpt2', device=0, return_full_text=False)
results = nlp(ds.df_train['content'].tolist(), max_length=250, do_sample=True, top_p=0.9, top_k=0, \
           repetition_penalty=1, num_return_sequences=16)




nlp = pipeline("fill-mask" , model = 'distilbert-base-cased', device=0)


text_mask = text.replace('Fassa Bortolo', nlp.tokenizer.mask_token, 1)

pred_tokens = nlp(text_mask)

text_mask.replace( nlp.tokenizer.mask_token, pred_tokens[0]['token_str'])


ds.df_train.sample(10000)['content'].map(lambda x: augmentor.augment(x)) 

augmentor = fillInmask(ner_set= 0 )







from transformers import AutoModelWithLMHead, AutoTokenizer

model = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
nlp = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer, device=0)

nlp(ds.df_train.sample(12)['content'].tolist(), max_length=100)






nlp = pipeline("ner", model="dslim/bert-base-NER")
content = random.sample(contents,1)[0]
print(content)
nlp(content)

