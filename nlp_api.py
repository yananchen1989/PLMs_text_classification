
from load_data import *
from transformers import pipeline
import random,torch
ds = load_data(dataset='ag', samplecnt=-1)
contents = ds.df.sample(10000)['content'].tolist()
content = contents[1]
print(content)


nlp  = pipeline("text-generation", model='gpt2', device=0, return_full_text=False)
results = nlp(ds.df_train['content'].tolist(), max_length=250, do_sample=True, top_p=0.9, top_k=0, \
           repetition_penalty=1, num_return_sequences=64)




nlp = pipeline("fill-mask" , model = 'distilbert-base-cased', device=0)


text_mask = text.replace('Fassa Bortolo', nlp.tokenizer.mask_token, 1)

pred_tokens = nlp(text_mask)

text_mask.replace( nlp.tokenizer.mask_token, pred_tokens[0]['token_str'])


ds.df_train.sample(10000)['content'].map(lambda x: augmentor.augment(x)) 

augmentor = fillInmask(ner_set= 0 )


sentences = []
for text in ds.df_train.sample(10000)['content'].tolist(): 
    ners_to_masked = augmentor.get_random_span(text)
    if not ners_to_masked:
        print(text)
        continue
    text_masked = text.replace(ners_to_masked[0], augmentor.nlp.tokenizer.mask_token, 1)
    sentences.append(text_masked)


results = nlp(sentences)






ners_to_masked = augmentor.get_ners(text)
for ner in ners_to_masked:
    if len(ner)<=2 or ner.lower() in stopwords:
        continue
    #text_masked = text.replace(ner, self.tokenizer.mask_token)
    text_masked = text.replace(ner, augmentor.nlp.tokenizer.mask_token, 1)

    pred_tokens = augmentor.nlp(text_masked)

    text = text_masked.replace( augmentor.nlp.tokenizer.mask_token, pred_tokens[0]['token_str'])



nlp = pipeline("translation" , model = "Helsinki-NLP/opus-mt-en-fr")
nlp(ds.df_train.sample(256)['content'].tolist())






nlp = pipeline("ner", model="dslim/bert-base-NER")
content = random.sample(contents,1)[0]
print(content)
nlp(content)

