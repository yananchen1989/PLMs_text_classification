
from load_data import *
from transformers import pipeline
import random,torch
print(torch.__version__)
ds = load_data(dataset='ag', samplecnt=32)


nlp = pipeline("fill-mask" , model = 'distilbert-base-cased', device=0)


text_mask = text.replace('Fassa Bortolo', nlp.tokenizer.mask_token, 1)

pred_tokens = nlp(text_mask)

text_mask.replace( nlp.tokenizer.mask_token, pred_tokens[0]['token_str'])


ds.df_train.sample(10000)['content'].map(lambda x: augmentor.augment(x)) 

augmentor = fillInmask(ner_set= 0 )




import os 

from transformers import pipeline

from transformers import AutoModelWithLMHead, AutoTokenizer
lang = 'zh'
model = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(lang), cache_dir="./cache")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(lang))
nlp = pipeline("translation_en_to_{}".format(lang), model=model, tokenizer=tokenizer, device=0)

results = nlp(ds.df.sample(1000)['content'].tolist(), max_length=128, do_sample=False)
print(results[0]['translation_text'])



nlp = pipeline("ner", model="dslim/bert-base-NER")
content = random.sample(contents,1)[0]
print(content)
nlp(content)




contents = ds.df_train['content'].tolist()[:50]

nlp  = pipeline("text-generation", model='gpt2', device=0, return_full_text=False)
results = nlp(['safe conduct , however ambitious and well-intentioned , fails to hit the entertainment bull s - eye .'], max_length=100, do_sample=True, top_p=0.9, top_k=0, \
                    repetition_penalty=1, num_return_sequences=16)

for ii in results:
    print('==>', ii['generated_text'])






import nltk; nltk.download('wordnet')
from eda import *



parser = argparse.ArgumentParser()


#generate more data with standard augmentation
sentence = "safe conduct , however ambitious and well-intentioned , fails to hit the entertainment bull s - eye ."
aug_sentences = eda(sentence, alpha_sr=args.eda_sr, alpha_ri=args.eda_ri, \
                    alpha_rs=args.eda_rs, p_rd=args.eda_rd, num_aug=args.eda_times)






