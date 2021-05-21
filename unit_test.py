import time,os
from load_data import *
from transformers import pipeline
import random,torch
print(torch.__version__)


ds = load_data(dataset='ag', samplecnt=-1)

for dsn in ['stsa']:
    for args.samplecnt in [1024, 2048]:
        accs = []
        for ite in range(5):
            ds = load_data(dataset=dsn, samplecnt=args.samplecnt)

            ds.df_train_aug = ds.df_train
            best_val_acc_noaug = do_train_test(ds)
            accs.append(best_val_acc_noaug)
        print('summary==> dsn==> {}'.format(dsn), 'samplecnt==> {}'.format(args.samplecnt),\
           'std==> {}'.format(round(np.array(accs).std(), 4) ), 'accs==> {}'.format(' '.join([str(i) for i in accs]) ) )










nlp = pipeline("fill-mask" , model = 'distilbert-base-cased', device=0)


text_mask = text.replace('Fassa Bortolo', nlp.tokenizer.mask_token, 1)

pred_tokens = nlp(text_mask)

text_mask.replace( nlp.tokenizer.mask_token, pred_tokens[0]['token_str'])


ds.df_train.sample(10000)['content'].map(lambda x: augmentor.augment(x)) 

augmentor = fillInmask(ner_set= 0 )






results = nlp(ds.df.sample(1000)['content'].tolist(), max_length=128, do_sample=False)
print(results[0]['translation_text'])





nlp = pipeline("ner", model="dslim/bert-base-NER")
content = random.sample(contents,1)[0]
print(content)
nlp(content)





content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''




tokenizer_backward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-{}-en".format(lang), cache_dir="./cache")
model_backward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-{}-en".format(lang), cache_dir="./cache")
tokenizer_forward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(lang), cache_dir="./cache")
model_forward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(lang), cache_dir="./cache")
if torch.cuda.is_available():
    nlp_backward = pipeline("translation", model=model_backward, tokenizer=tokenizer_backward, device=0)
    nlp_forward = pipeline("translation", model=model_forward, tokenizer=tokenizer_forward, device=0)





content_ =  nlp_forward(ds.df.sample(64)['content'].tolist(), do_sample=True, temperature=0.9, num_return_sequences=1)





# https://github.com/GT-SALT/MixText/blob/master/data/yahoo_answers_csv/back_translate.ipynb
import torch
while 1:
    en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
    ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')


en2ru.cuda()
ru2en.cuda()

en2de.cuda()
de2en.cuda()

ru2en.translate(en2ru.translate(content,  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)

