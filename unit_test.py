import time,os,argparse
from load_data import *
from transformers import pipeline
import random,torch
print(torch.__version__)
from transblock import * 
from dpp_model import * 

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


for ite in range(10):
    ds = load_data(dataset='ag', samplecnt=-1)
    ds.df_train_aug = ds.df_train
    (x_train, y_train),  (x_test, y_test), num_classes = get_keras_data(ds.df_train_aug, ds.df_test)

    #model = get_model_transormer(num_classes)
    model = get_model_bert(num_classes, 'albert')

    batch_size = 64

    history = model.fit(
        x_train, y_train, batch_size=32, epochs=50, \
        validation_batch_size=64,
        validation_data=(x_test, y_test), verbose=0,
        callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
    )
    best_val_acc = max(history.history['val_acc'])
    print(best_val_acc)


ds = load_data(dataset='ag', samplecnt=-1)


nlp  = pipeline("text-generation", model='gpt2', device=-1, return_full_text=False)

max_len = get_tokens_len(ds, 0.99)
results_trunk = nlp([content], max_length=max_len, do_sample=True, top_p=0.9, top_k=0, \
                    repetition_penalty=1, num_return_sequences=256)


enc = encoder('dan')

ori_sentence = ds.df_test.sample(1)['content'].tolist()[0]

ori_embed = enc.infer([ori_sentence])
syn_sentences = [sent['generated_text'] for sent in results_trunk]
syn_embeds = enc.infer(syn_sentences)
simis = cosine_similarity(ori_embed, syn_embeds)
df_simi = pd.DataFrame(zip(syn_sentences, simis[0]), columns=['content','simi'])
df_simi.sort_values(by=['simi'], ascending=False, inplace=True)
df_simi_filer = df_simi.loc[df_simi['simi']>= 0.6]


embeds = enc.infer(df_simi_filer['content'].tolist())

sorted_ixs = extract_ix_dpp(embeds, df_simi_filer['simi'].values)
df_simi_filer_dpp = df_simi_filer.reset_index().iloc[sorted_ixs]













tokenizer.tokenize(content)

content = ds.df_test['content'].tolist()[10]


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

