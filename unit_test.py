import time,os,argparse
from load_data import *
from transformers import pipeline
import random,torch
print(torch.__version__)
from transblock import * 

def do_train_test(ds):

    (x_train, y_train),  (x_test, y_test), num_classes = get_keras_data(ds.df_train_aug, ds.df_test)

    model = get_model_transormer(num_classes)

    batch_size = 8

    history = model.fit(
        x_train, y_train, batch_size=batch_size, epochs=50, \
        validation_batch_size=64,
        validation_data=(x_test, y_test), verbose=1,
        callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
    )

    best_val_acc = max(history.history['val_acc'])
    return round(best_val_acc, 4)

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="", type=str)
args = parser.parse_args()
print('args==>', args)

for samplecnt in [32, 64, 128, 256, 512, 1024, 2048]:
    accs = []
    for ite in range(5):
        ds = load_data(dataset=args.dsn, samplecnt=samplecnt)
        ds.df_train_aug = ds.df_train
        best_val_acc_noaug = do_train_test(ds)
        accs.append(best_val_acc_noaug)
    print('summary==> dsn==> {}'.format(args.dsn), 'samplecnt==> {}'.format(samplecnt),\
       'std==> {}'.format(round(np.array(accs).std(), 4) ), 'accs==> {}'.format(' '.join([str(i) for i in accs]) ) )




ds = load_data(dataset='ag', samplecnt=-1)
nlp  = pipeline("text-generation", model='gpt2', device=-1, return_full_text=False)

content = ds.df_test.sample(1)['content'].tolist()[0]
print(content)
results = nlp([content], max_length=120, do_sample=True, top_p=0.9, top_k=0, \
                    repetition_penalty=1, num_return_sequences=32)



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

