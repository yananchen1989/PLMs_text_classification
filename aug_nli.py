import sys,os,logging,glob,pickle,torch,random,gc
import numpy as np
import pandas as pd 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import datetime
def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
logging.Formatter.converter = beijing

logging.basicConfig(filename= 'aug_nli.log',
    filemode='w',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()


from transformers import AutoModelForSequenceClassification, AutoTokenizer
#https://huggingface.co/joeddav/bart-large-mnli-yahoo-answers
#MODEL = "joeddav/bart-large-mnli-yahoo-answers"
# 'facebook/bart-large-mnli'  'joeddav/xlm-roberta-large-xnli'
# from transformers import BartForSequenceClassification, BartTokenizer
# tokenizer = BartTokenizer.from_pretrained(MODEL)
# nli_model = BartForSequenceClassification.from_pretrained(MODEL)
# nli_model.to(device)

def get_sentences():
    df = pd.read_csv("/root/yanan/berts/datasets_aug/cnn_dailymail_stories.csv").sample(frac=1)
    sentences = []
    for content in df['content'].tolist():
        sentences.extend([s for s in content.split('\n') if len(s.strip().split(' ')) >= 50])
    return sentences

sentences = get_sentences()
random.shuffle(sentences) # 703351


#batch_size = 32
labels_candidate = ['Sports', 'Business', 'Science', 'Technology','Entertainment','Politics','Society',\
             'Culture', 'Mathematics','Health','Education','Reference','Computers','Internet',\
             'Finance','Music','Family','Relationships','Government']

'''
infos = []
ix = 0
while ix < len(sentences):
    premises = sentences[ix:ix+batch_size]

    for cate in labels:
        hypothesises = ['This text is about {}.'.format(cate)] * batch_size
        # run through model pre-trained on MNLI
        x = tokenizer(premises, hypothesises, return_tensors='pt', padding=True ,truncation=True,max_length=64)

        logits = nli_model(**x.to(device))[0] 
        entail_contradiction_logits = logits[:,[0,2]].softmax(dim=1)
        true_prob = entail_contradiction_logits[:,1].cpu().detach().numpy()
        
        if true_prob.max() >= 0.9: 
            pos = np.where(true_prob>=0.9)[0]
            for p in pos:
                #print(premises[p])
                infos.append((premises[p], cate))
    ix += batch_size
    logger.info('ix==>{}'.format(ix))

    gc.collect()


'''

from transformers import pipeline
nlp = pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers", device=0) #  

#nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# 

#content = "Johnson Helps D-Backs End Nine-Game Slide (AP) AP - Randy Johnson took a four-hitter into the ninth inning to help the Arizona Diamondbacks end a nine-game losing streak Sunday, beating Steve Trachsel and the New York Mets 2-0."
# generate selected samples which belong to the defined categories
import csv
save_train_file = open('cnn_dm_nli.csv', 'w')
writer = csv.writer(save_train_file, delimiter='\t')

for content in sentences:
    result = nlp(content, labels_candidate, multi_label=False, hypothesis_template="This text is about {}.")

    if result['scores'][0] < 0.4:
        continue
    writer.writerow([content.strip().replace('\t',' '), result['labels'][0]])



#### reload the nli classify results and inject into the original dataset
import pandas as pd 
def acquire_nli_for_aug():
    df_nli = pd.read_csv("../datasets_aug/cnn_dm_nli.csv", sep='\t', header=None)
    df_nli.columns = ['content','label']

    cates = []
    with open('../datasets_aug/yahoo_news/classes.txt','r') as f:
        for line in f:
            cates.append(line.strip())

    cate_ix = {cate:ix+1 for ix, cate in enumerate(cates)}

    cate_map = {}
    for cate in df_nli.label.unique():
        for cate_ in cate_ix.keys():
            if cate == 'Technology':
                cate_map[cate] = 'Computers & Internet'
            if cate in cate_:
                cate_map[cate] = cate_ 
                break

    df_nli['label'] = df_nli['label'].map(lambda x: cate_map[x]).map(lambda x: cate_ix[x])
    return df_nli
df_nli = acquire_nli_for_aug()
from load_data import * 
from transblock import *  

for ds in 
for i in range(5):
    ds = load_data(dataset='ag', samplecnt=1000)
    # inject the aug data 
    print('no nli')
    (train_x,train_y),  (test_x, test_y), num_classes = get_keras_data(ds.df_train,  ds.df_test)
    model = get_model_transormer(num_classes)
    history = model.fit(
        train_x,train_y, batch_size=32, epochs=100, validation_data=(test_x, test_y),verbose=1,
        callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
    )
    best_val_acc = max(history.history['val_acc'])
    print(' no nli acc:', best_val_acc)





with tf.device('/GPU:0'):
    for i in range(3):
        print('has nli')
        ds = load_data(dataset='yahoo', samplecnt=1000)
        ds.df_train = ds.df_train.append(df_nli)
        (train_x,train_y),  (test_x, test_y), num_classes = get_keras_data(ds.df_train,  ds.df_test)

        model = get_model_albert(num_classes)
        history = model.fit(
            train_x,train_y, batch_size=32, epochs=100, validation_data=(test_x, test_y),verbose=1,
            callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
        )
        best_val_acc = max(history.history['val_acc'])
        print('has nli acc:', best_val_acc)





