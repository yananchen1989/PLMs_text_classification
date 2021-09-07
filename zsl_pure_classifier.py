import os 
import numpy as np 
import pandas as pd 
from utils.load_data import * 
from utils.transblock import * 
from transformers import pipeline
import argparse,torch
#nlp = pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers", device=0) #  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--nlim", default="facebook/bart-large-mnli", type=str, \
    choices=["joeddav/bart-large-mnli-yahoo-answers", "facebook/bart-large-mnli", 'joeddav/xlm-roberta-large-xnli'])
parser.add_argument("--dsn", default="ag", type=str)
args = parser.parse_args()

ds = load_data(dataset=args.dsn, samplecnt=-1)

ds, proper_len = process_ds(ds, 256)
# kw_extractor = yake.KeywordExtractor(lan="en", top=3, n=2)
# text = '''
# 'Russia School Standoff Ends With 250 Dead BESLAN, Russia - The three-day hostage siege at a school in southern Russia ended in chaos and bloodshed Friday, after witnesses said Chechen militants set off bombs and Russian commandos stormed the building. Hostages fled in terror, many of them children who were half-naked and covered in blood...'
# '''
# keywords = kw_extractor.extract_keywords(text)

# if 'bart' in args.nlim:
#     from transformers import BartForSequenceClassification, BartTokenizer
#     tokenizer = BartTokenizer.from_pretrained(args.nlim, cache_dir="./buffer", local_files_only=False)
#     nli_model = BartForSequenceClassification.from_pretrained(args.nlim, cache_dir="./buffer", local_files_only=False)

# elif 'robert' in args.nlim:
#     from transformers import AutoModelForSequenceClassification, AutoTokenizer
#     nli_model = AutoModelForSequenceClassification.from_pretrained(args.nlim,cache_dir="./buffer", local_files_only=False)
#     tokenizer = AutoTokenizer.from_pretrained(args.nlim,cache_dir="./buffer", local_files_only=False)

# nli_model.to(device)

nli_nlp = pipeline("zero-shot-classification", model=args.nlim, device=0) #  1.8.1+cu102



#labels_candidates = list(ds.df_test['label_name'].unique())
import gensim,operator
from nltk.stem.porter import *
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

stemmer = PorterStemmer()
stemmer.stem('political')
model_google = gensim.models.KeyedVectors.load_word2vec_format('./torch_ds/GoogleNews-vectors-negative300.bin',binary=True)


# v-w2v
ln_extend = {}
for ln in ds.df_train['label_name'].unique():
    subln =  []
    if ln == 'World':
        ln = ' '.join(['Politics','War','Military','Terrorism','Election','Finance',\
                                  'Crime','Murder','Religion','Jurisdiction', 'Democracy'])
    sub_labels = ln.split(' ')
    print(sub_labels)
    for sub in sub_labels:
        if sub in stopwords:
            continue
        simi_words = model_google.most_similar(sub, topn=50)
        simi_words = [' '.join(ii[0].split('_')) for ii in simi_words if stemmer.stem(sub) != stemmer.stem(ii[0]) and ii[0] not in stopwords]
        if sub not in simi_words:
            simi_words.append(sub)
        subln.extend(simi_words)
    ln_extend[ln] = subln # sub_labels



# v1
ln_extend = {}
for l in ds.df_test['label_name'].unique():
    ln_extend[l] = expand_label_nli[l]

# ori
#ln_extend = {l:[l] for l in ds.df_test['label_name'].unique()}

ln_extend__rev = {}
for i, j in ln_extend.items():
    for jj in j:
        ln_extend__rev[jj] = i

labels_candidates = set()
for v in ln_extend.values():
    labels_candidates.update(v)

import operator
accs = []
for ix, row in ds.df_test.sample(1000).iterrows():
    sent = row['content']
    label_name = row['label_name']

    result = nli_nlp(sent,  list(labels_candidates), multi_label=True, hypothesis_template="This text is about {}.")
    accum_label = {l:[] for l in ds.df_test['label_name'].unique()}
  
    for l, score in zip(result['labels'], result['scores']):
        accum_label[ln_extend__rev[l]].append(score)

    for l in accum_label.keys():
        accum_label[l] = sum(accum_label[l]) / len(accum_label[l])

    pred_label = max(accum_label.items(), key=operator.itemgetter(1))[0]

    score = accum_label[pred_label]


    if pred_label == row['label_name']:
        accs.append(1)
    else:
        accs.append(0)

    if len(accs) % 50 ==0:
        print('nli acc:', sum(accs) / len(accs))




'''
ds.df_test['hypo'] = ds.df_test['label_name'].map(lambda x: 'This text is about {}.'.format(x))


thres = 0.6
results = []
ix = 0
while ix < ds.df_test.shape[0]:
    dfs = ds.df_test[ix:ix+32]

    premises = dfs['content'].tolist()
    hypothesises = dfs['hypo'].tolist()

    # run through model pre-trained on MNLI
    x = tokenizer(premises, hypothesises, return_tensors='pt', padding=True ,truncation=True,max_length=128)

    logits = nli_model(**x.to(device))[0] 
    entail_contradiction_logits = logits[:,[0,2]].softmax(dim=1)
    true_prob = entail_contradiction_logits[:,1].cpu().detach().numpy()
    results.extend(list(true_prob))
    
    for ii in zip(dfs['label_name'].tolist(), list(true_prob), dfs['content'].tolist()):
        print(ii[0], ii[1], ii[2])
    #results.extend([1 if ii>=thres else 0 for ii in true_prob ])
    ix += 32
    if ix % 1024 == 0:
        print(ix)

print('summary==>', args.nlim, args.dsn,  np.array(results).mean() )
'''


# python zsl_pure_classifier.py --dsn yelp2 --nlim joeddav/bart-large-mnli-yahoo-answers


