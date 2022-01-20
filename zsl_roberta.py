import pandas as pd
import time,argparse
import os,math,itertools
import numpy as np
import re,operator,joblib
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity 
import joblib,gensim,transformers
assert gensim.__version__ == '4.1.2'

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str)
parser.add_argument("--fillm", default="roberta-large", type=str) # bert-base-uncased
parser.add_argument("--topk", default = 1024, type=int)
parser.add_argument("--top_k_seeds", default = 64, type=int)
parser.add_argument("--gpu", default="4", type=str)
args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

transformers.logging.set_verbosity_error()
#from utils.flair_ners import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

from utils.load_data import * 
from utils.transblock import * 
from utils.encoders import *

ds = load_data(dataset=args.dsn, samplecnt= 8)
ds.df_test['label_name'] = ds.df_test['label_name'].map(lambda x: x.lower())
labels_candidates = ds.df_test['label_name'].unique().tolist()
print(labels_candidates)

if args.dsn in ['nyt','yahoo']:
    ds, proper_len = process_ds(ds, 128)

ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))


import torch
device0 = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")


from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained(args.fillm,cache_dir="./cache",local_files_only=True) 
model = AutoModelWithLMHead.from_pretrained(args.fillm,cache_dir="./cache",local_files_only=True)

if args.topk == -1:
    top_k = len(tokenizer.vocab)
else:
    top_k = args.topk

nlp_fill = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0, top_k = top_k ) 
id_token = {ix:token for token, ix in tokenizer.vocab.items()}

stopwords = joblib.load("./utils/stopwords")
stopwords = set(stopwords)


from transformers.file_utils import cached_path
gram_diff = joblib.load("gram_diff___{}".format(args.dsn))
model_w2v = gensim.models.KeyedVectors.load_word2vec_format('./resource/GoogleNews-vectors-negative300.bin',binary=True)

BAG_OF_WORDS_ARCHIVE_MAP = {
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
    }

BASE_NLI ={
    'politics':['Politics','War', 'Election','Constitution','Democracy','Conflict','Military',\
                'Terrorism', 'Government', 'Ideology', 'fascism', 'Socialism', 'Totalitarian', 'Religion'],
    'law':      ['Law', 'Legitimacy','Court','Crime','Murder','Jurisdiction'],
    'science':  ['Science','Aerospace','Physics','Chemistry','Biology','Scientist','Astronomy','Universe','Big Bang'],
    'technology':['Technology','Biotech', 'IT','Computers','Internet','Algorithm','Space','Bitcoin','artificial Intelligence','Robot'],
    'health': ['Health','Healthcare','Medicine','Clinics','Vaccine','Wellness','Nutrition','Dental','HIV','Disease'],
    'business': ['Business','Finance','Oil price','Supply','Inflation','Dollars','Bank','Wall Street','Bitcoin',
                        'Federal Reserve','Accrual','Accountancy','Sluggishness','Consumerism','Trade','Quarterly earnings',\
                         'Deposit','Revenue','Stocks','Recapitalization','Marketing','Futures'],
    'sports': ['Sports','Athletics','Championships','Football','Olympic','Tournament','Chelsea','League','Golf',
                            'NFL','Super bowl','World Cup']
}

def get_s3_words(ll):
    filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[ll])
    with open(filepath, "r") as f:
        words = f.read().strip().split("\n")
    return words

def get_seed_words():
    vocab_w2v = set(list(model_w2v.index_to_key))
    label_expands_auto = {}
    for l, gram_scores in gram_diff.items():
        gram_scores_sum = {g:round(np.array(scores).sum(),4) for g, scores in gram_scores.items() }
        gram_scores_sum_sort = sorted(gram_scores_sum.items(), key=operator.itemgetter(1), reverse=True) 
        gram_scores_mean = {g:round(np.array(scores).mean(),4) for g, scores in gram_scores.items() }
        gram_scores_mean_sort = sorted(gram_scores_mean.items(), key=operator.itemgetter(1), reverse=True) 
        gram_scores_sort = gram_scores_sum_sort + gram_scores_mean_sort
        label_expands_auto[l] = set()
        for j in gram_scores_sort:
            if j[0] not in vocab_w2v or j[0] in ['news']:
                #print(j[0])
                continue
            if ' and ' in l:
                w0 = l.split('and')[0].strip().lower()
                w1 = l.split('and')[1].strip().lower()
                simi = max(model_w2v.similarity(w0, j[0]), model_w2v.similarity(w1, j[0]) )
            else:
                simi = model_w2v.similarity(l.lower(), j[0])
            if simi >= 0.1:
                label_expands_auto[l].add(j[0])
            if len(label_expands_auto[l]) == args.top_k_seeds:
                break 
        if ' and ' in l:
            label_expands_auto[l].add(l.split('and')[0].strip())
            label_expands_auto[l].add(l.split('and')[1].strip())
        else:
            label_expands_auto[l].add(l)
        for ll in BAG_OF_WORDS_ARCHIVE_MAP:
            if (ll in l.lower()) or ('world' in l.lower() and ll == 'politics') or \
                (('science' in l.lower() or 'technology' in l.lower()) and ll == 'space'):
                words_s3 = get_s3_words(ll)
                label_expands_auto[l].update(words_s3)
        for ll, expands in BASE_NLI.items():
            if ll in l.lower():
                label_expands_auto[l].update( [w.lower() for w in expands ] )
        print(l, label_expands_auto[l], '\n')
    return label_expands_auto


label_expands_auto = get_seed_words()
print(label_expands_auto)

label_expands_auto_lower = {}
for l, seeds in label_expands_auto.items():
    label_expands_auto_lower[l.lower()] = set([w.lower() for w in list(seeds)])


# from sklearn.metrics.pairwise import cosine_distances,cosine_similarity
# enc = encoder('cmlm-base')


# label_expands_embed = {}
# for l, seeds in label_expands_auto.items():
#     embeds = enc.infer(list(seeds))
#     label_expands_embed[l.lower()] = embeds
#     print(l, embeds.shape)



acc_base = []
acc_embed = []
acc_kpt = []
for ix, row in ds.df_test.reset_index().iterrows(): 

    template1 = "{}. This News is about {}".format(row['content'], nlp_fill.tokenizer.mask_token)
    template2 = "{} News: {}.".format(nlp_fill.tokenizer.mask_token, row['content'])
    template3 = "[Category: {} ] {}.".format(nlp_fill.tokenizer.mask_token, row['content'])

    ls = {l:0 for l in labels_candidates}
    ls_embed = {l:0 for l in labels_candidates}
    ls_kpt = {l:[] for l in labels_candidates}

    filled_results = nlp_fill([template1, template2, template3])
    
    for filled_result in filled_results:

        tokens, scores = [], []
        for r in filled_result :
            token = r['token_str'].lower().strip()
            if token  in stopwords or token in string.punctuation or token.isdigit() :
                continue
            tokens.append(token)
            scores.append(r['score'])

            # base
            if token in labels_candidates:
                ls[token] += r['score']
            if token in ['science', 'technology']:
                ls['science and technology'] += r['score']


            for l, seeds in label_expands_auto_lower.items():
                if token in seeds:
                    ls_kpt[l].append(r['score'])

    ls_kpt_reduce = {l:sum(scores)/len(scores) for l, scores in ls_kpt.items()}
        # tokens_embeds = enc.infer(tokens)

        
        # for l, embeds in label_expands_embed.items():
        #     tokens_simis = cosine_similarity(tokens_embeds, embeds)
        #     l_scores = tokens_simis.mean(axis=1) * np.array(scores)
        #     ls_embed[l] += l_scores.mean()
            


    if max(ls, key=ls.get) == row['label_name']:
        acc_base.append(1)
    else:
        acc_base.append(0)

    if max(ls_kpt_reduce, key=ls_kpt_reduce.get) == row['label_name']:
        acc_kpt.append(1)
    else:
        acc_kpt.append(0)


    if ix > 0 and ix % 20 == 0:
        print(ix, sum(acc_base) / len(acc_base), sum(acc_kpt) / len(acc_kpt) )

print(ix, sum(acc_base) / len(acc_base), sum(acc_kpt) / len(acc_kpt) )














