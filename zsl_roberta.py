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
parser.add_argument("--dsn", default="yahoo", type=str)
parser.add_argument("--fillm", default="roberta-large", type=str) # bert-base-uncased
parser.add_argument("--topk", default = 1024, type=int)
parser.add_argument("--gpu", default="0", type=str)
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
ds.df_test['label_name'] = ds.df_test['label_name']
labels_candidates = ds.df_test['label_name'].unique().tolist()
print(labels_candidates)

if args.dsn in ['nyt','yahoo']:
    ds, proper_len = process_ds(ds, 400, True)

ds.df_test['content'] = ds.df_test['content'].map(lambda x: remove_str(x))


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



def zsl_roberta(row, labels_candidates):

    template1 = "{}. This News is about {}".format(row['content'], nlp_fill.tokenizer.mask_token)
    template2 = "{} News: {}.".format(nlp_fill.tokenizer.mask_token, row['content'])
    template3 = "[Category: {} ] {}.".format(nlp_fill.tokenizer.mask_token, row['content'])

    ls = {l:0 for l in labels_candidates}

    filled_results = nlp_fill([template1, template2, template3])
    
    for filled_result in filled_results:

        for r in filled_result :
            token = r['token_str'].lower().strip()

            if token  in stopwords or token in string.punctuation or token.isdigit() :
                continue

            for l in ls.keys():
                if token in l.lower():
                    ls[l] += r['score']  

    df_noexpand = pd.DataFrame(ls.items(), columns=['label','score_noexpand'])
    return df_noexpand





acc_base = []
for ix, row in ds.df_test.reset_index().iterrows(): 

    df_noexpand = zsl_roberta(row, labels_candidates)

    pred_label_name = df_noexpand.sort_values(by=['score_noexpand'], ascending=False).head(1)['label'].tolist()[0]


    if pred_label_name == row['label_name']:
        acc_base.append(1)
    else:
        acc_base.append(0)

    if ix > 0 and ix % 20 == 0:
        print(ix, sum(acc_base) / len(acc_base) )















