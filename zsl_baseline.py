import pandas as pd
import time,argparse
import os,math,itertools
import numpy as np
import re,operator,joblib
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity 
import joblib,gensim
assert gensim.__version__ == '4.1.2'

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str)
parser.add_argument("--fbs_gpt", default=256, type=int)
parser.add_argument("--fbs_para", default=32, type=int)
parser.add_argument("--acc_topn", default=1, type=int)
parser.add_argument("--topk", default=64, type=int)
parser.add_argument("--nli_ensure", default=0, type=int)
parser.add_argument("--expand", default='gpt', type=str)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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

if args.dsn == 'nyt':
    samplecnt = 256
else:
    samplecnt = 2048
ds = load_data(dataset=args.dsn, samplecnt= samplecnt)
labels_candidates = ds.df_train['label_name'].unique().tolist()
print(labels_candidates)

if args.dsn in ['nyt','yahoo']:
    ds, proper_len = process_ds(ds, 128)

ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))


import torch
device0 = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

#nsp model
from transformers import BertTokenizer, BertForNextSentencePrediction
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache', local_files_only=True)
bert_nsp = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', cache_dir='./cache', local_files_only=True)
bert_nsp.to(device0)



accs = []
for ix, row in ds.df_train.reset_index().iterrows():
    torch.cuda.empty_cache() 
    pairs = list(itertools.product([row['content']], [". It is about {}".format(l) for l in labels_candidates] ))
    score_nsp = nsp_infer_pairs(pairs, bert_nsp, bert_tokenizer, device0)[:,0]
    if labels_candidates[score_nsp.argmax()] == row['label_name']:
        accs.append(1)
    else:
        accs.append(0)

    if len(accs) % 50 ==0:
        print(ix, sum(accs) / len(accs))


print(args.dsn, sum(accs) / len(accs))

    



