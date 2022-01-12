

import pandas as pd
import time,argparse
import os,math
import numpy as np
import datasets,re,operator,joblib
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="yahoo", type=str)
parser.add_argument("--gpu", default="1", type=str)
args = parser.parse_args()
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

ds = load_data(dataset=args.dsn, samplecnt= -1)
labels_candidates = ds.df_train['label_name'].unique().tolist()
print(labels_candidates)
ixl = {ii[0]:ii[1] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
ixl_rev = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model_name = 'vicgalle/xlm-roberta-large-xnli-anli' #"facebook/bart-large-mnli"
model_nli = AutoModelForSequenceClassification.from_pretrained(nli_model_name, cache_dir='./cache', local_files_only=True)
tokenizer_nli = AutoTokenizer.from_pretrained(nli_model_name, cache_dir='./cache', local_files_only=True)
nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=len(gpus)-1)




df = get_cc_news(1)
df = df.loc[(~df['title'].isnull()) & (~df['content'].isnull())]

infos = []
for ix, row in df.iterrows():
    result_ori = nli_nlp(row['content'], labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
    result_ori.pop('sequence')
    df_nli_row = pd.DataFrame(result_ori)

    pred_label_names = df_nli_row.loc[df_nli_row['scores']>=0.9,'labels'].tolist()
    if not pred_label_names:
        continue

    for l in pred_label_names:
        infos.append((row['content'], l)) 

df_nli_pred = pd.DataFrame(infos, columns=['content','label_name'])
df_nli_pred['label'] = df_nli_pred['label_name'].map(lambda x: ixl_rev[x])
print("nli labelling completed==>", df_nli_pred.shape[0])
print(df_nli_pred['label_name'].value_counts())

df_nli_pred.to_csv("./df_cc_pred_nli/df_nli_pred_{}.csv".format(args.dsn), index=False)


df_nli_pred_sample = sample_stratify(df_nli_pred, df_nli_pred['label_name'].value_counts().min())

acc_aug_nli, _ = do_train_test_thread(df_nli_pred_sample, ds.df_test, 'albert', 32)
print(args.dsn, acc_aug_nli)
