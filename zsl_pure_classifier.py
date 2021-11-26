import os,time 
import numpy as np 
import pandas as pd 
os.environ['CUDA_VISIBLE_DEVICES'] =''

from utils.load_data import * 
from utils.transblock import * 
from transformers import pipeline
import argparse,torch

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str)
args = parser.parse_args()

ds = load_data(dataset=args.dsn, samplecnt=-1)

ds, proper_len = process_ds(ds, 256)


nli_nlp_dic = {}
acc_dic = {}
time_dic = {}
from transformers import AutoModelForSequenceClassification, AutoTokenizer
for nli_model_name in[ 'facebook/bart-large-mnli', "vicgalle/xlm-roberta-large-xnli-anli", 'joeddav/xlm-roberta-large-xnli']: # 
    model_nli = AutoModelForSequenceClassification.from_pretrained(nli_model_name, cache_dir='./cache', local_files_only=True)
    tokenizer_nli = AutoTokenizer.from_pretrained(nli_model_name, cache_dir='./cache', local_files_only=True)
    nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=-1)
    nli_nlp_dic[nli_model_name] = nli_nlp
    acc_dic[nli_model_name] = []
    time_dic[nli_model_name] = []

labels_candidates = list(ds.df_test['label_name'].unique())


for ix, row in ds.df_test.reset_index().iterrows():
    content = row['content']
    for nli_model_name, nli_nlp in nli_nlp_dic.items():
        t0 = time.time()
        nli_result = nli_nlp([content],  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
        t1 = time.time()
        time_dic[nli_model_name] = t1 - t0
        pred_label =  nli_result['labels'][0]
        if pred_label == row['label_name']:
            acc_dic[nli_model_name].append(1)
        else:
            acc_dic[nli_model_name].append(0)

    if ix % 50 == 0:
        print(ix)
        for nli_model_name in nli_nlp_dic.keys():
            print(nli_model_name)
            print("acc:", np.array(acc_dic[nli_model_name]).mean())
            print("time cost:", np.array(time_dic[nli_model_name]).mean())
        print('\n') 



