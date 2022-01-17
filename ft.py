import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import torch
from utils.load_data import * 

parser = argparse.ArgumentParser()
parser.add_argument("--genm", default="", type=str, choices=['gpt2', 't5'])
parser.add_argument("--dsn_summary", default="", type=str, choices=['xsum', 'cnndm'])
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_train_epochs", default=7, type=int)
parser.add_argument("--ft_pattern", default='pp', type=str, choices=['pp', 'tc', 'ep', 'entire', 'summary'])
parser.add_argument("--maxlen", default=512, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--ccsample", default=0.1, type=float)

args = parser.parse_args()

if args.genm in ['gpt2']:
    from transformers import GPT2Tokenizer
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
    #tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
    tokenizer_gpt2.sep_token = '<|sep|>'


# elif args.genm in ['t5']:
#     from transformers import T5Tokenizer
#     tokenizer_t5 = T5Tokenizer.from_pretrained('t5-base', cache_dir="./cache")


if not os.path.exists('fintune_csvs'):
    os.makedirs('fintune_csvs')

if args.ft_pattern in ['tc','pp','ep']:
    df = get_cc_text_double(args.ft_pattern, args.ccsample)
    df = df.loc[(~df['text1'].isnull()) & (~df['text2'].isnull())]

elif args.ft_pattern in ['entire']:
    assert args.genm == 'gpt2'
    df = get_cc_text_single(args.ccsample)

elif args.ft_pattern in ['summary']:
    assert args.genm == 't5'
    df = get_summary_text_double(args.dsn_summary, s=1)


if args.ft_pattern in ['summary', 'tc','pp','ep']:
    row = df.sample(1)
    print(row['text1'].tolist()[0])
    print('\n')
    print(row['text2'].tolist()[0])

df_train, df_valid = train_test_split(df, test_size=0.05)
print('data loaded', df_train.shape[0], df_valid.shape[0])

output_dir = 'ft_model_{}_{}'.format(args.genm, args.ft_pattern) 






import pandas as pd 
import glob
from sklearn.model_selection import train_test_split


files = glob.glob("./torch_ds/natcat-data/*/train.tsv*.data")
infos = []
for file in files:
    with open(file, 'r') as f:
        for line in f: 
            tokens = line.strip().split('\t')
            assert len(tokens) == 9 
            infos.append(tokens)

df_nat = pd.DataFrame(infos, columns=['label'] + ['neg_label_{}'.format(i) for i in range(7)] + ['content'] )

df_nat['text'] = df_nat['label'].map(lambda x: "This document is about {} : ".format(x)) \
                    + df_nat['content']  

df_nat_train, df_nat_test =  train_test_split(df_nat, test_size=0.001)

print(df_nat_train.shape[0], df_nat_test.shape[0])






###### for gpt

from utils.load_data import * 
with open ("df_nat_train.txt", 'w') as f:
    for line in df_nat_train['text'].tolist():
        f.write(remove_str(line) + '\n')

with open ("df_nat_test.txt", 'w') as f:
    for line in df_nat_test['text'].tolist():
        f.write(remove_str(line) + '\n')

with open ("df_nat_train_sample.txt", 'w') as f:
    for line in df_nat_train.sample(200000)['text'].tolist():
        f.write(remove_str(line) + '\n')





####### for t5



df_nat['prefix'] = df_nat['label'].map(lambda x: "This document is about {}".format(x))

df_nat_train, df_nat_test =  train_test_split(df_nat[['prefix', 'content']], test_size=0.001)

df_nat_train.to_csv("df_nat_train.csv", index=False)
df_nat_test.to_csv("df_nat_test.csv", index=False)

df_nat_train.sample(200000).to_csv("df_nat_train_sample.csv", index=False)


from datasets import load_dataset
data_files = {}
data_files["train"] = "df_nat_train_sample.csv"
data_files["validation"] = "df_nat_test.csv"
raw_datasets = load_dataset("csv", data_files=data_files)














