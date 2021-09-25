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
parser.add_argument("--genm", default="", type=str, choices=['gpt2','ctrl','t5','bart'])
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_train_epochs", default=7, type=int)
parser.add_argument("--ft_pattern", default='pp', type=str, choices=['pp', 'tc'])
parser.add_argument("--maxlen", default=512, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--ccsample", default=0.1, type=float)

args = parser.parse_args()

if args.genm in ['gpt2']:
    from transformers import GPT2Tokenizer
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
    tokenizer_gpt2.sep_token = '<|sep|>'
    #tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)


elif args.genm in ['t5']:
    from transformers import T5Tokenizer
    tokenizer_t5 = T5Tokenizer.from_pretrained('t5-base', cache_dir="./cache")





df_cc_text2text = get_cc_text_double(args.ft_pattern, args.ccsample)

row = df_cc_text2text.sample(1)
print(row['text1'].tolist()[0])
print('\n')
print(row['text2'].tolist()[0])
df_train, df_valid = train_test_split(df_cc_text2text, test_size=0.05)
print('data loaded', df_train.shape[0], df_valid.shape[0])

output_dir = '{}_{}_ft_on_ccnews'.format(args.genm, args.ft_pattern)

if args.genm =='gpt2':
    
    train_file = './fintune_csvs/{}_train_ft4cc.txt'.format(args.genm)
    validation_file = './fintune_csvs/{}_test_ft4cc.txt'.format(args.genm)

    with open (train_file, 'w') as f:
        for ix, row in df_train.iterrows():
            f.write("{} {} {} {}".format(row['text1'], tokenizer_gpt2.sep_token, row['text2'], tokenizer_gpt2.eos_token ) )
    print('train_file written')

    with open (validation_file, 'w') as f:
        for ix, row in df_valid.iterrows():
            f.write("{} {} {} {}".format(row['text1'], tokenizer_gpt2.sep_token, row['text2'], tokenizer_gpt2.eos_token ) )
    print('validation_file written')

    #   
    os.system(
    "CUDA_VISIBLE_DEVICES={} python -u ./run_clm_no_trainer.py \
            --num_train_epochs {} \
            --train_file {} \
            --validation_file {} \
            --model_name_or_path gpt2 \
            --per_device_train_batch_size {} \
            --per_device_eval_batch_size {} \
            --output_dir {} \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --model_type gpt2 \
            --block_size {}".format(args.gpu, args.num_train_epochs, train_file, validation_file, \
                    args.batch_size,  args.batch_size, \
                output_dir, 128) ) 



elif args.genm == 't5':
    train_file = './fintune_csvs/{}_train_ft4cc.csv'.format(args.genm)
    validation_file = './fintune_csvs/{}_test_ft4cc.csv'.format(args.genm)
    for col in ['text1', 'text2']:
        df_train[col] = df_train[col] + ' {}'.format(tokenizer_t5.eos_token)
        df_valid[col] = df_valid[col] + ' {}'.format(tokenizer_t5.eos_token)

    df_train.to_csv(train_file, index=False)
    df_valid.to_csv(validation_file, index=False)
    print('train_file validation_file written')
    #  
    os.system(
    "CUDA_VISIBLE_DEVICES={}  python -u ./run_summarization_no_trainer.py\
            --num_train_epochs {} \
            --train_file {} \
            --validation_file {} \
            --model_name_or_path t5-base \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --output_dir {} \
            --max_target_length {} \
            --val_max_target_length {} \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --max_length {} \
            --model_type t5 ".format(args.gpu, args.num_train_epochs, train_file, validation_file, output_dir,
                args.maxlen, args.maxlen, args.maxlen   ) ) 







