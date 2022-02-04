import pandas as pd
import time,argparse
import os,math,itertools
import numpy as np
import re,operator,joblib
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity 
import joblib,gensim,transformers
#assert gensim.__version__ == '4.1.2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch,operator
from torch import nn
from transformers import pipeline




# from transformers import T5Tokenizer, AutoModelWithLMHead
# tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
# print(tokenizer_t5)
# t5 = AutoModelWithLMHead.from_pretrained("./finetunes/t5_natcat")    
# gen_nlp_t5  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=-1)




from utils.seed_words import *

stopwords = joblib.load("./utils/stopwords")
stopwords = set(stopwords)  

id_token = {ix:token for token, ix in tokenizer.vocab.items()}



for l, w in label_expands_auto.items():



from utils.load_data import * 

ds = load_data(dataset='ag', samplecnt= -1)
labels_candidates = ds.df_train['label_name'].unique().tolist()





for ix, row in ds.df_test.sample(frac=1).iterrows():
    content = row['content'] 


template1 = "{}. This News is about {}".format(content, nlp_fill.tokenizer.mask_token)
template2 = "{} News: {}.".format(nlp_fill.tokenizer.mask_token, content)
template3 = "[Category: {} ] {}.".format(nlp_fill.tokenizer.mask_token, content)

ls = {l:0 for l in labels_candidates}

filled_results = nlp_fill([template1, template2, template3])






from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("roberta-large",cache_dir="./cache",local_files_only=True) 
model = AutoModelWithLMHead.from_pretrained("roberta-large",cache_dir="./cache",local_files_only=True)
nlp_fill = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=-1, top_k = len(tokenizer.vocab) ) 
id_token = {ix:token for token, ix in tokenizer.vocab.items()}




















def gpt4zsl_pred(sent):
    prompt = "{}. This document is about".format(remove_str(sent))
    inputs = tokenizer(prompt, \
                     truncation=True, max_length=64, return_tensors="pt").to(device)

    output_sequences = model.generate(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'] ,
            max_length= 54,
            temperature=1,
            top_k=0,
            top_p=0.99,
            repetition_penalty=1,
            do_sample=True,
            num_return_sequences=64
        )

    preds = []
    for output_ids in output_sequences:
        #syn_sent = tokenizer.batch_decode(output_ids[inputs['input_ids'][0].shape[0]:], clean_up_tokenization_spaces=True, skip_special_tokens=True)
        
        syn_sent = tokenizer.batch_decode(output_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        
        preds.append(''.join(syn_sent).strip().split('This document is about')[-1].strip() )
    return preds 

def get_ppl(sent, ori_sent):
    encodings_ori = tokenizer(ori_sent, return_tensors='pt')
    ori_tokens_cnt = encodings_ori.input_ids.size(1)

    encodings = tokenizer(sent, return_tensors='pt')

    max_length = model.config.n_positions
    stride = 1
    nlls = []
    for i in range(1, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100
        # print(i)
        # print(input_ids)
        # print(target_ids)
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls)[-ori_tokens_cnt:].sum() / end_loc)
    #print(ppl.numpy().reshape(-1)[0])
    return ppl.cpu().numpy().reshape(-1)[0]



for ix, row in ds.df_test.sample(frac=1).iterrows():
    torch.cuda.empty_cache() 
    preds = gpt4zsl_pred(remove_str(row['content']))
    print(row['label_name'])
    print(preds)
    print('\n')
    # embed_pred = enc.infer(preds)
    # simis = cosine_similarity(embed_label, embed_pred)

    # df_simi = pd.DataFrame( zip(labels_candidates, simis.mean(axis=1)), columns=['label', 'simi'])

    # pred_label = df_simi.sort_values(by=['simi'], ascending=False).head(1)['label'].tolist()[0]
    # print(row['label_name'], pred_label)

    infos = []
    for l in labels_candidates:
        #seed_words = random.sample(list(label_expands_auto[l]), 16)

        #template = "{}.  This document is about {}".format(remove_str(row['content']), ' and '.join(seed_words))
        template = "This document is about {} : {}".format(l, remove_str(row['content']) )
        ppl = get_ppl(template, remove_str(row['content']) )

        infos.append((l, ppl ))
        #print(l, ppl)
    dfi = pd.DataFrame(infos, columns=['label','ppl'])
    pred_label = dfi.sort_values(['ppl']).head(1)['label'].tolist()[0]
    print(row['label_name'])
    print( dfi.sort_values(['ppl']))
    print()


















