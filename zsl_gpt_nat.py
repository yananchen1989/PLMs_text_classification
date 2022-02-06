import pandas as pd
import time,argparse
import os,math,itertools
import numpy as np
import re,operator,joblib
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity 
import joblib,transformers
#assert gensim.__version__ == '4.1.2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch,operator
from torch import nn
from transformers import pipeline

from utils.seed_words import *



from utils.load_data import * 

ds = load_data(dataset='ag', samplecnt= 8)
labels_candidates = ds.df_train['label_name'].unique().tolist()






from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("roberta-large",cache_dir="./cache",local_files_only=True) 
model = AutoModelWithLMHead.from_pretrained("roberta-large",cache_dir="./cache",local_files_only=True)
nlp_fill = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0, top_k = len(tokenizer.vocab) ) 
id_token = {ix:token for token, ix in tokenizer.vocab.items()}


from sklearn import metrics
def check_fallin(seeds, fillin_token):
    for seed in seeds:
        if fillin_token in seed:
            return 1 
    return 0

accs = []
for ix, row in ds.df_test.sample(frac=1).reset_index().iterrows():
    content = row['content'] 
     

    template1 = "{}. This News is about {}".format(content, nlp_fill.tokenizer.mask_token)
    template2 = "{} News: {}.".format(nlp_fill.tokenizer.mask_token, content)
    template3 = "[Category: {} ] {}.".format(nlp_fill.tokenizer.mask_token, content)

    filled_results = nlp_fill([template1, template2, template3])


    # auc score
    ls_auc = {l:0 for l in labels_candidates}
    for filled_result in filled_results:
        infos = []
        for ii in filled_result:
            fillin_token = ii['token_str'].lower().strip()
            if fillin_token in stopwords or fillin_token in string.punctuation or fillin_token.isdigit() :
                continue
            score = ii['score']
            #print(fillin_token)
            rows = [score]
            for l in labels_candidates:
                checkin = check_fallin(label_expand[l], fillin_token)
                rows.append(checkin)
            infos.append(rows)

        dfr = pd.DataFrame(infos, columns=['score'] + labels_candidates)

        for l in labels_candidates:
            auc = metrics.roc_auc_score(dfr[l].values, dfr['score'].values)
            ls_auc[l] += auc 
    df_auc = pd.DataFrame(ls_auc.items(), columns=['label','score_auc'])

    # ori score
    ls = {l:0 for l in labels_candidates}
    for filled_result in filled_results:

        for r in filled_result :
            token = r['token_str'].lower().strip()

            if token in stopwords or token in string.punctuation or token.isdigit() :
                continue

            for l in ls.keys():
                if token in l.lower():
                    ls[l] += r['score']  
    df_noexpand = pd.DataFrame(ls.items(), columns=['label','score_noexpand'])

    df_fuse = pd.merge(df_noexpand, df_auc, on=['label'], how='inner')
    df_fuse['score'] = df_fuse['score_noexpand'].map(lambda x: math.log(x))    \
                                + df_fuse['score_auc'].map(lambda x: math.log(x))

    pred_label = df_fuse.sort_values(by=['score'], ascending=False).head(1)['label'].tolist()[0]

    print('pred===>', pred_label)
    print('label===>', row['label_name'])
    

    if row['label_name'] == pred_label:
        accs.append(1)
    else:
        accs.append(0)
        print(row['content'])

    if ix % 64 == 0:
        print(sum(accs) / len(accs))

    print()
















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


















