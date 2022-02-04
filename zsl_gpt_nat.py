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



from transformers import GPT2LMHeadModel, GPT2TokenizerFast

import torch,transformers
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained("./finetunes/gpt_nat_zsl/epoch_1_ppl_11.918800044184868").to(device)

#model = GPT2LMHeadModel.from_pretrained("./finetunes/gpt2_natcat").to(device)

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)

# model.trainable = False
# model.config.pad_token_id=50256
# gen_nlp  = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, return_full_text=False)



from transformers.file_utils import cached_path
gram_diff = joblib.load("gram_diff___{}".format('ag'))
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

def get_seed_words(topk):
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
            if len(label_expands_auto[l]) == topk:
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


label_expands_auto = get_seed_words(64)
print(label_expands_auto)

stopwords = joblib.load("./utils/stopwords")
stopwords = set(stopwords)  

id_token = {ix:token for token, ix in tokenizer.vocab.items()}












from utils.load_data import * 

ds = load_data(dataset='yahoo', samplecnt= 8)
labels_candidates = ds.df_train['label_name'].unique().tolist()


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


















