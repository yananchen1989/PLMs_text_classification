import pandas as pd
import time,argparse
import os,math,itertools
import numpy as np
import datasets,re,operator,joblib
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity 

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="yahoo", type=str)
parser.add_argument("--fbs_gpt", default=256, type=int)
parser.add_argument("--fbs_para", default=32, type=int)
parser.add_argument("--acc_topn", default=1, type=int)
parser.add_argument("--norm", default=0, type=int)
parser.add_argument("--param", default='t5', type=str)
parser.add_argument("--gpu", default="2", type=str)


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

if args.dsn == 'nyt':
    ds, proper_len = process_ds(ds, 128)

ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))


# nli model
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model_name = 'vicgalle/xlm-roberta-large-xnli-anli' #"facebook/bart-large-mnli"
model_nli = AutoModelForSequenceClassification.from_pretrained(nli_model_name, cache_dir='./cache', local_files_only=True)
tokenizer_nli = AutoTokenizer.from_pretrained(nli_model_name, cache_dir='./cache', local_files_only=True)
nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=len(gpus)-1)


#nsp model
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
device0 = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache', local_files_only=True)
bert_nsp = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', cache_dir='./cache', local_files_only=True)
bert_nsp.to(device0)


if args.param == 't5':
    from transformers import T5Tokenizer, AutoModelWithLMHead
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
    print(tokenizer_t5)
    t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)    
    gen_nlp_t5  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=len(gpus)-1)

elif args.param == 'bt':
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer_backward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", cache_dir="./cache", local_files_only=True)
    model_backward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en", cache_dir="./cache", local_files_only=True)
    tokenizer_forward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh", cache_dir="./cache", local_files_only=True)
    model_forward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh", cache_dir="./cache", local_files_only=True)
    nlp_backward = pipeline("translation", model=model_backward, tokenizer=tokenizer_backward, device=len(gpus)-1)
    nlp_forward = pipeline("translation", model=model_forward, tokenizer=tokenizer_forward, device=len(gpus)-1)
    print('bt model loaded')    

def para_t5(content):
    dsn_maxlen = {'uci':64, 'agt':64, 'ag':128, 'yahoo':128, 'nyt':128, 'amazon2':128, 'yelp2':128, 'imdb':128}
    result_gpt = gen_nlp_t5([content], max_length=dsn_maxlen[args.dsn], \
                                        do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                        repetition_penalty=1.2, num_return_sequences= args.fbs_para,\
                                        clean_up_tokenization_spaces=True)
    ori_gen_contents = [ii['generated_text'] for ii in result_gpt if ii['generated_text']] 
    return ori_gen_contents


def para_bt(content):
    content_ =  nlp_forward([content], truncation=True, \
                       do_sample=True, temperature=0.9, max_length=128, num_return_sequences=8)
    content_ = list(set([ii['translation_text'] for ii in content_]))
    content__ =  nlp_backward(content_, truncation=True, \
                        do_sample=True, max_length=128, temperature=0.9, num_return_sequences=8 )
    content__ = list(set([ii['translation_text'] for ii in content__]))
    return random.sample(content__, min(args.fbs_para, len(content__)))





def para_ranking(content_ori):

    if args.param == 't5':
        contents_para = para_t5(content_ori)
    elif args.param == 'bt':
        contents_para = para_bt(content_ori)

    ls = {l:0 for l in labels_candidates}
    nli_result_ll = []
    for j in range(0, len(contents_para), 16):
        contents_tmp = contents_para[j:j+16] 
        nli_result = nli_nlp(contents_tmp,  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
        if isinstance(nli_result, dict):
            nli_result_ll.append(nli_result)
        else:
            nli_result_ll.extend(nli_result)

    for r in nli_result_ll:
        for l,s in zip(r['labels'], r['scores']):
            ls[l] += s

    ls_sort = sorted(ls.items(), key=operator.itemgetter(1), reverse=True)
    df_t5 = pd.DataFrame(ls_sort, columns=['label','score_t5'])
    if args.norm:
        df_t5['score_t5'] = df_t5['score_t5'] / len(contents_para)
    return df_t5


def continuation_ranking(content_ori):
    infos = []
    for l in labels_candidates:
        scores_tmp = []
        # gpt generation
        contents_syn = df_contents_arxiv.loc[df_contents_arxiv['label_name']==l].sample(args.fbs_gpt)['content'].tolist()
        pairs = list(itertools.product([content_ori], contents_syn ))
        #for ii in range(0, args.fbs, 8):
            #pairs = [[row['content'], sent] for sent in contents_syn[ii:ii+8]]
        for j in range(0, len(pairs), 8):
            score_nsp = nsp_infer_pairs(pairs[j:j+8], bert_nsp, bert_tokenizer, device0)[:,0]
            scores_tmp.extend(list(score_nsp)) 

        score_nsp_reduce = np.array(scores_tmp).mean()
        infos.append((l, score_nsp_reduce))

    df_nsp = pd.DataFrame(infos, columns=['label','score_nsp'])
    return df_nsp




'''
from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)
gpt2.trainable = False
gpt2.config.pad_token_id=50256
gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=0, return_full_text=False)

infos = []
while True:
    #prompts = ["topic {} source strait stimes title".format(label) for label in labels_candidates]
    prompts = ["This is {} News: ".format(label) for label in labels_candidates] # gpt
    #prompts = ["Links In {} : ".format(label) for label in labels_candidates] # ctrl
    result_gpt = gen_nlp(prompts, max_length=128, \
                                                do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                                repetition_penalty=1.2, num_return_sequences= 16,\
                                                clean_up_tokenization_spaces=True)

    for label, rg in zip(labels_candidates, result_gpt):
        contents = [ ii['generated_text'] for ii in rg if len(ii['generated_text'])>=20 ] 
        for sent in contents:
            infos.append((remove_str(sent) , label ))

        # result_nlp = nli_nlp(contents, labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
        # for ii in  result_nlp:
        #     df_tmp = pd.DataFrame(ii)
        #     df_tmp_sel = df_tmp.loc[df_tmp['scores']>=0.9]
        #     if df_tmp_sel.shape[0] == 0:
        #         continue
        #     if label in df_tmp_sel['labels'].tolist():
        #         infos.append((remove_str(ii['sequence']), label))

    if len(infos) > 0 and len(infos) % 1000:
        df = pd.DataFrame(infos, columns = ['content','label_name'])
        print(df['label_name'].value_counts())
        df.to_csv("df_gen_ctrl_{}.csv".format(args.dsn), index=False)
        if df['label_name'].value_counts().min() >= 2048:
            break 
'''

df_contents_arxiv = pd.read_csv("df_gen_{}.csv".format(args.dsn))

acc = {}
for ix, row in ds.df_train.reset_index().iterrows():
    torch.cuda.empty_cache()

    nli_result = nli_nlp([row['content']],  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
    nli_result.pop('sequence')
    df_noexpand = pd.DataFrame(nli_result)
    df_noexpand = df_noexpand.rename(columns={'labels': 'label', 'scores':'score_noexpand'})


    df_t5 = para_ranking(row['content'])
    df_nsp = continuation_ranking(row['content'])


    df_fuse_ = pd.merge(df_noexpand, df_t5, on=['label'], how='inner')
    df_fuse  = pd.merge(df_fuse_, df_nsp, on=['label'], how='inner')

    df_fuse['score_w_nsp'] =  df_fuse['score_noexpand'].map(lambda x: math.log(x)) \
                            + df_fuse['score_nsp'].map(lambda x: math.log(x)) 

    df_fuse['score_w_t5'] =  df_fuse['score_noexpand'].map(lambda x: math.log(x)) \
                            + df_fuse['score_t5'].map(lambda x: math.log(x)) 

    df_fuse['score_w_t5_nsp'] =  df_fuse['score_noexpand'].map(lambda x: math.log(x)) \
                            + df_fuse['score_t5'].map(lambda x: math.log(x))    \
                            + df_fuse['score_nsp'].map(lambda x: math.log(x))     

    for col in ['score_noexpand','score_w_t5', 'score_w_nsp', 'score_w_t5_nsp']:
        if col not in acc.keys():
            acc[col] = []

        pred_label = df_fuse.sort_values(by=[col], ascending=False).head(args.acc_topn)['label'].tolist()
        if row['label_name'] in pred_label:
            acc[col].append(1)
        else:
            acc[col].append(0)

    if ix % 64 == 0 and ix > 0:
        print(ix)
        for col in ['score_noexpand','score_w_t5', 'score_w_nsp', 'score_w_t5_nsp']:
            print(col, round(np.array(acc[col]).mean(), 4))
        print()

print("final_summary==>", ' '.join(['{}:{}'.format(k, v) for k, v in vars(args).items()]))