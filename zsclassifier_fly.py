import pandas as pd
import time,argparse
import os,math,itertools
import numpy as np
import datasets,re,operator,joblib
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk,gensim
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity 
assert gensim.__version__ == '4.1.2'

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="yahoo", type=str)
parser.add_argument("--fbs", default=64, type=int)
parser.add_argument("--acc_topn", default=1, type=int)
parser.add_argument("--w1", default=0.5, type=float)
parser.add_argument("--w2", default=0.5, type=float)
parser.add_argument("--gpu", default="7", type=str)
parser.add_argument("--para", default=1, type=int)
parser.add_argument("--para_cnt", default=8, type=int)

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

if args.para:
    from transformers import T5Tokenizer, AutoModelWithLMHead
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
    print(tokenizer_t5)
    t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)    
    gen_nlp_t5  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=0)

def para_t5(content):
    dsn_maxlen = {'uci':64, 'agt':64, 'ag':128, 'yahoo':128, 'nyt':128, 'amazon2':128, 'yelp2':128, 'imdb':128}
    result_gpt = gen_nlp_t5([remove_str(content)], max_length=dsn_maxlen[args.dsn], \
                                        do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                        repetition_penalty=1.2, num_return_sequences= args.para_cnt,\
                                        clean_up_tokenization_spaces=True)
    ori_gen_contents = [ii['generated_text'] for ii in result_gpt if ii['generated_text']] + [remove_str(content)]
    return ori_gen_contents

 


df_contents_arxiv = pd.read_csv("df_gen_{}.csv".format(args.dsn))

accs_noexpand = []
accs_expand = []
for ix, row in ds.df_train.reset_index().iterrows():
    torch.cuda.empty_cache()

    nli_result = nli_nlp([row['content']],  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
    pred_label =  nli_result['labels'][:args.acc_topn]
    if row['label_name'] in pred_label:
        accs_noexpand.append(1)
    else:
        accs_noexpand.append(0)

    nli_result.pop('sequence')
    df_noexpand = pd.DataFrame(nli_result)
    df_noexpand = df_noexpand.rename(columns={'labels': 'label', 'scores':'score_noexpand'})

    # pairs_l = [[row['content'], "This text is about {}".format(l)] for l in labels_candidates]
    # score_nsp_l = nsp_infer_pairs(pairs_l, bert_nsp, bert_tokenizer)[:,0]

    # df_nsp_l = pd.DataFrame(zip(labels_candidates, list(score_nsp_l)), columns=['label','score_nsp_l'])

    if args.para:
        content_ori = para_t5(row['content'])
    else:
        content_ori = [row['content']]

    infos = []
    for l in labels_candidates:
        scores_tmp = []
        contents_syn = df_contents_arxiv.loc[df_contents_arxiv['label_name']==l].sample(args.fbs)['content'].tolist()
        pairs = list(itertools.product(content_ori, contents_syn ))
        #for ii in range(0, args.fbs, 8):
            #pairs = [[row['content'], sent] for sent in contents_syn[ii:ii+8]]
        for j in range(0, len(pairs), 8):
            score_nsp = nsp_infer_pairs(pairs[j:j+8], bert_nsp, bert_tokenizer)[:,0]
            scores_tmp.extend(list(score_nsp)) 

        score_nsp_reduce = np.array(scores_tmp).mean()
        infos.append((l, score_nsp_reduce))

    df_nsp = pd.DataFrame(infos, columns=['label','score_nsp'])

    df_fuse = pd.merge(df_noexpand, df_nsp, on=['label'], how='inner')
    #df_fuse  = pd.merge(df_merge, df_nsp_l, on=['label'], how='inner')

    df_fuse['score_fuse'] = args.w1 * df_fuse['score_noexpand'].map(lambda x: math.log(x)) \
                                + args.w2 * df_fuse['score_nsp'].map(lambda x: math.log(x)) 
                                #+ ( 1-args.w1 -args.w2 ) * df_fuse['score_nsp_l'].map(lambda x: math.log(x))

    pred_label = df_fuse.sort_values(by=['score_fuse'], ascending=False).head(args.acc_topn)['label'].tolist()

    if row['label_name'] in pred_label:
        accs_expand.append(1)
    else:
        accs_expand.append(0)

    if ix % 32 == 0 and ix > 0:
        print(ix, sum(accs_noexpand) / len(accs_noexpand), sum(accs_expand)/len(accs_expand))

print("final_summary==>", ' '.join(['{}:{}'.format(k, v) for k, v in vars(args).items()]),
     sum(accs_noexpand) / len(accs_noexpand), sum(accs_expand)/len(accs_expand) )