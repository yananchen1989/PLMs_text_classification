import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math,time,operator,traceback,shutil,string
from sklearn import metrics
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# import GPUtil
# GPUtil.showUtilization()
# deviceIDs = [0,1,2,3]
# #deviceIDs = GPUtil.getAvailable(order = 'memory', limit = 4, maxLoad = 1, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
# print("deviceIDs ==> ", deviceIDs)
# assert len(deviceIDs) >= 2

parser = argparse.ArgumentParser()
parser.add_argument("--aug", default="generate", type=str)
parser.add_argument("--dsn", default="ag", type=str, choices=['uci','ag','agt','nyt','yelp2','amazon2','stsa'])
parser.add_argument("--samplecnt", default=8, type=int)
parser.add_argument("--max_aug_times", default=1, type=int)

parser.add_argument("--model", default="albert", type=str)
parser.add_argument("--verbose", default=0, type=int)
parser.add_argument("--basemode", default="max", type=str) # rank or thres

#parser.add_argument("--nlim", default="joeddav/bart-large-mnli-yahoo-answers", type=str)
parser.add_argument("--epochs", default=100, type=int)
#parser.add_argument("--freq", default=25, type=int)
parser.add_argument("--testbed", default=1, type=int)
parser.add_argument("--testvalid", default='test', type=str)
parser.add_argument("--filter", default="nlinsp", type=str, choices=['nlinsp', 'clsembed'])

parser.add_argument("--valid_files_cnt", default=16, type=int)
parser.add_argument("--threads", default=64, type=int)

parser.add_argument("--genm", default="gpt", type=str, choices=['gpt','ctrl', 't5'])
parser.add_argument("--genft", default='no', type=str, choices=['no','lambda','tc','pp', 'ep'])

# dpfuture
#parser.add_argument("--future_steps", default=64, type=int)
#parser.add_argument("--test_beams", default=64, type=int)
parser.add_argument("--candidates", default=64, type=int)

#parser.add_argument("--num_return_sequences", default=4, type=int)
#parser.add_argument("--abundance", default=1, type=int)

parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu", default="", type=str)

# parser.add_argument("--ddi", default=2, type=int)
# parser.add_argument("--di", default=2, type=int)



# parser.add_argument("--encm", default='dan', type=str, \
#      choices=['dan', 'cmlm', \
#      'paraphrase-distilroberta-base-v2','paraphrase-mpnet-base-v2','paraphrase-TinyBERT-L6-v2',\
#      'paraphrase-MiniLM-L3-v2', 'average_word_embeddings_glove.6B.300d','nli-distilroberta-base-v2'])

args = parser.parse_args()
print('args==>', args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#args.filter = args.filter.split(',')

import numpy as np
import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
from transformers import pipeline
from threading import Thread
#tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
#tf.keras.backend.set_floatx('float16')
import nltk 
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity
#nltk.download('wordnet')
gpus = tf.config.list_physical_devices('GPU')
print('======>',gpus,'<=======')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

print("number of gpus==>", len(gpus))
device0 = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
#assert device0.type=='cuda' 

from utils.load_data import * 
from utils.transblock import * 
from utils.encoders import *
from utils.cbert_cgpt_config import * 
#from utils.dpp_model import * 
from utils.flair_ners import *


from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)

gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)

gpt2.trainable = False
gpt2.config.pad_token_id=50256
gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=len(gpus)-1, return_full_text=False)



from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_nli = AutoModelForSequenceClassification.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
tokenizer_nli = AutoTokenizer.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=len(gpus)-1)

dsn_maxlen = {'uci':64, 'agt':64, 'ag':128, 'nyt':128, 'amazon2':128, 'yelp2':128}

for args.dsn in ['uci','ag']:
    infos = []
    ds = load_data(dataset=args.dsn, samplecnt= 256)
    ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))
    ds, proper_len = process_ds(ds, 128)
    ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))

    ixl = {ii[0]:ii[1] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
    ixl_rev = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}

    with tf.distribute.MirroredStrategy().scope():
        model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
    model_cls.load_weights("./model_cls/model_full_{}.h5".format(args.dsn))   


    for ix, row in ds.df_train.sample(frac=1).reset_index().iterrows():
        torch.cuda.empty_cache()
        print(ix, "of", ds.df_train.shape[0], "ori====>", row['content'], "<===", row['label_name'])
         
        contents_syn = []
        fbs_gen = 64
        for _ in range(0, args.candidates//fbs_gen):
            torch.cuda.empty_cache()
            result_gpt = gen_nlp([row['content']], max_length=dsn_maxlen[args.dsn], \
                                            do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                            repetition_penalty=1.2, num_return_sequences= fbs_gen,\
                                            clean_up_tokenization_spaces=True)

            contents_syn_tmp = [remove_str(ii['generated_text']) for ii in result_gpt if ii]
            contents_syn.extend(contents_syn_tmp)
        torch.cuda.empty_cache()

        ners = get_ners(row['content'])
        labels_candidates_ners = [row['label_name']] + ners
        print(labels_candidates_ners)

        nli_scores_ner, nli_scores = [], []

        #fbs = 16
        #for ix in range(0, len(contents_syn), fbs):
        nli_result_ner = nli_nlp(contents_syn,  labels_candidates_ners, multi_label=True, hypothesis_template="This text is about {}.")

        for r in nli_result_ner:
            nli_scores_ner.append(np.array(r['scores']).mean())
            lr = {ii[0]:ii[1] for ii in zip(r['labels'], r['scores'])}
            nli_scores.append(lr[row['label_name']])


        dfdic = {'nli_score':nli_scores, 'nli_score_ner':nli_scores_ner, 'contents':contents_syn}
        df_tmp = pd.DataFrame(dfdic)

        preds = model_cls.predict(np.array(contents_syn),  batch_size=32, verbose=0)
        df_tmp['preds'] = preds[:, ixl_rev[row['label_name']]]
        corr_noner = df_tmp[['nli_score','preds']].corr().values[0][1]
        corr_ner = df_tmp[['nli_score_ner','preds']].corr().values[0][1]

        contents_head_noner = df_tmp.sort_values(by=['nli_score'], ascending=False).head(16)['contents'].values
        contents_head_ner = df_tmp.sort_values(by=['nli_score_ner'], ascending=False).head(16)['contents'].values

        scores_noner = model_cls.predict(contents_head_noner,  batch_size=32, verbose=0)
        scores_ner = model_cls.predict(contents_head_ner,  batch_size=32, verbose=0)   

        pred_ori = model_cls.predict(np.array([row['content']]),  batch_size=1, verbose=0) 
        if pred_ori.argmax() != ixl_rev[row['label_name']]:
            print("nomax", pred_ori.argmax(), ixl_rev[row['label_name']])

        infos.append((corr_noner, corr_ner, \
            scores_noner[ixl_rev[row['label_name']]].mean(), scores_ner[ixl_rev[row['label_name']]].mean() ))
    df_info = pd.DataFrame(infos, columns=['corr_noner', 'corr_ner', 'score_noner', 'score_ner'])
    print(args.dsn, df_info['corr_noner'].mean(), df_info['corr_ner'].mean(), \
                df['score_noner'].mean(), df['score_ner'].mean() )


