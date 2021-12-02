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
parser.add_argument("--samplecnt", default=32, type=int)
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
parser.add_argument("--gpu", default="0", type=str)

parser.add_argument("--ft_epochs", default=2, type=int)
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

ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))
ds, proper_len = process_ds(ds, 128)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))



from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache")
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)


gpt2_noft = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache")

# lambda
if not os.path.exists('ft_tmp'):
    os.makedirs('ft_tmp')

train_file = './ft_tmp/{}_train_finetune_{}_{}_ablation.txt'.format(args.dsn, args.samplecnt, args.seed)
validation_file = './ft_tmp/{}_test_finetune_{}_{}_ablation.txt'.format(args.dsn,  args.samplecnt, args.seed)

df_train_ft = ds.df_train.copy()
df_test_ft = ds.df_test.copy()

df_train_ft['text'] = df_train_ft['label_name'].map(lambda x: '[{}]'.format(x) ) + df_train_ft['content']
df_test_ft['text'] = df_test_ft['label_name'].map(lambda x: '[{}]'.format(x) ) + df_test_ft['content']

with open (train_file, 'w') as f:
    f.write(tokenizer_gpt2.eos_token.join(df_train_ft['text'].tolist()))

with open (validation_file, 'w') as f:
    f.write(tokenizer_gpt2.eos_token.join(df_test_ft['text'].tolist()))

model_output_path = "./ft_tmp/{}_{}_{}_lambda".format(args.dsn, args.samplecnt, args.seed) 
os.system(
"CUDA_VISIBLE_DEVICES={} python -u ./run_clm_no_trainer.py \
        --num_train_epochs {} \
        --train_file {} \
        --validation_file {} \
        --model_name_or_path gpt2 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --output_dir {} \
        --preprocessing_num_workers 8 --overwrite_cache True \
        --block_size {}".format(len(gpus)-1, args.ft_epochs, train_file, validation_file, model_output_path, 64) ) 
gpt2_lambda = GPT2LMHeadModel.from_pretrained(model_output_path)



# entire
with open (train_file, 'w') as f:
    f.write(tokenizer_gpt2.eos_token.join(df_train_ft['content'].tolist()))

with open (validation_file, 'w') as f:
    f.write(tokenizer_gpt2.eos_token.join(df_test_ft['content'].tolist()))

model_output_path = "./ft_tmp/{}_{}_{}_entire".format(args.dsn, args.samplecnt, args.seed) 
os.system(
"CUDA_VISIBLE_DEVICES={} python -u ./run_clm_no_trainer.py \
        --num_train_epochs {} \
        --train_file {} \
        --validation_file {} \
        --model_name_or_path gpt2 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --output_dir {} \
        --preprocessing_num_workers 8 --overwrite_cache True \
        --block_size {}".format(len(gpus)-1, args.ft_epochs, train_file, validation_file, model_output_path, 64) ) 
gpt2_entire = GPT2LMHeadModel.from_pretrained(model_output_path)


gpt2_tc = GPT2LMHeadModel.from_pretrained('ft_model_gpt_tc' )
gpt2_pp = GPT2LMHeadModel.from_pretrained('ft_model_gpt_pp' )

device_id = 0
gen_nlp = {}

gpt2_noft.trainable = False
gpt2_noft.config.pad_token_id=50256
gen_nlp['noft']  = pipeline("text-generation", model=gpt2_noft, tokenizer=tokenizer_gpt2, device=device_id, return_full_text=False)

gpt2_lambda.trainable = False
gpt2_lambda.config.pad_token_id=50256
gen_nlp['lambda']  = pipeline("text-generation", model=gpt2_lambda, tokenizer=tokenizer_gpt2, device=device_id, return_full_text=False)

gpt2_entire.trainable = False
gpt2_entire.config.pad_token_id=50256
gen_nlp['entire']  = pipeline("text-generation", model=gpt2_entire, tokenizer=tokenizer_gpt2, device=device_id, return_full_text=False)

gpt2_tc.trainable = False
gpt2_tc.config.pad_token_id=50256
gen_nlp['tc']  = pipeline("text-generation", model=gpt2_tc, tokenizer=tokenizer_gpt2, device=device_id, return_full_text=False)

gpt2_pp.trainable = False
gpt2_pp.config.pad_token_id=50256
gen_nlp['pp']  = pipeline("text-generation", model=gpt2_pp, tokenizer=tokenizer_gpt2, device=device_id, return_full_text=False)




dsn_maxlen = {'uci':64, 'agt':64, 'ag':128, 'nyt':128, 'amazon2':128, 'yelp2':128}

def gen_text(gen_nlp, prompt, ft):
    if ft == 'lambda':
        prompt = ' '.join(prompt.split(' ')[:4])
    result_gpt = gen_nlp([prompt], max_length=dsn_maxlen[args.dsn], \
                                    do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                    repetition_penalty=1.2, num_return_sequences= 8,\
                                    clean_up_tokenization_spaces=True)

    contents_syn_tmp = [remove_str(ii['generated_text']) for ii in result_gpt if ii]
    return random.sample(contents_syn_tmp, 1)[0]

infos = []
for ix, row in ds.df_train.reset_index().iterrows():
    torch.cuda.empty_cache()
    print(ix, "of", ds.df_train.shape[0], "ori====>", row['content'], "<===", row['label_name'])

    for ft, gen_nlp_v in gen_nlp.items():
        content_syn = gen_text(gen_nlp_v, row['content'], ft)
        content_syn = remove_str(content_syn)
        print("ft:{}==>{}".format(ft, content_syn))
        infos.append((content_syn, row['label_name'], row['label'], ft))


df_synthesize = pd.DataFrame(infos, columns=['content','label_name','label', 'ft'])


ds.df_train['ft'] = 'ori'
df_train_aug = pd.concat([ds.df_train + df_synthesize ]).sample(frac=1)

for ft in df_synthesize['ft'].unique():
    acc_aug, _ = thread_testing(args.testvalid, df_train_aug.loc[df_train_aug['ft'].isin(['ori',ft])], ds.df_test)
    print(ft, acc_aug)


