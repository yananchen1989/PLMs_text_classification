import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math,time,operator,traceback,shutil,string
from sklearn import metrics
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="uci", type=str, choices=['uci','ag','agt','nyt','yelp2','amazon2','stsa'])
parser.add_argument("--samplecnt", default=128, type=int)
parser.add_argument("--max_aug_times", default=1, type=int)
parser.add_argument("--testmode", default=0, type=int)
parser.add_argument("--model", default="albert", type=str)
parser.add_argument("--genm", default="gpt", type=str, choices=['gpt','ctrl', 't5'])
parser.add_argument("--test_beams", default=32, type=int)
parser.add_argument("--candidates", default=64, type=int)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()
print('args==>', args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#args.filter = args.gpu.split(',')

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

ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))
ds, proper_len = process_ds(ds, 128)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))

print(ds.df_train.sample(8))
print('proper_len==>', proper_len)
ixl = {ii[0]:ii[1] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
ixl_rev = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
#seed = random.sample(list(range(10000)), 1)[0]


####################### generation setting ######################
if args.genm == 'gpt':
    from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
    #tokenizer_gpt2.padding_side = "left" 
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
    tokenizer_gpt2.sep_token = '<|sep|>'
    #tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
    print(tokenizer_gpt2)
    gpt2_noft = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
    gpt2_noft.trainable = False
    gpt2_noft.config.pad_token_id=50256
    gen_nlp  = pipeline("text-generation", model=gpt2_noft, tokenizer=tokenizer_gpt2, device=len(gpus)-1, return_full_text=False)

elif  args.genm == 't5':
    from transformers import T5Tokenizer, AutoModelWithLMHead
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
    print(tokenizer_t5)
    t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
    gen_nlp  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=len(gpus)-2)

print('generate model loaded ==>{}'.format(args.genm))

dsn_maxlen = {'uci':64, 'agt':64, 'ag':128, 'nyt':128, 'amazon2':128, 'yelp2':128}

print("begin_to_test_noaug")

if not args.testmode: 
    acc_noaug, model_cls = do_train_test_thread(ds.df_train, ds.df_test, 'albert', 16)
    print("base acc==>", acc_noaug)

else:
    with tf.distribute.MirroredStrategy().scope():
        model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
        model_cls.load_weights("./model_cls/model_full_{}.h5".format(args.dsn))   
        acc_noaug = -1

def mc_gen(row):
    # get mc scores
    result_gpt = gen_nlp([row['content']], max_length=dsn_maxlen[args.dsn], \
                                    do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                    repetition_penalty=1.2, num_return_sequences= args.candidates,\
                                    clean_up_tokenization_spaces=True)

    contents_syn = [remove_str(ii['generated_text']) for ii in result_gpt if ii]

    contents_syn_mc_trunk = []

    fbs_mc = 8
    for ii in range(0, len(contents_syn), fbs_mc):
        result_mc = gen_nlp(contents_syn[ii:ii+fbs_mc], max_length=dsn_maxlen[args.dsn], \
                                        do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                        repetition_penalty=1.2, num_return_sequences= args.test_beams,\
                                        clean_up_tokenization_spaces=True)
        if args.genm == 'gpt':
            for s in result_mc:
                samples = [ss['generated_text'] for ss in s ]
                contents_syn_mc_trunk.extend(samples)
        elif args.genm == 't5':
            contents_syn_mc_trunk.extend([s['generated_text'] for s in result_mc ])
        
    assert len(contents_syn_mc_trunk) == len(contents_syn) * args.test_beams
    preds = model_cls.predict(np.array(contents_syn_mc_trunk),  batch_size=64, verbose=0) 

    mc_scores_tmp = []
    for j in range(0, len(contents_syn_mc_trunk), args.test_beams):
        pred_mean = preds[j:j+args.test_beams, row['label']].mean()
        mc_scores_tmp.append(pred_mean)
    assert len(mc_scores_tmp) == len(contents_syn)

    df_future = pd.DataFrame(zip(contents_syn, mc_scores_tmp), columns=['content','mc_score'])
    contents_syn_sort = df_future.sort_values(by=['mc_score'], ascending=False)['content'].tolist()

    return contents_syn_sort





infos = []
for ix, row in ds.df_train.reset_index().iterrows():
    torch.cuda.empty_cache()
    print(ix, "of", ds.df_train.shape[0], "ori====>", row['content'], "<===", row['label_name'])

    contents_syn_sort = mc_gen(row)
    print("mc0==>", contents_syn_sort[0])
    print("mc1==>", contents_syn_sort[1])
    print("mc2==>", contents_syn_sort[2])
    print('\n')

    for i in range(args.max_aug_times):
        infos.append((contents_syn_sort[i], row['label_name'], row['label']))

df_synthesize = pd.DataFrame(infos, columns=['content','label_name','label'])

print("final generated==>", df_synthesize.shape[0]/ds.df_train.shape[0])



df_train_aug = pd.concat([ds.df_train, df_synthesize] ).sample(frac=1)
print("begin_to_test_aug")
acc_aug, _ = do_train_test_thread(df_train_aug, ds.df_test, 'albert', 16)
summary = ['summary===>'] + ['{}:{}'.format(k, v) for k, v in vars(args).items()] +  ['acc_base:{} acc_aug:{}'.format( acc_noaug, acc_aug )]
print('success', ' '.join(summary))



