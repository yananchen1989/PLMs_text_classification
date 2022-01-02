import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math,time,operator,traceback,shutil,string
from sklearn import metrics
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str, choices=['uci','ag','agt','nyt','yelp2','amazon2','stsa'])
parser.add_argument("--samplecnt", default=8, type=int)
parser.add_argument("--max_aug_times", default=1, type=int)

parser.add_argument("--model", default="albert", type=str)
parser.add_argument("--verbose", default=0, type=int)
parser.add_argument("--basemode", default="max", type=str) # rank or thres

#parser.add_argument("--nlim", default="joeddav/bart-large-mnli-yahoo-answers", type=str)
parser.add_argument("--epochs", default=100, type=int)
#parser.add_argument("--freq", default=25, type=int)
parser.add_argument("--testvalid", default='test', type=str)

parser.add_argument("--genm", default="gpt", type=str, choices=['gpt','ctrl', 't5'])

parser.add_argument("--candidates", default=64, type=int)

#parser.add_argument("--num_return_sequences", default=4, type=int)
#parser.add_argument("--abundance", default=1, type=int)

parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu", default="1", type=str)
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

ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))
ds, proper_len = process_ds(ds, 128)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))

print(ds.df_train.sample(8))
print('proper_len==>', proper_len)
ixl = {ii[0]:ii[1] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
ixl_rev = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
#seed = random.sample(list(range(10000)), 1)[0]

testbed_func = {"test":do_train_test_thread, "valid":do_train_test_valid_thread}
def thread_testing(testvalid, df_train, df_test):
    best_test_accs = []
    models = []

    for ddi in range(1):
        threads = []
        for di in range(1):
            t = Thread(target=testbed_func[testvalid], args=(df_train, df_test, best_test_accs, models, di + ddi*2, \
                              args.epochs,  args.verbose, 'albert', 8))
            t.start()
            threads.append(t)
        # join all threads
        for t in threads:
            t.join() 

    if args.basemode == 'mean':
        acc = round(np.array(best_test_accs).mean(), 4)
    elif args.basemode == 'max':
        acc = round(np.array(best_test_accs).max(), 4)

    model_best = models[np.array(best_test_accs).argmax()]
    return  acc, model_best

print("begin_to_test_noaug")
acc_noaug, model_cls = thread_testing(args.testvalid, ds.df_train, ds.df_test)

# with tf.distribute.MirroredStrategy().scope():
#     model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
# model_cls.load_weights("./model_cls/model_full_{}.h5".format(args.dsn))   

####################### generation setting ######################
#if args.genm == 'gpt':
from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)

gpt2_noft = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)

if not os.path.exists('ft_tmp'):
    os.makedirs('ft_tmp')

train_file = './ft_tmp/{}_train_finetune_{}_{}.txt'.format(args.dsn, args.samplecnt, args.seed)
validation_file = './ft_tmp/{}_test_finetune_{}_{}.txt'.format(args.dsn,  args.samplecnt, args.seed)

df_train_ft = ds.df_train.copy()
df_test_ft = ds.df_test.copy()

df_train_ft['text'] = df_train_ft['label_name'].map(lambda x: '[{}]'.format(x) ) + df_train_ft['content']
df_test_ft['text'] = df_test_ft['label_name'].map(lambda x: '[{}]'.format(x) ) + df_test_ft['content']

with open (train_file, 'w') as f:
    f.write(tokenizer_gpt2.eos_token.join(df_train_ft['text'].tolist()))

with open (validation_file, 'w') as f:
    f.write(tokenizer_gpt2.eos_token.join(df_test_ft['text'].tolist()))

model_output_path = "./ft_tmp/{}_{}_{}".format(args.dsn, args.samplecnt, args.seed) 
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
        --block_size {}".format(len(gpus)-1, 12, train_file, validation_file, model_output_path, 64) ) 
gpt2_ft = GPT2LMHeadModel.from_pretrained(model_output_path)


gpt2_noft.trainable = False
gpt2_noft.config.pad_token_id=50256
gen_nlp_gpt2_noft  = pipeline("text-generation", model=gpt2_noft, tokenizer=tokenizer_gpt2, device=len(gpus)-1, return_full_text=False)


gpt2_ft.trainable = False
gpt2_ft.config.pad_token_id=50256
gen_nlp_gpt2_ft  = pipeline("text-generation", model=gpt2_ft, tokenizer=tokenizer_gpt2, device=len(gpus)-1, return_full_text=False)

print('generate model loaded ==>{}'.format(args.genm))

dsn_maxlen = {'uci':64, 'agt':64, 'ag':128, 'nyt':128, 'amazon2':128, 'yelp2':128}

####################### filter setting ######################
#if 'nlinsp' in args.filter: 
#nli_nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=1) #  1.8.1+cu102
# vicgalle/xlm-roberta-large-xnli-anli joeddav/xlm-roberta-large-xnli 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_nli = AutoModelForSequenceClassification.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
tokenizer_nli = AutoTokenizer.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=len(gpus)-1)

# with tf.distribute.MirroredStrategy().scope():
#     bert_nsp  = get_model_nsp(256)
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
device0 = torch.device("cuda:{}".format(len(gpus)-1) if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache', local_files_only=True)
bert_nsp = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', cache_dir='./cache', local_files_only=True)
bert_nsp.to(device0)


enc = encoder('cmlm-base')
enc_dic = {}
for l in ds.df_train['label'].unique():
    contents_ = ds.df_train.loc[ds.df_train['label']==l]['content'].values
    embeds = enc.infer(contents_)
    centroid = embeds.mean(axis=0).reshape(1, -1) 
    enc_dic[l] = centroid
    


def gen(row, ft_flag):
    if ft_flag == 'noft':
        prompt = remove_str(row['content']) 
        gen_nlp = gen_nlp_gpt2_noft
    elif ft_flag == 'ft':
        prompt = '[{}]'.format(row['label_name']) +  ' '.join(row['content'].split(' ')[:3] )
        gen_nlp = gen_nlp_gpt2_ft

    contents_syn = []
    fbs_gen = 64
    for _ in range(0, args.candidates//fbs_gen):
        torch.cuda.empty_cache()
        result_gpt = gen_nlp([prompt], max_length=dsn_maxlen[args.dsn], \
                                        do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                        repetition_penalty=1.2, num_return_sequences= fbs_gen,\
                                        clean_up_tokenization_spaces=True)

        contents_syn_tmp = [remove_str(ii['generated_text']) for ii in result_gpt if ii]
        contents_syn.extend(contents_syn_tmp)
    return contents_syn



def clsembednlinsp_gen(row, ft_flag):
          
    contents_syn = gen(row, ft_flag)
    torch.cuda.empty_cache()

    #nlisp
    nli_result = nli_nlp(contents_syn,  [row['label_name']], multi_label=True, hypothesis_template="This text is about {}.")
    nli_scores = [r['scores'][0] for r in nli_result] 

    torch.cuda.empty_cache()
    # get nsp score
    pairs = [[remove_str(row['content']), sent] for sent in contents_syn ]

    nsp_scores = []
    for j in range(0, len(pairs), 8):
        score_nsp = nsp_infer_pairs(pairs[j:j+8], bert_nsp, bert_tokenizer, device0)[:,0]
        nsp_scores.extend(list(score_nsp)) 
    
    df_tmp = pd.DataFrame(zip(contents_syn, nli_scores, nsp_scores ), columns=['content','nli_score', 'nsp_score'])

    df_tmp['score'] = df_tmp['nli_score'].map(lambda x: math.log(x)) + df_tmp['nsp_score'].map(lambda x: math.log(x))

    result_syn = {}
    result_syn['{}-nlisp'.format(ft_flag)] = df_tmp.sort_values(by=['score'], ascending=False).head(1)['content'].tolist()[0] 
    result_syn['{}-nli'.format(ft_flag)] = df_tmp.sort_values(by=['nli_score'], ascending=False).head(1)['content'].tolist()[0] 
    result_syn['{}-nsp'.format(ft_flag)] = df_tmp.sort_values(by=['nsp_score'], ascending=False).head(1)['content'].tolist()[0]  
    result_syn['{}-nofil'.format(ft_flag)] = df_tmp.sample(1)['content'].tolist()[0] 

    # clsembed
    torch.cuda.empty_cache()

    embeds_syn = enc.infer(contents_syn)
    embeds_score = cosine_similarity(embeds_syn, enc_dic[row['label']])

    preds = model_cls.predict(np.array(contents_syn),  batch_size=32, verbose=0)
    cls_score = preds[:, row['label'] ]

    df_tmp = pd.DataFrame(zip(contents_syn, list(embeds_score.reshape(-1)), list(cls_score)),\
                 columns=['content', 'embed_score', 'cls_score'])

    result_syn['{}-cls'.format(ft_flag)] = df_tmp.sort_values(by=['cls_score'], ascending=False).head(1)['content'].tolist()[0] 
    result_syn['{}-embed'.format(ft_flag)] = df_tmp.sort_values(by=['embed_score'], ascending=False).head(1)['content'].tolist()[0] 
    

    return result_syn



def synthesize(ds): 
    infos = []
    for ix, row in ds.df_train.reset_index().iterrows():
        torch.cuda.empty_cache()
        print(ix, "of", ds.df_train.shape[0], "ori====>", row['content'], "<===", row['label_name'])

        t0 = time.time()

        result_syn__ft =    clsembednlinsp_gen(row,  'ft')
        result_syn__noft =  clsembednlinsp_gen(row,  'noft')

        for result_syn in [result_syn__ft, result_syn__noft]:
            print("gen===>")
            for fmark, content in result_syn.items():
                print("{} ==>{}".format(fmark, content) )
                infos.append((content, row['label_name'], row['label'], fmark))
            print('\n')
            t1 = time.time()
            print("timecost:", (t1-t0)/60 )

    df_synthesize = pd.DataFrame(infos, columns=['content','label_name','label', 'fmark'])

    print("final generated==>", df_synthesize.shape[0], ds.df_train.shape[0], df_synthesize.shape[0]/ds.df_train.shape[0])

    assert df_synthesize.loc[df_synthesize['fmark']==df_synthesize['fmark'].unique()[0],'label_name'].value_counts().min() >= args.samplecnt
    print(df_synthesize.loc[df_synthesize['fmark']==df_synthesize['fmark'].unique()[0], 'label_name'].value_counts())

    return df_synthesize 



ds.df_train['fmark'] = 'ori'

syn_df_ll = []
for augi in range(args.max_aug_times):
    print("augi==>{}".format(augi))
    df_synthesize = synthesize(ds)
    syn_df_ll.append(df_synthesize)

df_train_aug = pd.concat([ds.df_train] + syn_df_ll ).sample(frac=1)
print("begin_to_test_aug")

for fmark in df_synthesize['fmark'].unique():
    print("fmark:", fmark)
    acc_aug, _ = thread_testing(args.testvalid, df_train_aug.loc[df_train_aug['fmark'].isin(['ori',fmark])], ds.df_test)

    # if acc_noaug > 0:
    #     gain = round((acc_aug - acc_noaug) / acc_noaug * 100, 2)
    # else:
    #     gain = -1

    summary = ['summary===>'] + ['{}:{}'.format(k, v) for k, v in vars(args).items() if not k.startswith('eda_')] + \
        ['fmark:{} acc_base:{} acc_aug:{} '.format(fmark, acc_noaug, acc_aug )]

    # if args.testbed and args.epochs > 10 and gain != -1 :
    #     record_log('log__baselines', summary)
    print('success', ' '.join(summary))
