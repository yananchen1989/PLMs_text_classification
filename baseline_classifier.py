import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math,time
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
from utils.eda import *
import nltk 
#nltk.download('wordnet')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

parser = argparse.ArgumentParser()
parser.add_argument("--aug", default="no", type=str)
parser.add_argument("--dsn", default="", type=str)
parser.add_argument("--samplecnt", default=128, type=int)
#parser.add_argument("--ite", default=5, type=int)
parser.add_argument("--lang", default="zh", type=str)
#parser.add_argument("--generate_m", default="gpt2", type=str)
parser.add_argument("--batch_size", default=32, type=int)
#parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--model", default="albert", type=str)
#parser.add_argument("--verbose", default=1, type=int)
parser.add_argument("--checkmode", default="rank", type=str) # rank or thres
parser.add_argument("--beams", default=100, type=int)
parser.add_argument("--rp", default=1.0, type=float)
parser.add_argument("--check", default='enc', type=str, choices=['nli', 'enc', 'no','self'])
parser.add_argument("--enc_m", default='dan', type=str, choices=['dan','cmlm','distil'])
#parser.add_argument("--nli_m", default="joeddav/bart-large-mnli-yahoo-answers", type=str)
#parser.add_argument("--thres", default=0.65, type=float)
#parser.add_argument("--times", default=2, type=int)
parser.add_argument("--cap3rd", default=0.99, type=float)
parser.add_argument("--trunk_size", default=50, type=int)
#parser.add_argument("--dpp", default=0, type=int)
#parser.add_argument("--dpp_retain", default=0.7, type=float)
parser.add_argument("--max_aug_times", default=10, type=int)
parser.add_argument("--setbase", default=1, type=int)
parser.add_argument("--maxlen", default=500, type=int)
parser.add_argument("--eda_times", required=False, type=int, default=1, help="number of augmented sentences per original sentence")
parser.add_argument("--eda_sr", required=False, type=float, default=0.2, help="percent of words in each sentence to be replaced by synonyms")
parser.add_argument("--eda_ri", required=False, type=float, default=0.2, help="percent of words in each sentence to be inserted")
parser.add_argument("--eda_rs", required=False, type=float, default=0.2, help="percent of words in each sentence to be swapped")
parser.add_argument("--eda_rd", required=False, type=float, default=0.2, help="percent of words in each sentence to be deleted")


args = parser.parse_args()
print('args==>', args)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)
assert gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type=='cuda'
device = 0

from utils.aug_fillinmask import *
#from aug_generation import * 
#from aug_translation import *
from utils.load_data import * 
from utils.transblock import * 
from utils.encoders import *
from utils.dpp_model import * 


if args.aug == 'generate' and args.check == 'enc':
    enc = encoder(args.enc_m)

if args.aug == 'generate' and args.check == 'nli':
    nlp_nli = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli', device=device) #  1.8.1+cu102

# "facebook/bart-large-mnli"  'joeddav/xlm-roberta-large-xnli'  "joeddav/bart-large-mnli-yahoo-answers"
# https://huggingface.co/facebook/bart-large-mnli

if args.aug == 'generate':
    # if args.generate_m == 'ctrl':
    #     args.rp = 1.2
    nlp  = pipeline("text-generation", model='gpt2', device=device, return_full_text=False)

if args.aug == 'bt':
    tokenizer_backward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-{}-en".format(args.lang), cache_dir="./cache", local_files_only=True)
    model_backward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-{}-en".format(args.lang), cache_dir="./cache", local_files_only=True)
    tokenizer_forward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(args.lang), cache_dir="./cache", local_files_only=True)
    model_forward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(args.lang), cache_dir="./cache", local_files_only=True)
    nlp_backward = pipeline("translation", model=model_backward, tokenizer=tokenizer_backward, device=device)
    nlp_forward = pipeline("translation", model=model_forward, tokenizer=tokenizer_forward, device=device)


# def dpp_rerank(df_simi_filer, enc, dpp_retain):
#     embeds = enc.infer(df_simi_filer['content'].tolist())
#     sorted_ixs = extract_ix_dpp(embeds, df_simi_filer['simi'].values)
#     df_simi_filer_dpp = df_simi_filer.reset_index().iloc[sorted_ixs]
#     dpp_sents = df_simi_filer_dpp['content'].tolist()[:math.ceil(df_simi_filer_dpp.shape[0] * dpp_retain)]
#     return dpp_sents

def synthesize(ds, max_len):
    if args.aug == 'generate':
        contents = ds.df_train['content'].tolist()
        labels = ds.df_train['label'].tolist()
        labels_candidates = list(ds.df_test.label.unique())
        results = []
        for i in range(0, ds.df_train.shape[0], args.trunk_size):
            contents_trunk = contents[i:i+args.trunk_size]
            labels_trunk = labels[i:i+args.trunk_size] 
            results_trunk = nlp(contents_trunk, max_length=max_len, do_sample=True, top_p=0.9, top_k=0, \
                    repetition_penalty=args.rp, num_return_sequences=args.beams)
            results.extend(results_trunk)
            print('generate trunk==>', i, i+args.trunk_size, 'of', ds.df_train.shape[0])

        assert len(results) == ds.df_train.shape[0] and len(results[0]) == args.beams
        infos = []
        print('filtering...')
        for ii in range(ds.df_train.shape[0]):
            if args.check != 'no':
                if args.check in ['enc']:
                    ori_sentence = contents[ii]
                    ori_embed = enc.infer([ori_sentence])
                    syn_sentences = [sentence['generated_text'] for sentence in results[ii]]
                    syn_embeds = enc.infer(syn_sentences)
                    simis = cosine_similarity(ori_embed, syn_embeds)
                    df_simi = pd.DataFrame(zip(syn_sentences, simis[0]), columns=['content','simi'])

                    #df_simi_filer_enc = df_simi_filer

                if args.check in ['nli']:
                    infos_trunk = []
                    for sentence in results[ii]:
                        if not sentence['generated_text']:
                            continue
                        result_nli = nlp_nli(sentence['generated_text'], labels_candidates, multi_label=False, hypothesis_template="This text is about {}.")
                        #if result_nli['scores'][0] >= args.thres and result_nli['labels'][0] == labels[ii]:                    
                        infos_trunk.append((sentence['generated_text'], result_nli['scores'][0], \
                                             result_nli['labels'][0], labels[ii] ))
                    df_simi = pd.DataFrame(infos_trunk, columns=['content','simi','nli_label', 'ori_label'])
                    # print(args.dsn, 'nli ==> left', df_simi_filer.shape[0], 'of', len(results[ii]))  
                    # if df_simi_filer.shape[0] == 0:
                    #     continue                    
                    #df_simi_filer_nli = df_simi_filer
                df_simi.sort_values(by=['simi'], ascending=False, inplace=True)
                if args.checkmode == 'thres':
                    df_simi_filer = df_simi.loc[df_simi['simi']>= args.thres]
                elif args.checkmode =='rank':
                    df_simi_filer = df_simi.head(1)

                assert df_simi_filer.shape[0] == 1
                # print(args.dsn, 'enc ==> left', df_simi_filer.shape[0], 'of', len(results[ii]))  
                # if df_simi_filer.shape[0] == 0:
                #     continue 
                # if args.check == 'double':
                #     # use enc and nli to filter
                #     df_simi_filer = pd.merge(df_simi_filer_enc, df_simi_filer_nli, on='content', how='inner')
                #     if df_simi_filer.shape[0] == 0:
                #         continue 
                #     beta = 0.5
                #     df_simi_filer['simi'] = df_simi_filer['simi_x']*beta + df_simi_filer['simi_y']*(1-beta)
                #     print('dsn:{} double check==>'.format(args.dsn), \
                #             'enc:', df_simi_filer_enc.shape[0], 'nli:', df_simi_filer_nli.shape[0],\
                #            'join:', df_simi_filer.shape[0])

                # if args.dpp:
                #     try:
                #         dpp_sents = dpp_rerank(df_simi_filer, enc, args.dpp_retain)
                #     except:
                #         print('dpp_rerank error==>', df_simi_filer)
                #         continue 
                # else:
                #     dpp_sents = df_simi_filer['content'].tolist()

                for sentence in df_simi_filer['content'].tolist():
                    infos.append((sentence, labels[ii] ))

            elif args.check == 'no':
                #for sentence in results[ii]:
                sentence = random.sample(results[ii], 1)[0]
                infos.append((sentence['generated_text'], labels[ii] ))

            else:
                raise KeyError("args.check illegal!")  


        
    elif args.aug == 'eda':
        aug_sentences = ds.df_train['content'].map(lambda x: eda(x, alpha_sr=args.eda_sr, alpha_ri=args.eda_ri, \
                                   alpha_rs=args.eda_rs, p_rd=args.eda_rd, num_aug=args.eda_times)).tolist()
        ori_labels = ds.df_train['label'].tolist()
        assert len(aug_sentences) == ds.df_train.shape[0] and len(aug_sentences[1]) == args.eda_times \
                and len(aug_sentences) == len(ori_labels)
        infos = []
        for ii in range(len(aug_sentences)):
            for sent in aug_sentences[ii]:
                infos.append((sent, ori_labels[ii]))

    elif args.aug == 'fillin':
        augmentor = fillInmask()
        sentences = ds.df_train['content'].map(lambda x: augmentor.augment(x)).tolist()
        infos = zip(sentences, ds.df_train['label'].tolist())

    elif args.aug == 'bt':
        contents = ds.df_train['content'].tolist()
        labels = ds.df_train['label'].tolist()
        infos = []
        for i in range(0, ds.df_train.shape[0], args.trunk_size):
            contents_trunk = contents[i:i+args.trunk_size]
            labels_trunk = labels[i:i+args.trunk_size]

            content_ =  nlp_forward(contents_trunk, truncation=True, \
                       do_sample=True, temperature=0.9, max_length=max_len, num_return_sequences=1)
            content__ =  nlp_backward([ii['translation_text'] for ii in content_], truncation=True, \
                        do_sample=True, max_length=max_len, temperature=0.9, num_return_sequences=1)
            infos_trunk = list(zip([ii['translation_text'] for ii in content__], labels_trunk ))
            infos.extend(infos_trunk)
            print('translate trunk==>', i, i+args.trunk_size, 'of', ds.df_train.shape[0])
        assert len(infos) == ds.df_train.shape[0]
    else:
        raise KeyError("args.aug model illegal!")        

    df_synthesize = pd.DataFrame(infos, columns = ['content','label'])
    assert df_synthesize.shape[0] == ds.df_train.shape[0]
    return df_synthesize 


ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: truncate(x, args.maxlen))

if args.cap3rd > 1:
    max_len = int(args.cap3rd)
else:
    max_len = get_tokens_len(ds, args.cap3rd)

if args.samplecnt > 0:
    assert ds.df_train['label'].value_counts().min() == args.samplecnt

if args.setbase:
    print("before augmentating")
    ds.df_train_aug = ds.df_train
    best_val_acc_noaug, model_base = do_train_test(ds)
else:
    best_val_acc_noaug = -99

if args.aug == 'no':
    record_log('logg', \
                 ['noaug==> '] + ['{}:{}'.format(k, v) for k, v in vars(args).items() if not k.startswith('eda_')] +\
                      ['noaug_acc:{}'.format(best_val_acc_noaug)])
    os._exit(0)

print("augmentating...")

syn_df_ll = []
accs_iters = []

while 1:

    df_synthesize = synthesize(ds, max_len)
    syn_df_ll.append(df_synthesize)

    ds.df_train_aug = pd.concat([ds.df_train] + syn_df_ll )

    aug_ratio = round(pd.concat(syn_df_ll).shape[0] / ds.df_train.shape[0], 2)
    cur_acc, model_aug = do_train_test(ds)
    accs_iters.append(cur_acc)
    gain = round( (max(accs_iters) - best_val_acc_noaug) / best_val_acc_noaug, 4)

    if aug_ratio >= args.max_aug_times:
        break
   
    # if (len(accs_iters) >= 7 and accs_iters[-1] < accs_iters[-3] and accs_iters[-2] < accs_iters[-3]) \
    #     or aug_ratio>= args.max_aug_times \
    #     or (len(accs_iters) >= 10 and accs_iters[-1] < best_val_acc_noaug and accs_iters[-2] < best_val_acc_noaug):
    #     break

record_log('logb', \
    ['summary==>'] + ['{}:{}'.format(k, v) for k, v in vars(args).items() if not k.startswith('eda_')] + \
    ['baseline_acc {}'.format(best_val_acc_noaug),
    'aug_ratio {}'.format(aug_ratio), \
    'accs_iters {}'.format(' '.join([str(ii) for ii in accs_iters])),  \
    'cur_best_acc {}'.format(max(accs_iters)), \
    'cur_gain {}'.format(gain)
    ])










