import torch,argparse,time,os,nltk,random,nltk,collections
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()
import tensorflow_hub as hub
from transformers import GPT2Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf 
from utils.cbert_cgpt_config import * 
from sklearn.metrics.pairwise import cosine_distances

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str)
parser.add_argument("--samplecnt", default=128, type=int)
parser.add_argument("--model", default="albert", type=str)
parser.add_argument("--verbose", default=0, type=int)
parser.add_argument("--basemode", default="max", type=str) 
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--basetry", default=3, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--freq", default=20, type=int)

parser.add_argument("--ppo_batchsize", default=1, type=int)
parser.add_argument("--forward_batch_size", default=1, type=int)
parser.add_argument("--load_bert", default=1, type=int)


parser.add_argument("--external_frac", default=0.9, type=float)
parser.add_argument("--external_thres", default=0.9, type=float)
parser.add_argument("--add_external_ft", default=0, type=int)
parser.add_argument("--add_external_ppo", default=0, type=int)


#parser.add_argument("--warmup_epoch", default=-1, type=int)
parser.add_argument("--fbs", default=8, type=int)
parser.add_argument("--ppo_epoch", default=100, type=int)
parser.add_argument("--use_query", default=1, type=int)
parser.add_argument("--ref_ft", default=1, type=int)
parser.add_argument("--gpt_ft", default=0, type=int)
parser.add_argument("--shuffle_ctrl", default=0, type=int) 
parser.add_argument("--above", default=0.9, type=float) 
parser.add_argument("--external_cnt", default=0, type=int) 
parser.add_argument("--internal_cnt", default=0, type=int) 

parser.add_argument("--init_kl_coef", default=0.2, type=float) 
parser.add_argument("--cliprange", default=0.2, type=float) 
parser.add_argument("--cliprange_value", default=0.2, type=float) 
parser.add_argument("--temperature", default=1.0, type=float) 
parser.add_argument("--min_tokens_to_keep", default=1, type=int) 
parser.add_argument("--maxlen", default=64, type=int) 
parser.add_argument("--ft_pattern", default='no', type=str, choices=['pp', 'tc', 'no'])
args = parser.parse_args()
print('args==>', args)

import GPUtil
GPUtil.showUtilization()
deviceIDs = GPUtil.getAvailable(order = 'memory', limit = 8, maxLoad = 0.99, maxMemory = 0.9, includeNan=False, excludeID=[], excludeUUID=[])

gpus = tf.config.experimental.list_physical_devices('GPU')
print('======>',gpus,'<=======')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)
#assert gpus
device_0 = torch.device("cuda:{}".format(deviceIDs[0]) if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:{}".format(deviceIDs[1]) if torch.cuda.is_available() else "cpu")
assert device_0.type=='cuda'

from utils.load_data import * 
from utils.transblock import * 
#from utils.encoders import *
from utils.ppo_config import * 


# get dataset
ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)


ixl = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
ixl_rev = {ii[1]:ii[0] for ii in ixl.items()}
# ds.df_train['label_name'] = ds.df_train['label'].map(lambda x: {0:'negative',1:'positive'}.get(x))
# ds.df_test['label_name'] = ds.df_test['label'].map(lambda x: {0:'negative',1:'positive'}.get(x))
print('ixl==>', ixl)
num_classes = len(ixl)


ppo_trainer, gpt2_model_trl, gpt2_model_ref_trl = get_ppo_trainer('gpt2', device_0, vars(args))

if args.load_bert:
    
    model = get_model_bert(ds.df_test.label.unique().shape[0])
    model.load_weights("./cls/model_full_{}.h5".format(args.dsn) )
    best_val_acc_noaug = -1
else:
    best_val_acc_noaug, model = do_train_test(ds.df_train, ds.df_test, args.epochs, args.freq, args.verbose, \
                                            args.basetry, args.samplecnt, args.basemode, args.model)
    print('best_val_acc_noaug:', best_val_acc_noaug)
    assert best_val_acc_noaug > 0.6


# def get_external_news_purified(model, cols, thres, external_frac):
#     df_news = get_external_news(external_frac)
#     preds = model.predict(df_news['content'].values, batch_size=256, verbose=1)

#     df_news['label'] = preds.argmax(axis=1)
#     df_news['label_name'] = df_news['label'].map(lambda x: ixl_rev[x])
#     df_news['pred_score'] = preds.max(axis=1)

#     df_news_use = df_news.loc[(df_news['pred_score']>=thres)]
#     print('df_news_use==>')
#     print(df_news_use['label_name'].value_counts())
#     min_class_cnt = df_news_use['label_name'].value_counts().min()

#     df_news_use_balance = sample_stratify(df_news_use, min_class_cnt)
#     df_news_use_balance['content'] = df_news_use_balance['content'].map(lambda x: truncate(x, 300))
#     return df_news_use_balance[cols]

# if args.add_external_ft or args.add_external_ppo:
#     df_news_use_balance = get_external_news_purified(model, list(ds.df_train.columns), \
#         args.external_thres, args.external_frac )
#     print('df_news_use_balance==>', df_news_use_balance.shape[0])



#ctrl_str = ['[negative]', '[positive]'] # 
# ctrl_str = ['[{}]'.format(ii) for ii in ixl.values()]
# ctrl_tokens = dict((s, gpt2_tokenizer.encode(s, return_tensors="pt").squeeze().to(device)) for s in ctrl_str)

# reward function
# def pos_logit_to_reward(logit, task):
#     """
#     Take the positive sentiment logit and scale it for the task.
#         task [negative]: reward = -logit
#         task [neutral]: reward = -2*abs(logit)+4
#         task [positive]: reward = logit
#     """
#     for i in range(len(logit)):
#         if task[i]=='[negative]':
#             logit[i] = -logit[i]
#         elif task[i]=='[neutral]':
#             logit[i] = -2*torch.abs(logit[i])+4
#         elif task[i]=='[positive]':
#             pass
#         else:
#             raise ValueError('task has to be in [0, 1, 2]!')
#     return logit

# def get_rewards_sentiment(texts, task_list, model):
#     preds_test = model.predict(texts, batch_size=64)
#     preds_rev = np.log(preds_test) - np.log(1-preds_test)
#     preds_rev = preds_rev.reshape(-1)
#     pos_logits = torch.tensor(preds_rev)
#     rewards = pos_logit_to_reward(pos_logits, task_list)
#     return rewards.reshape(-1)

scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
even_loss = scce([[0]], [[1/num_classes]*num_classes ]).numpy()[0].item()

# def get_rewards_topics(df, model, spec_col):

#     preds_test = model.predict(df[spec_col].values, batch_size=64) # (128, 4)

#     loss = scce(df['label'].values, preds_test).numpy()

#     rewards = torch.tensor([even_loss-i for i in loss])

#     return rewards

#config['txt_in_len'] = proper_len
# ds.df_train['tokens'] = ds.df_train['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt").to(device)[0, :config['txt_in_len']])
# ds.df_train['query'] = ds.df_train['tokens'].map(lambda x: gpt2_tokenizer.decode(x))

# steps = int(np.ceil(ds.df_train.shape[0] / config['batch_size']))
# print('total steps:', steps)
#gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# init model test
# result_test = model.evaluate(ds.df_test['content'].values.reshape(-1,1), ds.df_test['label'].values, batch_size=64)
# result_train = model.evaluate(ds.df_train['content'].values.reshape(-1,1), ds.df_train['label'].values, batch_size=64)
# print('base_model_check_accuracy ==>', result_train[1], result_test[1] )

# pred0_score_gold, pred1_score_gold = get_ft_performance(ds.df_test, model) 
# print('pred0_score_gold:',pred0_score_gold, 'pred1_score_gold', pred1_score_gold, \
#     'goldgap:', pred1_score_gold-pred0_score_gold)

# reward check
# for dd in [ds.df_train, ds.df_test]:
#     rewards = get_rewards_topics(dd, model, 'content')
#     print("{} rewards_upperbond:".format(dd.shape[0]), rewards.cpu().numpy().mean() )

# ds.df_train['tokens_cnt'] = ds.df_train['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt")[0].shape[0]).tolist()
# print('tokens_cnt ==>', ds.df_train['tokens_cnt'].min(), ds.df_train['tokens_cnt'].max(), ds.df_train['tokens_cnt'].mean())
#  # 32 0.77867
 # -1 1.17
# if args.dsn == 'uci':
#     maxlen = 32 
# elif args.dsn == 'ag':
#     maxlen = 128



# from transformers import pipeline
# nli_nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=deviceIDs[1])


#. ppo training

from collections import deque
memory = deque(maxlen=100)
rewards_epoch = []
for epoch in range(args.ppo_epoch):
    ds.df_train = ds.df_train.sample(frac=1)
    for ix, row in ds.df_train.reset_index().iterrows():
        query = "[{}]".format(row['label']) + row['content']
        query_tensor = gpt2_tokenizer.encode(query, return_tensors="pt").to(device_0)
        response_tensor  = respond_to_batch(gpt2_model_trl, query_tensor, txt_len=args.maxlen)
        response = gpt2_tokenizer.decode(response_tensor[0], clean_up_tokenization_spaces=True, skip_special_tokens=True).strip().replace('\n',' ')

        # get reward from another component
        preds_test = model.predict([response], steps=1) # (128, 4)

        loss = scce(row['label'], preds_test).numpy()

        reward = torch.tensor([even_loss - loss[0]]).to(device_0)

        train_stats = ppo_trainer.step(query_tensor, response_tensor, reward)
        #print(ix,  'pred:', preds_test.argmax(), 'label', row['label'], 'reward:', round(reward.cpu().numpy()[0],4))
        memory.append(reward.cpu().numpy()[0])
        rewards_epoch.append(reward.cpu().numpy()[0])
        if ix % 100 == 0 :
            print('iter_reward:', np.array(memory).mean())
        if ix % 10 == 0:
            print('label==>', row['label_name'])
            print('ori==>', row['content'])
            print('syn==>', response)
            print('\n')
    print("epoch:", epoch, np.array(rewards_epoch).mean())
    rewards_epoch = []

'''

nohup python -u aug_ppo.py --dsn ag --maxlen 64 --init_kl_coef 0.1 > 

'''



# for epoch in range(args.ppo_epoch):
#     print('\n')
#     print('<<<epoch {} begin>>>>'.format(epoch))
#     #syn_df_ll = []

#     ds.df_train = ds.df_train.sample(frac=1)
#     ix = 0
#     while ix < ds.df_train.shape[0]:
#         torch.cuda.empty_cache()    

#         #game_data = dict()
#         #### get a batch from the dataset
#         #df_batch = ds.df_train.sample(config['batch_size']) 
#         df_batch = ds.df_train[ix:ix+args.ppo_batchsize].copy()

#         df_batch['query'] = '[' + df_batch['label_name'].astype(str) + ']' \
#          + df_batch['content'].map(lambda x: ' '.join(x.split(' ')[:4] )) + '{}'.format(gpt2_tokenizer.sep_token)

#         contents_tokens_lens = [ii.shape[1] for ii in df_batch['query'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt")).tolist()]    

#         maxlen_query = min(contents_tokens_lens)

#         #df_batch, query_tensors, response_tensors = reponse_(df_batch, gpt2_model_trl, maxlen, \
#          #               gpt2_tokenizer, device, vars(args))
#         df_batch, query_tensors, response_tensors = reponse(df_batch, gpt2_model_trl, \
#                         maxlen_query, gpt2_tokenizer, device, vars(args))

#         rewards = []
#         for _, row in df_batch.iterrows():
#             result = nli_nlp(row['response'],  list(ds.df_test['label_name'].unique()), \
#                        multi_label=True, hypothesis_template="This text is about {}.")
#             if  result['labels'][0] != row['label_name']:
#                 reward = -1 
#             else:
#                 reward = result['scores'][0]
#             rewards.append(reward)

#         #rewards
#         #rewards = get_rewards_topics(df_batch, model, 'response')   
#         df_batch['reward'] = rewards

#         df_batch['label_name'].value_counts()

#         for cate in ds.df_test['label_name'].unique():
#             print(cate, df_batch.loc[df_batch['label_name']==cate]['reward'].mean())

#         # train ppo               
#         stats = ppo_trainer.step(query_tensors, response_tensors, torch.tensor(rewards).to(device))

#         #syn_df_ll.append(df_batch)
#         print('ix:', ix, 'of', ds.df_train.shape[0], 'epoch', epoch, \
#                  'rewards ==>', df_batch['reward'].mean(), '\n')
#         ix += args.ppo_batchsize
    
#     for ix, row in df_batch.sample(5).iterrows():
#         print(row['label_name'])
#         print("ori==>", row['content'])
#         print("syn==>", row['response'])
#         print()

#     # df_syn_epoch = pd.concat(syn_df_ll)

#     # df_syn_epoch_use = df_syn_epoch.loc[df_syn_epoch['reward']>=even_loss*args.above]

#     # above_ratio_0p9 = df_syn_epoch_use.shape[0] / df_syn_epoch.shape[0]
#     # print('above_ratio_0p9 value counts:')
#     # print(df_syn_epoch_use['label'].value_counts() )

#     # min_cnt = df_syn_epoch_use['label'].value_counts().min()
#     # if min_cnt > 0:
#     #     df_syn_epoch_use_sample_ll = [df_syn_epoch_use.loc[df_syn_epoch_use['label']==label].sample(min_cnt, weights='reward') for label in df_syn_epoch_use['label'].unique()]
#     #     df_syn_epoch_use_sample = pd.concat(df_syn_epoch_use_sample_ll).sample(frac=1)
#     #     df_syn_aug_ll.append(df_syn_epoch_use_sample)

#     # print('summary== epoch:{}  reward:{} cosd:{} above_0p9:{} above_0p9_min:{} current_aug_samples::{} above:{}'.format(epoch, \
#     #     round(df_syn_epoch['reward'].mean().item(), 4), \
#     #     round(df_syn_epoch['cosine_distance'].mean().item(), 4), \
#     #     round(above_ratio_0p9, 4), \
#     #     round(df_syn_epoch_use_sample.shape[0] / df_syn_epoch.shape[0], 4),\
#     #     pd.concat(df_syn_aug_ll).shape[0] , args.above ))

#     # print('summary== groupby==>')
#     # for label in df_syn_epoch['label'].unique():
#     #     print(ixl_rev[label], df_syn_epoch.loc[df_syn_epoch['label']==label]['reward'].mean())


#     # print('<==demostration==>')
#     # for l in df_syn_epoch.label.unique():
#     #     df_syn_epoch_01 = df_syn_epoch.loc[df_syn_epoch['label']==l].sample(4)
#     #     for ix, row in df_syn_epoch_01.iterrows():
#     #         print('\nlabel:', row['label'], row['label_name'],  \
#     #          'reward:', round(row['reward'],4), 'cos_distance:', round(row['cosine_distance'],4),'\n', \
#     #           'syn==>', row['responses'].replace('\n',' '), '\n', \
#     #           'ori==>', row['content'].replace('\n',' '))

#     # if epoch % 5 == 0:
#     #     os.makedirs('finetune_gpt2_ppo', exist_ok=True)
#     #     model_output_path_ppo = "./finetune_gpt2_ppo/{}_{}_{}".format(args.dsn, args.samplecnt, seed) 
#     #     os.makedirs(model_output_path_ppo, exist_ok=True)
#     #     gpt2_model_trl.save_pretrained(model_output_path_ppo)






