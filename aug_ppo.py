import torch,argparse,time,os,nltk,random,nltk,collections
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()
import tensorflow_hub as hub
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf 
from sklearn.metrics.pairwise import cosine_distances

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="uci", type=str)
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

#parser.add_argument("--warmup_epoch", default=-1, type=int)
parser.add_argument("--fbs", default=8, type=int)
parser.add_argument("--ppo_epoch", default=100, type=int)

parser.add_argument("--init_kl_coef", default=0.2, type=float) 
parser.add_argument("--cliprange", default=0.2, type=float) 
parser.add_argument("--cliprange_value", default=0.2, type=float) 
parser.add_argument("--temperature", default=1.0, type=float) 
parser.add_argument("--min_tokens_to_keep", default=1, type=int) 
parser.add_argument("--maxlen", default=64, type=int) 
parser.add_argument("--gpu", default="6,7", type=str)
parser.add_argument("--future_steps", default=32, type=int)
args = parser.parse_args()
print('args==>', args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

gpus = tf.config.experimental.list_physical_devices('GPU')

# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


# logical_gpus = tf.config.experimental.list_logical_devices('GPU')
# print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")


print('======>',gpus,'<=======')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# tf.config.experimental.set_virtual_device_configuration(gpus[0], \
#            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11019)])
# tf.config.experimental.set_virtual_device_configuration(gpus[1], \
#            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0)])


#assert gpus
device_1 = torch.device("cuda:{}".format(len(gpus)-2) if torch.cuda.is_available() else "cpu")
device_2 = torch.device("cuda:{}".format(len(gpus)-1) if torch.cuda.is_available() else "cpu")

from transformers import pipeline
from utils.load_data import * 

# get dataset
ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)
#ixl = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
#ixl_rev = {ii[1]:ii[0] for ii in ixl.items()}
# ds.df_train['label_name'] = ds.df_train['label'].map(lambda x: {0:'negative',1:'positive'}.get(x))
# ds.df_test['label_name'] = ds.df_test['label'].map(lambda x: {0:'negative',1:'positive'}.get(x))
#print('ixl==>', ixl)
#num_classes = len(ixl)
from utils.transblock import * 
    
#tf.debugging.set_log_device_placement(True) 
#with tf.device('/GPU:0'):
with tf.distribute.MirroredStrategy().scope():
    model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
    model_cls.load_weights("./model_cls/model_uci.h5")    



from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt

gpt2_model_ref_trl = GPT2HeadWithValueModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
gpt2_model_trl = GPT2HeadWithValueModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
config = {
    # "lm_name": "lvwerra/gpt2-imdb",
    # "ref_lm_name": "lvwerra/gpt2-imdb",
     "cls_model_name": "lvwerra/bert-imdb",
    #"tk_name": "gpt2",
    #"steps": 25600,
    "forward_batch_size": args.forward_batch_size,
    "ppo_epochs": 4,   
    #"txt_in_len": 5,
    #"txt_out_len": 15,
    "batch_size": args.ppo_batchsize ,
    "lr": 1.41e-5,
    "init_kl_coef": args.init_kl_coef,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": args.cliprange,
    "cliprange_value": args.cliprange_value,
    "vf_coef":.1, 
}
gpt2_model_ref_trl.to(device_2)
gpt2_model_trl.to(device_2)
ppo_trainer = PPOTrainer(gpt2_model_trl, gpt2_model_ref_trl, **config)



#from utils.transblock import * 



from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)

gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
gpt2.config.pad_token_id = 50256
gen_nlp_gpt2  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=device_1.index, return_full_text=True)


#. ppo training

def get_loss(result, model_cls):
    x = np.array([ii['generated_text'] for ii in result])
    y = np.array([label] * x.shape[0])
    eval_result = model_cls.evaluate(x, y, batch_size=64, verbose=0) 
    future_loss_0 = eval_result[0]
    return future_loss_0


from collections import deque
memory = deque(maxlen=32)

for epoch in range(args.ppo_epoch):
    ds.df_train = ds.df_train.sample(frac=1)
    for ix, row in ds.df_train.reset_index().iterrows():
        torch.cuda.empty_cache()

        query = row['content']
        label = row['label']
        label_name = row['label_name']
        
         
        query_ids = tokenizer_gpt2.encode(query, return_tensors="pt")
        response_ids  = respond_to_batch(gpt2_model_trl, query_ids.to(device_2), \
                                txt_len=args.future_steps, temperature=args.temperature, top_p=0.9)
        response = tokenizer_gpt2.decode(response_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=True).strip().replace('\n',' ')

        query_response_ids = torch.cat([query_ids, response_ids], dim=-1)
        query_response = tokenizer_gpt2.decode(query_response_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=True).strip().replace('\n',' ')

        result_query = gen_nlp_gpt2([query], max_length=query_ids.shape[1] + args.future_steps, do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                                    repetition_penalty=1.0, num_return_sequences=256, clean_up_tokenization_spaces=True)
        #print("result_query generated")

        result_query_response = gen_nlp_gpt2([query_response], max_length=query_response_ids.shape[1] + args.future_steps, \
                                      do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                                    repetition_penalty=1.0, num_return_sequences=256, clean_up_tokenization_spaces=True)
        #print("result_query_response generated")

        result_response = gen_nlp_gpt2([response], max_length=query_response_ids.shape[1] + args.future_steps, \
                                      do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                                    repetition_penalty=1.0, num_return_sequences=256, clean_up_tokenization_spaces=True)
        #print("result_response generated")

        future_loss_query = get_loss(result_query, model_cls)
        future_loss_query_response = get_loss(result_query_response, model_cls)
        future_loss_response = get_loss(result_response, model_cls)

        print("ori===>", query,  "<===", label_name)
        print("response==>", response)
        print("loss reduction:", future_loss_query-future_loss_query_response, future_loss_query-future_loss_response)
        print("\n")

        loss_diff = future_loss_query-future_loss_query_response + future_loss_query-future_loss_response

        reward = torch.tensor([loss_diff])

        train_stats = ppo_trainer.step(query_ids.to(device_2), response_ids.to(device_2), reward.to(device_2) )
        #print(ix,  'pred:', preds_test.argmax(), 'label', row['label'], 'reward:', round(reward.cpu().numpy()[0],4))
        memory.append(reward.numpy()[0])

        rewards_epoch.append(reward.cpu().numpy()[0])
        if ix % 32 == 0 :
            print('iter_reward:', np.array(memory).mean())

    print("epoch:", epoch, np.array(rewards_epoch).mean())
    rewards_epoch = []



'''

nohup python -u aug_ppo.py --dsn ag --maxlen 64 --init_kl_coef 0.1 --temperature 1.0 > ppo.log & 

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






