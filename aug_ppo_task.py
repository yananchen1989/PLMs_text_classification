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
parser.add_argument("--samplecnt", default=32, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--num_train_epochs_ft", default=1, type=int)
parser.add_argument("--model", default="albert", type=str)
parser.add_argument("--verbose", default=0, type=int)
parser.add_argument("--basemode", default="max", type=str) 
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--basetry", default=3, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--freq", default=20, type=int)
parser.add_argument("--boostmodel", default=1, type=int) # tune
parser.add_argument("--boostsample_ppo", default=0, type=int)# tune
parser.add_argument("--use_ent", default=0, type=int) # tune
parser.add_argument("--boostsample_ft", default=0, type=int)# tune
#parser.add_argument("--txt_in_len", default=5, type=int)
#parser.add_argument("--txt_out_len", default=64, type=int)
parser.add_argument("--ppo_batchsize", default=64, type=int)
parser.add_argument("--load_bert", default=0, type=int)
parser.add_argument("--returnmodel", default=1, type=int)
parser.add_argument("--noeval", default=0, type=int)
#parser.add_argument("--txtin", default="mean", type=str)

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
parser.add_argument("--temperature", default=1.2, type=float) 
parser.add_argument("--min_tokens_to_keep", default=1, type=int) 
parser.add_argument("--res_beams", default=1, type=int) 

args = parser.parse_args()
print('args==>', args)

#os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
assert device.type=='cuda'

from utils.load_data import * 
from utils.transblock import * 
from utils.encoders import *
if gpus:
    enc = encoder('dan', 'cuda')
else:
    enc = encoder('dan', 'cpu')

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='./cache', local_files_only=True)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token 


# get dataset
ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)


ixl = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
ixl_rev = {ii[1]:ii[0] for ii in ixl.items()}
# ds.df_train['label_name'] = ds.df_train['label'].map(lambda x: {0:'negative',1:'positive'}.get(x))
# ds.df_test['label_name'] = ds.df_test['label'].map(lambda x: {0:'negative',1:'positive'}.get(x))
print('ixl==>', ixl)
num_classes = len(ixl)

ds, proper_len = process_ds(ds, 256)
print(ds.df_train.sample(10))
print('proper_len:', proper_len)



if args.load_bert:
    with tf.distribute.MirroredStrategy().scope():
        model = get_model_bert(ds.df_test.label.unique().shape[0])
        model.load_weights("./cls/model_full_{}.h5".format(args.dsn) )
    best_val_acc_noaug = -1
else:
    best_val_acc_noaug, model = do_train_test(ds.df_train, ds.df_test, args.epochs, args.freq, args.verbose, \
                                            args.basetry, args.samplecnt, args.basemode, args.model)
    print('best_val_acc_noaug:', best_val_acc_noaug)
    assert best_val_acc_noaug > 0.6


def get_external_news_purified(model, cols, thres, external_frac):
    df_news = get_external_news(external_frac)
    preds = model.predict(df_news['content'].values, batch_size=256, verbose=1)

    df_news['label'] = preds.argmax(axis=1)
    df_news['label_name'] = df_news['label'].map(lambda x: ixl_rev[x])
    df_news['pred_score'] = preds.max(axis=1)

    df_news_use = df_news.loc[(df_news['pred_score']>=thres)]
    print('df_news_use==>')
    print(df_news_use['label_name'].value_counts())
    min_class_cnt = df_news_use['label_name'].value_counts().min()

    df_news_use_balance = sample_stratify(df_news_use, min_class_cnt)
    df_news_use_balance['content'] = df_news_use_balance['content'].map(lambda x: truncate(x, 300))
    return df_news_use_balance[cols]

if args.add_external_ft or args.add_external_ppo:
    df_news_use_balance = get_external_news_purified(model, list(ds.df_train.columns), \
        args.external_thres, args.external_frac )
    print('df_news_use_balance==>', df_news_use_balance.shape[0])


if args.gpt_ft or args.ref_ft:
    seed = random.sample(list(range(10000)), 1)[0]
    os.makedirs('fintune_csvs', exist_ok=True)
    os.makedirs('finetune_gpt2', exist_ok=True)
    # finetune
    model_output_path = "./finetune_gpt2/{}_{}_{}".format(args.dsn, args.samplecnt, seed) 
    os.makedirs(model_output_path, exist_ok=True)

    train_file = './fintune_csvs/{}_train_finetune_{}_{}.csv'.format(args.dsn, args.samplecnt, seed)
    validation_file = './fintune_csvs/{}_test_finetune_{}_{}.csv'.format(args.dsn, args.samplecnt, seed)

    if args.add_external_ft:
        df_train_ft = pd.concat([ds.df_train.copy(), df_news_use_balance])
    else:
        df_train_ft = ds.df_train.copy()
    df_test_ft = ds.df_test.copy()


    # if args.boostsample_ft > 0:
    #     df_train_ft_aug = boost_for_ft(df_train_ft, args.boostsample_ft, model, use_ent=args.use_ent)
    # else:
    #     df_train_ft_aug = df_train_ft

    # df_train_ft['content_prefix'] = df_train_ft['label_name'] + ' {} '.format(gpt2_tokenizer.bos_token) + df_train_ft['content'] + ' {}'.format(gpt2_tokenizer.eos_token)
    # df_train_ft['prompt'] = df_train_ft['label_name'] + ' {} '.format(gpt2_tokenizer.bos_token) + df_train_ft['content'].map(lambda x: ' '.join(x.split(' ')[:3]))

    df_train_ft['ctrl'] = df_train_ft['label'].map(lambda x: '[{}]'.format(x) )
    df_train_ft['text'] = df_train_ft['ctrl'] + df_train_ft['content'] + gpt2_tokenizer.eos_token

    df_test_ft['ctrl'] = df_test_ft['label'].map(lambda x: '[{}]'.format(x) )
    df_test_ft['text'] = df_test_ft['ctrl'] + df_test_ft['content'] + gpt2_tokenizer.eos_token

    if args.external_cnt > 0:
        df_cnndm = pd.read_csv("./torch_ds/cnndm.csv", nrows=args.external_cnt)
        df_cc = pd.read_csv("./torch_ds/cc_news.csv", nrows=args.external_cnt)
        del df_cc['title']
        df_external = pd.concat([df_cnndm, df_cc])
        df_external['content'] = df_external['content'].map(lambda x: x.replace('\n',' ')) + gpt2_tokenizer.eos_token
        df_external['tokens_cnt'] = df_external['content'].map(lambda x: len(x.split(' ')))
        df_external = df_external.loc[df_external['tokens_cnt']>=50]
        print('cnndmcc shape:', df_external.shape[0])
        del df_external['tokens_cnt']
        df_train_ft = pd.concat([df_train_ft, df_external.rename(columns={'content': 'text'})])

    if args.internal_cnt > 0:
        ds_ = load_data(dataset=args.dsn, samplecnt= -1)
        df_train_ft = pd.concat([df_train_ft, ds_.df_train.rename(columns={'content': 'text'})])

    # imdb_str = " <|endoftext|> ".join(df['review'].tolist())
    # with open ('imdb.txt', 'w') as f:
    #     f.write(imdb_str)


    df_train_ft[['text']].sample(frac=1).to_csv(train_file, index=False)
    df_test_ft[['text']].sample(frac=1).to_csv(validation_file, index=False)

    os.system(
    "CUDA_VISIBLE_DEVICES={} python -u ./run_clm_no_trainer.py \
            --num_train_epochs 1 \
            --train_file {} \
            --validation_file {} \
            --model_name_or_path gpt2 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --output_dir {} \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --block_size 256".format(args.gpu,  train_file, validation_file, model_output_path) ) 

else:
    model_output_path = 'gpt2'

#. ppo
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt
config = {
    # "lm_name": "lvwerra/gpt2-imdb",
    # "ref_lm_name": "lvwerra/gpt2-imdb",
     "cls_model_name": "lvwerra/bert-imdb",
    #"tk_name": "gpt2",
    #"steps": 25600,
    "forward_batch_size": 16,
    "ppo_epochs": 4,   
    #"txt_in_len": 5,
    #"txt_out_len": 15,
    "batch_size":args.ppo_batchsize,
    "lr": 1.41e-5,
    "init_kl_coef":args.init_kl_coef,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": args.cliprange,
    "cliprange_value":args.cliprange_value,
    "vf_coef":.1, 
}
def get_ft_performance(df, model):
    df_0 = df.loc[df['label']==0]
    df_1 = df.loc[df['label']==1]
    (x_train_0, y_train_0),  (x_train_1, y_train_1)= get_keras_data(df_0, df_1)
    pred_0 = model.predict(x_train_0, batch_size=64, verbose=0)
    pred_1 = model.predict(x_train_1, batch_size=64, verbose=0)
    # print('neg pred score:',pred_0.mean() )
    # print('pos pred score:',pred_1.mean() )
    return pred_0.mean(), pred_1.mean()

def get_tokens_len_(df, tokenizer):
    tokens_lens = []
    for sent in df['content'].tolist():
        tokens_len = tokenizer.encode(sent, return_tensors='pt')[0].shape[0]
        tokens_lens.append(tokens_len)
    return int(np.quantile(np.array(tokens_lens), 0.9, axis=0))

if args.ref_ft:
    gpt2_model_ref_trl = GPT2HeadWithValueModel.from_pretrained(model_output_path)
else:
    gpt2_model_ref_trl = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_model_ref_trl.to(device)

if args.gpt_ft:
    gpt2_model_trl = GPT2HeadWithValueModel.from_pretrained(model_output_path)
else:
    gpt2_model_trl = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_model_trl.to(device)

ppo_trainer = PPOTrainer(gpt2_model_trl, gpt2_model_ref_trl, **config)

#ctrl_str = ['[negative]', '[positive]'] # 
ctrl_str = ['[{}]'.format(ii) for ii in ixl.values()]
ctrl_tokens = dict((s, gpt2_tokenizer.encode(s, return_tensors="pt").squeeze().to(device)) for s in ctrl_str)

# reward function
def pos_logit_to_reward(logit, task):
    """
    Take the positive sentiment logit and scale it for the task.
        task [negative]: reward = -logit
        task [neutral]: reward = -2*abs(logit)+4
        task [positive]: reward = logit
    """
    for i in range(len(logit)):
        if task[i]=='[negative]':
            logit[i] = -logit[i]
        elif task[i]=='[neutral]':
            logit[i] = -2*torch.abs(logit[i])+4
        elif task[i]=='[positive]':
            pass
        else:
            raise ValueError('task has to be in [0, 1, 2]!')
    return logit

def get_rewards_sentiment(texts, task_list, model):
    preds_test = model.predict(texts, batch_size=64)
    preds_rev = np.log(preds_test) - np.log(1-preds_test)
    preds_rev = preds_rev.reshape(-1)
    pos_logits = torch.tensor(preds_rev)
    rewards = pos_logit_to_reward(pos_logits, task_list)
    return rewards.reshape(-1)

scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
even_loss = scce([[0]], [[1/num_classes]*num_classes ]).numpy()[0].item()

def get_rewards_topics(df, model, spec_col):

    preds_test = model.predict(df[spec_col].values, batch_size=64) # (128, 4)

    loss = scce(df['label'].values, preds_test).numpy()

    rewards = torch.tensor([even_loss-i for i in loss])

    return rewards

#config['txt_in_len'] = proper_len
# ds.df_train['tokens'] = ds.df_train['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt").to(device)[0, :config['txt_in_len']])
# ds.df_train['query'] = ds.df_train['tokens'].map(lambda x: gpt2_tokenizer.decode(x))

# steps = int(np.ceil(ds.df_train.shape[0] / config['batch_size']))
# print('total steps:', steps)
#gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# init model test
result_test = model.evaluate(ds.df_test['content'].values.reshape(-1,1), ds.df_test['label'].values, batch_size=64)
result_train = model.evaluate(ds.df_train['content'].values.reshape(-1,1), ds.df_train['label'].values, batch_size=64)
print('base_model_check_accuracy ==>', result_train[1], result_test[1] )

# pred0_score_gold, pred1_score_gold = get_ft_performance(ds.df_test, model) 
# print('pred0_score_gold:',pred0_score_gold, 'pred1_score_gold', pred1_score_gold, \
#     'goldgap:', pred1_score_gold-pred0_score_gold)

# reward check
for dd in [ds.df_train, ds.df_test]:
    rewards = get_rewards_topics(dd, model, 'content')
    print("{} rewards_upperbond:".format(dd.shape[0]), rewards.cpu().numpy().mean() )

ds.df_train['tokens_cnt'] = ds.df_train['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt")[0].shape[0]).tolist()
print('tokens_cnt ==>', ds.df_train['tokens_cnt'].min(), ds.df_train['tokens_cnt'].max(), ds.df_train['tokens_cnt'].mean())
 # 32 0.77867
 # -1 1.17

if args.add_external_ppo:
    ds.df_train_ = pd.concat([ds.df_train, df_news_use_balance])
else:
    ds.df_train_ = ds.df_train
#. ppo training
gains = []
df_syn_aug_ll = []
for epoch in range(args.ppo_epoch):
    print('\n')
    print('<<<epoch {} begin>>>>'.format(epoch))
    syn_df_ll = []

    ds.df_train_ = ds.df_train_.sample(frac=1)
    ix = 0
    while ix < ds.df_train_.shape[0]:
        torch.cuda.empty_cache()    

        #game_data = dict()
        #### get a batch from the dataset
        #df_batch = ds.df_train.sample(config['batch_size']) 
        df_batch = ds.df_train_[ix:ix+args.ppo_batchsize].copy()
        if args.boostsample_ppo:
            df_batch['content'] = df_batch['content'].map(lambda x: pick_prefix(x))

        contents_tokens_lens = [ii.shape[1] for ii in df_batch['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt")).tolist()]    
        # min_tokens_len = min(contents_tokens_lens)
        # max_tokens_len = max(contents_tokens_lens)
        # mean_tokens_len = int(sum(contents_tokens_lens) / len(contents_tokens_lens))

        # if args.txtin == 'max':
        #     args.txt_in_len = max_tokens_len
        # elif args.txtin == 'mean':
        #     args.txt_in_len = mean_tokens_len
        # elif args.txtin == 'min':
        #     args.txt_in_len = min_tokens_len
        maxlen = min(int(sum(contents_tokens_lens) / len(contents_tokens_lens)), 60)
        # get responses
        '''
        assert min_tokens_len >= 2
        print('cur ix', ix, 'min_tokens_len:', min_tokens_len)

        df_batch['query_tokens'] = df_batch['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt", \
                                truncation=True, padding='max_length', max_length=args.txt_in_len)\
                                .to(device)[0, :min(min_tokens_len,args.txt_in_len)])

        df_batch['query'] =  df_batch['query_tokens'].map(lambda x: gpt2_tokenizer.decode(x))

        task_list = df_batch['label'].map(lambda x: '[{}]'.format(x)).tolist()

        query_tensors = torch.cat((torch.stack([ctrl_tokens[t] for t in task_list]), \
                                   torch.stack(df_batch['query_tokens'].tolist())), axis=1)
        args.txt_out_len = min(get_tokens_len_(df_batch, gpt2_tokenizer), args.txt_out_len)
        '''

        # padding strategy
        df_batch['query'] = df_batch['label'].map(lambda x: '[{}]'.format(x)).tolist() + df_batch['content']
        query_tensors = gpt2_tokenizer(df_batch['query'].tolist(), return_tensors="pt", pad_to_max_length =True, \
                                truncation=True, padding=True, max_length=maxlen+3)['input_ids'].to(device)
        
        #args.txt_out_len = args.txt_in_len
        #### get response from gpt2
        response_tensors_ll = []
        for i in range(int(args.ppo_batchsize/args.fbs)):
            response  = respond_to_batch(gpt2_model_trl, query_tensors[i*args.fbs:(i+1)*args.fbs],
                                          txt_len = maxlen, top_p=0.9, \
                                          temperature=args.temperature, min_tokens_to_keep=args.min_tokens_to_keep)
            response_tensors_ll.append(response)
        response_tensors = torch.cat(response_tensors_ll)

        df_batch['responses'] = [gpt2_tokenizer.decode(response_tensor, clean_up_tokenization_spaces=True, skip_special_tokens=True) \
                                    for response_tensor in response_tensors]

        #texts = [r for q,r in zip(query_list, game_data['response'])]
        
        # cosine distance
        embeds_ori_contents = enc.infer(df_batch['content'].tolist(), batch_size=64 )
        embeds_syn_contents = enc.infer(df_batch['responses'].tolist(), batch_size=64 )
        assert embeds_ori_contents.shape[0] == embeds_syn_contents.shape[0]
        dist_matrix = cosine_distances(embeds_ori_contents, embeds_syn_contents)
        distances = [dist_matrix[j][j] for j in range(df_batch.shape[0])]
        df_batch['cosine_distance'] = distances

        #rewards
        rewards = get_rewards_topics(df_batch, model, 'responses')   
        df_batch['reward'] = rewards.cpu().numpy()

        # train ppo               
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards.to(device))

        # for ctrl in ctrl_str:
        #     df_syn['content'] = df_syn['content'].map(lambda x: x.replace(ctrl, ' '))
        syn_df_ll.append(df_batch)
        #print('ix:', ix, ix+args.ppo_batchsize,'of',ds.df_train.shape[0], 'rewards ==>', rewards.cpu().numpy().mean())
        ix += args.ppo_batchsize
    

    df_syn_epoch = pd.concat(syn_df_ll)

    df_syn_epoch_use = df_syn_epoch.loc[df_syn_epoch['reward']>=even_loss*args.above]

    above_ratio_0p9 = df_syn_epoch_use.shape[0] / df_syn_epoch.shape[0]
    print('above_ratio_0p9 value counts:')
    print(df_syn_epoch_use['label'].value_counts() )

    min_cnt = df_syn_epoch_use['label'].value_counts().min()
    if min_cnt > 0:
        df_syn_epoch_use_sample_ll = [df_syn_epoch_use.loc[df_syn_epoch_use['label']==label].sample(min_cnt, weights='reward') for label in df_syn_epoch_use['label'].unique()]
        df_syn_epoch_use_sample = pd.concat(df_syn_epoch_use_sample_ll).sample(frac=1)
        df_syn_aug_ll.append(df_syn_epoch_use_sample)

    print('summary== epoch:{}  reward:{} cosd:{} above_0p9:{} above_0p9_min:{} current_aug_samples::{} above:{}'.format(epoch, \
        round(df_syn_epoch['reward'].mean().item(), 4), \
        round(df_syn_epoch['cosine_distance'].mean().item(), 4), \
        round(above_ratio_0p9, 4), \
        round(df_syn_epoch_use_sample.shape[0] / df_syn_epoch.shape[0], 4),\
        pd.concat(df_syn_aug_ll).shape[0] , args.above ))

    print('summary== groupby==>')
    for label in df_syn_epoch['label'].unique():
        print(ixl_rev[label], df_syn_epoch.loc[df_syn_epoch['label']==label]['reward'].mean())


    print('<==demostration==>')
    for l in df_syn_epoch.label.unique():
        df_syn_epoch_01 = df_syn_epoch.loc[df_syn_epoch['label']==l].sample(4)
        for ix, row in df_syn_epoch_01.iterrows():
            print('\nlabel:', row['label'], row['label_name'],  \
             'reward:', round(row['reward'],4), 'cos_distance:', round(row['cosine_distance'],4),'\n', \
              'syn==>', row['responses'].replace('\n',' '), '\n', \
              'ori==>', row['content'].replace('\n',' '))

    if epoch % 5 == 0:
        os.makedirs('finetune_gpt2_ppo', exist_ok=True)
        model_output_path_ppo = "./finetune_gpt2_ppo/{}_{}_{}".format(args.dsn, args.samplecnt, seed) 
        os.makedirs(model_output_path_ppo, exist_ok=True)
        gpt2_model_trl.save_pretrained(model_output_path_ppo)


# df_stats = pd.DataFrame(stats_rg, columns=['reward','gap'])
# print('finalsummary', df_stats['reward'].mean(), df_stats['gap'].mean())



if not args.noeval:
    df_train_aug = pd.concat([ds.df_train, df_syn_epoch] )
    aug_ratio = df_syn_epoch.shape[0] / ds.df_train.shape[0]
    acc_aug, model_aug = do_train_test(df_train_aug, ds.df_test, args.epochs, args.freq, args.verbose, \
                                        args.basetry, args.samplecnt, args.basemode, args.model)

    gain = (acc_aug - best_val_acc_noaug) / best_val_acc_noaug
    gains.append(gain)
    print('summary ==> base_acc:', best_val_acc_noaug, 'aug_acc:', acc_aug,\
     'epoch:', epoch, 'cur_gain:', max(gains), 'gains==>', gains)

    # if args.boostmodel==1 and args.load_bert==0  and epoch >= 3 and pred0_score <= 0.25 and pred1_score>=0.85:
    #     model = model_aug



# Model inspection


# os.makedirs('gpt2-imdb-pos')
# gpt2_model_trl.save_pretrained('gpt2-imdb-pos')









