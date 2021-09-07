import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math,random, time
import numpy as np
import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import accuracy_score
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="yahoo", type=str)
parser.add_argument("--samplecnt", default=32, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--txt_in_len", default=50, type=int) # tune
parser.add_argument("--txt_out_len", default=50, type=int)
parser.add_argument("--ppo_batchsize", default=32, type=int)
parser.add_argument("--ref_ft", default=0, type=int)
parser.add_argument("--gpt_ft", default=0, type=int)
# parser.add_argument("--shuffle_ctrl", default=0, type=int) # tune
# parser.add_argument("--boostsample_ppo", default=0, type=int) # tune
parser.add_argument("--num_train_epochs_ft", default=20, type=int)
parser.add_argument("--boostsample_ft", default=0, type=int)
parser.add_argument("--model", default="albert", type=str)
parser.add_argument("--noeval", default=0, type=int)
args = parser.parse_args()
print('args==>', args)

#os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu )
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type=='cuda'
from utils.load_data import * 
from utils.transblock import * 
from utils.cbert_cgpt_config import * 
#from utils.gan_config import * 

assert gpus

ds = load_data(dataset=args.dsn, samplecnt=args.samplecnt)
ds, max_len = process_ds(ds)


ixl = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
num_classes = len(ixl)
ixl_rev = {ii[1]:ii[0] for ii in ixl.items()}



# get baseline acc
if args.noeval:
    acc_base = -1
else:
    acc_base, _ = do_train_test(ds.df_train, ds.df_test, epochs=20, freq=5, verbose=0, \
                                            basetry=3, samplecnt=args.samplecnt, basemode='max')
    print('best_val_acc_noaug:', acc_base)




# finetune
model_output_path = "./finetune_gpt2_{}_{}".format(args.dsn, args.samplecnt) 
os.makedirs('./fintune_csvs', exist_ok=True)
train_file = './fintune_csvs/{}_train_finetune_{}.csv'.format(args.dsn, args.samplecnt)
validation_file = './fintune_csvs/{}_test_finetune_{}.csv'.format(args.dsn, args.samplecnt)

df_train_ft = ds.df_train.copy()

if args.boostsample_ft > 0:
    df_train_ft_aug = boost_for_ft(df_train_ft, args.boostsample_ft, model=None, use_ent=0)
else:
    df_train_ft_aug = df_train_ft

# df_train_ft['content_prefix'] = df_train_ft['label_name'] + ' {} '.format(gpt2_tokenizer.bos_token) + df_train_ft['content'] + ' {}'.format(gpt2_tokenizer.eos_token)
# df_train_ft['prompt'] = df_train_ft['label_name'] + ' {} '.format(gpt2_tokenizer.bos_token) + df_train_ft['content'].map(lambda x: ' '.join(x.split(' ')[:3]))

df_train_ft_aug['ctrl'] = df_train_ft_aug['label'].map(lambda x: '[{}]'.format(x) )
df_train_ft_aug['content'] = df_train_ft_aug['ctrl'] + df_train_ft_aug['content']

df_train_ft_aug.rename(columns={'content': 'text'}).sample(frac=1).to_csv(train_file, index=False)
df_train_ft_aug.rename(columns={'content': 'text'}).sample(frac=0.2).to_csv(validation_file, index=False)
#for args.num_train_epochs_ft in [3, 5, 7, 10, 12, 15]:
if args.num_train_epochs_ft > 0:
    os.system(
    "CUDA_VISIBLE_DEVICES={} python -u ./run_clm_no_trainer.py \
            --num_train_epochs {} \
            --train_file {} \
            --validation_file {} \
            --model_name_or_path gpt2 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --output_dir {} \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --block_size 256".format(args.gpu, args.num_train_epochs_ft, train_file, train_file, model_output_path) ) 



from transformers import GPT2Tokenizer
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
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='./cache', local_files_only=True)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token 
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

ctrl_str = ["[{}]".format(ii) for ii in ds.df_test['label'].unique().tolist()] # 
ctrl_tokens = dict((s, gpt2_tokenizer.encode(s, return_tensors="pt").squeeze().to(device)) for s in ctrl_str)


def synthesize_from_gptppo(df_batch, gpt2_model_trl):
    min_tokens_len = min([ii.shape[1] for ii in df_batch['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt")).tolist()])
    assert min_tokens_len >= 2
    print('min_tokens_len:', min_tokens_len)
    df_batch['query_tokens'] = df_batch['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt").to(device)[0, :min_tokens_len])
    df_batch['query'] =  df_batch['query_tokens'].map(lambda x: gpt2_tokenizer.decode(x))

    task_list = df_batch['label'].map(lambda x: '[{}]'.format(x)).tolist()

    game_data = {}
    task_tensors = torch.stack([ctrl_tokens[t] for t in task_list])
    query_list = df_batch['query'].tolist()
    game_data['query'] = [t+q for t,q in zip(task_list, query_list)]
    
    query_tensors = torch.stack(df_batch['query_tokens'].tolist())
    query_tensors = torch.cat((task_tensors, query_tensors), axis=1)
    args.txt_out_len = min(get_tokens_len_(df_batch, gpt2_tokenizer), args.txt_out_len)

    #### get response from gpt2

    fbs = 8
    response_tensors = []
    for i in range( int(df_batch.shape[0] / fbs) ):
        response  = respond_to_batch(gpt2_model_trl, query_tensors[i*fbs:(i+1)*fbs],
                                      txt_len = args.txt_out_len, top_p=0.9, temperature=1.5)
        response_tensors.append(response)
    response_tensors = torch.cat(response_tensors)
    
    game_data['response'] = [gpt2_tokenizer.decode(response_tensor, clean_up_tokenization_spaces=True, skip_special_tokens=True) \
                        for response_tensor in response_tensors]
    #### tokenize text for sentiment analysis
    texts = [r for q,r in zip(query_list, game_data['response'])]
    labels_syn = df_batch['label'].tolist()

    return pd.DataFrame(zip(texts, labels_syn), columns=['content','label']), \
           query_tensors, response_tensors

def get_reward(pred, label_syn):
    if pred == label_syn:
        return 0
    elif pred == label_syn - num_classes:
        return 1 
    else:
        return -1


base_accs = []
gan_accs=[]
for epoch in range(args.epochs):
    print("\nStart epoch", epoch)
    ds.df_train = ds.df_train.sample(frac=1)

    ix = 0
    df_batch_syn_ll = []
    df_batch_ori_syn_ll = []
    while ix < ds.df_train.shape[0]:
        df_batch = ds.df_train[ix:ix+args.ppo_batchsize].copy()

        df_batch_syn, query_tensors, response_tensors = synthesize_from_gptppo(df_batch, gpt2_model_trl)

        df_batch_syn['label'] = df_batch_syn['label'] + num_classes
        df_batch_ori_syn = pd.concat([df_batch, df_batch_syn]).sample(frac=1)

        df_batch_syn_ll.append((df_batch_syn, query_tensors, response_tensors))
        df_batch_ori_syn_ll.append(df_batch_ori_syn)
        ix += args.ppo_batchsize


    print('classifier training')
    df_train, df_test = train_test_split(pd.concat(df_batch_ori_syn_ll), test_size=0.1)
    #df_train, df_test = train_test_split(ds.df_train, test_size=0.1)
    
    (x_train, y_train),  (x_test, y_test)= get_keras_data(df_train, df_test)
    model = get_model_bert(y_train.max()+1, 'albert')
    model.fit(
            x_train, y_train, batch_size=16, epochs=7, \
            validation_data=(x_test, y_test), verbose=1, validation_batch_size=64,validation_freq=7,
            callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
        )


    print("ppo training")
    rewards_epoch = []
    for replay in df_batch_syn_ll:
        df_syn = replay[0]
        query_tensors = replay[1]
        response_tensors = replay[2]
        preds = model.predict(df_syn['content'].values.reshape(-1,1), batch_size=64)
        pred_ix = preds.argmax(axis=1)
        rewards = [get_reward(ii[0], ii[1]) for ii in zip(list(pred_ix), df_syn['label'].tolist())]
        rewards_epoch.extend(rewards)
        stats = ppo_trainer.step(query_tensors, response_tensors, torch.tensor(rewards).to(device))
        df_syn['pred'] = pred_ix
        print('df_syn pred==>')
        print(df_syn['pred'].value_counts())

    # evaluate acc of both models
    preds = model.predict(ds.df_test['content'].values.reshape(-1,1), batch_size=64)
    preds_accum =  preds[:,:num_classes] + preds[:,num_classes:]
    pred_ix = preds_accum.argmax(axis=1)
    acc_gan = accuracy_score(ds.df_test['label'].values, pred_ix)
    print('acc_gan:', acc_gan)


    gain = (acc_gan-acc_base ) / acc_base
    print('current==>', 'base acc:', acc_base, 'gan acc:', acc_gan)
    base_accs.append(round(acc_base,4))
    gan_accs.append(round(acc_gan,4))
    print('summary==>', 'base:', max(base_accs), 'gan', max(gan_accs),\
            'rewards:', round(np.array(rewards_epoch).mean(),4))

summary = ['finalsummary===>'] + \
    ['{}:{}'.format(k, v) for k, v in vars(args).items()]  + \
    ['acc_base:{}'.format(max(base_accs)), 'acc_aug:{}'.format(max(gan_accs))]

print('success', ' '.join(summary))







