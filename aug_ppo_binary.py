import torch,argparse,time,os,nltk,random,nltk
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()
import tensorflow_hub as hub
from transformers import GPT2Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf 

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="yelp2", type=str)
parser.add_argument("--samplecnt", default=128, type=int)
parser.add_argument("--gpu", default=1, type=int)
parser.add_argument("--num_train_epochs_ft", default=20, type=int)
parser.add_argument("--model", default="albert", type=str)
parser.add_argument("--verbose", default=0, type=int)
parser.add_argument("--basemode", default="max", type=str) 
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--basetry", default=3, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--freq", default=10, type=int)
parser.add_argument("--boostmodel", default=1, type=int) # tune
parser.add_argument("--boostsample_ppo", default=1, type=int)# tune
parser.add_argument("--use_ent", default=0, type=int) # tune
parser.add_argument("--boostsample_ft", default=0, type=int)# tune
parser.add_argument("--txt_in_len", default=5, type=int)
parser.add_argument("--txt_out_len", default=64, type=int)
parser.add_argument("--ppo_batchsize", default=64, type=int)
parser.add_argument("--load_bert", default=0, type=int)
parser.add_argument("--returnmodel", default=1, type=int)
parser.add_argument("--noeval", default=0, type=int)
parser.add_argument("--ft", default=1, type=int)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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
assert gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type=='cuda'

from utils.load_data import * 
from utils.transblock import * 


gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='./cache', local_files_only=True)

ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)



# baseline
ds, proper_len = process_ds(ds)
print(ds.df_train.sample(10))

model = get_model_bert(2, 'albert')

if args.load_bert:  
    model.load_weights("model_{}_full/".format(args.dsn) )

if args.noeval:
    best_val_acc_noaug = -1
else:
    best_val_acc_noaug, model = do_train_test(ds.df_train, ds.df_test)
    print('best_val_acc_noaug:', best_val_acc_noaug)

if args.ft:
    model_output_path = "./finetune_gpt2_{}".format(args.dsn)
    # finetune
    os.makedirs('./fintune_csvs', exist_ok=True)
    train_file = './fintune_csvs/{}_train_finetune.csv'.format(args.dsn)
    validation_file = './fintune_csvs/{}_test_finetune.csv'.format(args.dsn)

    df_train_ft = ds.df_train.copy()

    if args.boostsample_ft > 0:
        df_train_ft_ll = []
        df_train_ft_ll.append(df_train_ft)
        for _ in range(args.boostsample_ft):
            df_train_ft['content'] = df_train_ft['content'].map(lambda x: pick_prefix(x, model, use_ent=args.use_ent))
            df_train_ft_ll.append(df_train_ft)
        df_train_ft = pd.concat(df_train_ft_ll)

    # df_train_ft['content_prefix'] = df_train_ft['label_name'] + ' {} '.format(gpt2_tokenizer.bos_token) + df_train_ft['content'] + ' {}'.format(gpt2_tokenizer.eos_token)
    # df_train_ft['prompt'] = df_train_ft['label_name'] + ' {} '.format(gpt2_tokenizer.bos_token) + df_train_ft['content'].map(lambda x: ' '.join(x.split(' ')[:3]))

    df_train_ft.rename(columns={'content': 'text'}).to_csv(train_file, index=False)
    df_train_ft.rename(columns={'content': 'text'}).to_csv(validation_file, index=False)
    #for args.num_train_epochs_ft in [3, 5, 7, 10, 12, 15]:
    os.system(
        "python -u ./run_clm_no_trainer.py \
                --num_train_epochs {} \
                --train_file {} \
                --validation_file {} \
                --model_name_or_path gpt2 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --output_dir {} \
                --preprocessing_num_workers 16 \
                --block_size 256".format(args.num_train_epochs_ft, train_file, validation_file, model_output_path) ) 

elif not args.ft:
    model_output_path='gpt2'

#. ppo
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt
config = {
    # "lm_name": "lvwerra/gpt2-imdb",
    # "ref_lm_name": "lvwerra/gpt2-imdb",
    # "cls_model_name": "lvwerra/bert-imdb",
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
def get_ft_performance(df, model):
    df_0 = df.loc[df['label']==0]
    df_1 = df.loc[df['label']==1]
    (x_train_0, y_train_0),  (x_train_1, y_train_1)= get_keras_data(df_0, df_1)
    pred_0 = model.predict(x_train_0, batch_size=64, verbose=0)
    pred_1 = model.predict(x_train_1, batch_size=64, verbose=0)
    print('neg pred score:',pred_0.mean() )
    print('pos pred score:',pred_1.mean() )
    return pred_0.mean(), pred_1.mean()

def get_tokens_len(df, tokenizer):
    tokens_lens = []
    for sent in df['content'].tolist():
        tokens_len = tokenizer.encode(sent, return_tensors='pt')[0].shape[0]
        tokens_lens.append(tokens_len)
    return int(np.quantile(np.array(tokens_lens), 0.99, axis=0))

gpt2_model_ref_trl = GPT2HeadWithValueModel.from_pretrained(model_output_path)
gpt2_model_ref_trl.to(device)

gpt2_model_trl_dic = {}
ppo_trainer_dic = {}
for l in ds.df_test.label.unique():
    gpt2_model_trl = GPT2HeadWithValueModel.from_pretrained(model_output_path)
    gpt2_model_trl.to(device)
    gpt2_model_trl_dic[l] = gpt2_model_trl
    ppo_trainer = PPOTrainer(gpt2_model_trl, gpt2_model_ref_trl, **config)
    ppo_trainer_dic[l] = ppo_trainer

ctrl_str = ['[negative]', '[neutral]', '[positive]']
ctrl_tokens = dict((s, gpt2_tokenizer.encode(s, return_tensors="pt").squeeze().to(device)) for s in ctrl_str)

#config['txt_in_len'] = proper_len
# ds.df_train['tokens'] = ds.df_train['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt").to(device)[0, :config['txt_in_len']])
# ds.df_train['query'] = ds.df_train['tokens'].map(lambda x: gpt2_tokenizer.decode(x))

# steps = int(np.ceil(ds.df_train.shape[0] / config['batch_size']))
# print('total steps:', steps)
#gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token


gains = []
for epoch in range(50):
    syn_df_ll = []
    ds.df_train = ds.df_train.sample(frac=1)
    ix = 0
    while ix < ds.df_train.shape[0]:
        torch.cuda.empty_cache()
        #logs = dict()
        game_data = dict()
        
        #### get a batch from the dataset
        #df_batch = ds.df_train.sample(config['batch_size'])
        df_batch = ds.df_train[ix:ix+args.ppo_batchsize].copy()
        if args.boostsample_ppo:
            df_batch['content'] = df_batch['content'].map(lambda x: pick_prefix(x, model, use_ent=args.use_ent))

        min_tokens_len = min([ii.shape[1] for ii in ds.df_train['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt")).tolist()])
        df_batch['tokens'] = df_batch['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt").to(device)[0, :min(min_tokens_len,args.txt_in_len)])
        df_batch['query'] =  df_batch['tokens'].map(lambda x: gpt2_tokenizer.decode(x))

        game_data['query'] = df_batch['query'].tolist()
        query_tensors = torch.stack(df_batch['tokens'].tolist())

        args.txt_out_len = min(get_tokens_len(df_batch, gpt2_tokenizer), args.txt_out_len)
        #### get response from gpt2
        rewards__ = []
        fbs = 8
        for l in [1,0]:
            response_tensors = []
            for i in range(int(args.ppo_batchsize/fbs)):
                response  = respond_to_batch(gpt2_model_trl_dic[l], query_tensors[i*fbs:(i+1)*fbs],
                                              txt_len = args.txt_out_len, top_p=0.9, temperature=1.5)
                response_tensors.append(response)
            response_tensors = torch.cat(response_tensors)

            game_data['response'] = [gpt2_tokenizer.decode(response_tensors[i]) for i in range(args.ppo_batchsize)]
            #timing['time/get_response'] = time.time()-t

            #### tokenize text for sentiment analysis
            #t = time.time()
            texts = [q + r for q,r in zip(game_data['query'], game_data['response'])]
            #sentiment_inputs, attention_masks = build_bert_batch_from_txt(texts, sentiment_tokenizer, device)    
            preds = model.predict(texts, batch_size=64, verbose=0)
            preds = preds.reshape(-1)

            rewards_np = np.log(preds) - np.log(1-preds)
            if l == 1:
                rewards = torch.tensor(rewards_np).to(device)

            elif l == 0:
                rewards = torch.tensor(-rewards_np).to(device)
                        
            stats = ppo_trainer_dic[l].step(query_tensors, response_tensors, rewards)
            rewards__.append(rewards.cpu().numpy().mean())
            # bert check score     
            df_syn = pd.DataFrame(zip(texts, df_batch.shape[0]*[l] ), columns=['content','label'])
            syn_df_ll.append(df_syn)
        print('ix:', ix, ix+args.ppo_batchsize,'of',ds.df_train.shape[0], 'rewards mean==>', sum(rewards__)/2)
        ix += args.ppo_batchsize

    print('epoch:', epoch)
    

    # if epoch <= 2 or sum(rewards__)/2 <= 1:
    #     print('do not evaluate')
    #     continue

    df_syn_epoch = pd.concat(syn_df_ll)
    pred0_score, pred1_score = get_ft_performance(df_syn_epoch, model) 
    

    print('<==demostration==>')
    for l in ds.df_test.label.unique():
        df_syn_epoch_01 = df_syn_epoch.loc[df_syn_epoch['label']==l].sample(10)
        for sent in df_syn_epoch_01['content'].tolist():
            print('label:', l, '==>', sent.replace('\n',' '))

    if not args.noeval:
        df_train_aug = pd.concat([ds.df_train, df_syn_epoch] )
        aug_ratio = df_syn_epoch.shape[0] / ds.df_train.shape[0]
        acc_aug, model_aug = do_train_test(df_train_aug, ds.df_test)
        
        gain = (acc_aug - best_val_acc_noaug) / best_val_acc_noaug
        gains.append(gain)
        print('base_acc:', best_val_acc_noaug, 'aug_acc:', acc_aug, 'epoch:', epoch, 'cur_gain:', max(gains))
        print('gains==>', gains)
        if args.boostmodel==1 and args.load_bert==0  and epoch >= 3 and pred0_score <= 0.25 and pred1_score>=0.85:
            model = model_aug



# Model inspection


# os.makedirs('gpt2-imdb-pos')
# gpt2_model_trl.save_pretrained('gpt2-imdb-pos')









