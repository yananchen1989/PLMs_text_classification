import torch,argparse,time,os,nltk,random,nltk,collections,math
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()
import tensorflow_hub as hub
from transformers import GPT2Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf 
from utils.cbert_cgpt_config import * 
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity

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
parser.add_argument("--boostsample_ft", default=0, type=int)# tune
#parser.add_argument("--txt_in_len", default=5, type=int)
#parser.add_argument("--txt_out_len", default=64, type=int)
parser.add_argument("--ppo_batchsize", default=64, type=int)

#parser.add_argument("--warmup_epoch", default=-1, type=int)
parser.add_argument("--fbs", default=16, type=int)
parser.add_argument("--ppo_epoch", default=100, type=int)
parser.add_argument("--use_query", default=1, type=int)
parser.add_argument("--ref_ft", default=1, type=int)
parser.add_argument("--gpt_ft", default=0, type=int)

parser.add_argument("--init_kl_coef", default=0.2, type=float) 
parser.add_argument("--cliprange", default=0.2, type=float) 
parser.add_argument("--cliprange_value", default=0.2, type=float) 
parser.add_argument("--temperature", default=1.2, type=float) 
parser.add_argument("--min_tokens_to_keep", default=1, type=int) 
parser.add_argument("--external", default=0, type=int) 
parser.add_argument("--ccsample", default=0, type=int) 
parser.add_argument("--steps", default=120, type=int) 

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

gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='./cache', local_files_only=True)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token 


# get dataset
# ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)

# ixl = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
# ixl_rev = {ii[1]:ii[0] for ii in ixl.items()}
# # ds.df_train['label_name'] = ds.df_train['label'].map(lambda x: {0:'negative',1:'positive'}.get(x))
# # ds.df_test['label_name'] = ds.df_test['label'].map(lambda x: {0:'negative',1:'positive'}.get(x))
# print('ixl==>', ixl)
# num_classes = len(ixl)

# ds, proper_len = process_ds(ds, 256)
# print(ds.df_train.sample(10))
# print('proper_len:', proper_len)


# if args.gpt_ft or args.ref_ft:
#     seed = random.sample(list(range(10000)), 1)[0]
#     os.makedirs('fintune_csvs', exist_ok=True)
#     os.makedirs('finetune_gpt2', exist_ok=True)
#     # finetune
#     model_output_path = "./finetune_gpt2/{}_{}_{}".format(args.dsn, args.samplecnt, seed) 
#     os.makedirs(model_output_path, exist_ok=True)

#     train_file = './fintune_csvs/{}_train_finetune_{}_{}.csv'.format(args.dsn, args.samplecnt, seed)
#     validation_file = './fintune_csvs/{}_test_finetune_{}_{}.csv'.format(args.dsn, args.samplecnt, seed)

#     df_train_ft = ds.df_train.copy()
#     df_test_ft = ds.df_test.copy()


#     df_train_ft['ctrl'] = df_train_ft['label_name'].map(lambda x: '[{}]'.format(x) )
#     df_train_ft['text'] = df_train_ft['ctrl'] + df_train_ft['content'] + gpt2_tokenizer.eos_token

#     df_test_ft['ctrl'] = df_test_ft['label_name'].map(lambda x: '[{}]'.format(x) )
#     df_test_ft['text'] = df_test_ft['ctrl'] + df_test_ft['content'] + gpt2_tokenizer.eos_token

#     df_train_ft[['text']].sample(frac=1).to_csv(train_file, index=False)
#     df_test_ft[['text']].sample(frac=1).to_csv(validation_file, index=False)

#     os.system(
#     "CUDA_VISIBLE_DEVICES={} python -u ./run_clm_no_trainer.py \
#             --num_train_epochs 1 \
#             --train_file {} \
#             --validation_file {} \
#             --model_name_or_path gpt2 \
#             --per_device_train_batch_size 16 \
#             --per_device_eval_batch_size 16 \
#             --output_dir {} \
#             --preprocessing_num_workers 16 --overwrite_cache True \
#             --block_size 64".format(args.gpu,  train_file, validation_file, model_output_path) ) 

# else:
model_output_path = 'finetune_gpt2/external_cccnndm_epoch12' #'gpt2'


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
    "batch_size": args.ppo_batchsize ,
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


#. ppo training
#if args.external:
df_train_  = pd.read_csv("./torch_ds/cc_news.csv") #get_cc_news(s)
df_train_ = df_train_.loc[(~df_train_['title'].isnull()) & (~df_train_['content'].isnull()) ]


def pick_sent(para):
    sents = nltk.sent_tokenize(para)
    if len(sents)==1:
        return para
    return random.sample(sents, 1)[0]

from transformers import pipeline
nli_nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=args.gpu)



labels_candidates = ['science', 'technology', 'entertainment', 'health', 'business', \
          'religion' 'sports', 'politics', 'art', 'law','education', 'finance', 'society', 'culture']


def filter_external_df(contents):
    infos = []
    for title in contents:
        #result_milti = nli_nlp(title,  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
        result_only = nli_nlp(title,  labels_candidates, multi_label=False, hypothesis_template="This text is about {}.")
        
        # max_score = max(result['scores']) 
        # max_ix = result['scores'].index(max_score) 
        # pred_label = result['labels'][max_ix]
        infos.append( result_only['scores'])

    nli_embeddings = np.array(infos)
    #df_batch_valid = pd.DataFrame(infos, columns=['content','label_name', 'label'])
    #min_cate_cnt = df_batch_valid['label_name'].value_counts().min()
    #df_batch_valid_sample = sample_stratify(df_batch_valid, min_cate_cnt)
    return nli_embeddings

scce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
even_loss = scce([[0, 1]], [[0.5, 0.5]]).numpy()[0].item()



# accs = []
# for ix, row in ds.df_test.iterrows():
#     result = nli_nlp(row['content'],  labels_candidates, multi_label=False, hypothesis_template="This text is about {}.")
#     if result['scores'][0] >  result['scores'][1]:
#         pred = result['labels'][0]
#     else:
#         pred = result['labels'][1]
#     if pred == row['label_name']:
#         accs.append(1)
#     else:
#         accs.append(0)

#     if len(accs) % 10 ==0:
#         print(sum(accs) / len(accs))

def nli_reward_stsa(reponse ):

    result = nli_nlp(reponse,  labels_candidates, multi_label=False, hypothesis_template="This text is about {}.")
    if reponse == labels_candidates[0]:
        label_vec = [1, 0] 
    else:
        label_vec = [0, 1] 
    score_vec = result['scores']
    reward = even_loss - scce([label_vec], [score_vec]).numpy()[0].item()
    return reward


def nli_reward(response, ori_label):
    if len(reponse) <= 10:
        return -1
    result = nli_nlp(response,  labels_candidates_subs, multi_label=True, hypothesis_template="This text is about {}.")
    max_score = max(result['scores']) 
    max_ix = result['scores'].index(max_score) 
    pred_label = result['labels'][max_ix]

    if pred_label in ['science', 'technology']:
        pred_label = 'science and technology'
    elif pred_label in ['politics','religion']:
        pred_label = 'World' 

    if pred_label == ori_label:
        return  max_score
    else:
        return -1 

kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)


for step in range(args.steps):
    df_batch = df_train_.sample(args.ppo_batchsize)
    torch.cuda.empty_cache()   
    #df_batch['content'] = df_batch['content'].map(lambda x: pick_sent(x))
    nli_embeddings_raw = filter_external_df(df_batch['title'].tolist())

    contents_tokens_lens = [ii.shape[1] for ii in df_batch['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt")).tolist()]    

    maxlen = min(int(sum(contents_tokens_lens) / len(contents_tokens_lens)), 64)

    #padding strategy
    # if args.external:
    #     df_batch['label'] = [random.sample(list(ixl.values()), 1)[0] for _ in range(df_batch.shape[0])]
    #     df_batch['label_name'] = df_batch['label'].map(lambda x: ixl_rev[x])
    #     df_batch['query'] = df_batch['label'].map(lambda x: '[{}]'.format(x)).tolist() + df_batch['content']
    # else:
    df_batch['query'] =  df_batch['content']
    query_tensors = gpt2_tokenizer(df_batch['query'].tolist(), return_tensors="pt", pad_to_max_length =True, \
                                truncation=True, padding=True, max_length=maxlen)['input_ids'].to(device)
        
    response_tensors_ll = []
    for i in range(int(df_batch.shape[0]/args.fbs)):
        response  = respond_to_batch(gpt2_model_trl, query_tensors[i*args.fbs:(i+1)*args.fbs],
                                      txt_len = maxlen, top_p=0.9, \
                                      temperature=args.temperature, min_tokens_to_keep=args.min_tokens_to_keep)
        response_tensors_ll.append(response)
    response_tensors = torch.cat(response_tensors_ll)

    df_batch['response'] = [gpt2_tokenizer.decode(response_tensor, clean_up_tokenization_spaces=True, skip_special_tokens=True).strip() \
                                for response_tensor in response_tensors]


    nli_embeddings_res = filter_external_df(df_batch['response'].tolist())


    y_true = [[0]*(len(labels_candidates)-1) + [1]]
    y_pred = [1 / len(labels_candidates) ] * len(labels_candidates) 
    worst_kl = kl(y_true, y_pred).numpy()[0]

    kls = kl(nli_embeddings_raw, nli_embeddings_res).numpy()

    rewards =  worst_kl-kls
    df_batch['reward'] =rewards

    # train ppo               
    stats = ppo_trainer.step(query_tensors, response_tensors, torch.tensor(rewards).to(device))

    print('step==>', step, 'rewards==>', round(df_batch['reward'].mean(), 4))

    for sent in df_batch.sample(4)['response'].tolist():
        print(sent)
    print()

    

