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
parser.add_argument("--samplecnt", default=8, type=int)
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
parser.add_argument("--gpu", default="7", type=str)

# parser.add_argument("--ddi", default=2, type=int)
# parser.add_argument("--di", default=2, type=int)



# parser.add_argument("--encm", default='dan', type=str, \
#      choices=['dan', 'cmlm', \
#      'paraphrase-distilroberta-base-v2','paraphrase-mpnet-base-v2','paraphrase-TinyBERT-L6-v2',\
#      'paraphrase-MiniLM-L3-v2', 'average_word_embeddings_glove.6B.300d','nli-distilroberta-base-v2'])

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
from utils.cbert_cgpt_config import * 
#from utils.dpp_model import * 
from utils.flair_ners import *

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

if args.testbed:
    print("begin_to_test_noaug")
    acc_noaug, model_cls = thread_testing(args.testvalid, ds.df_train, ds.df_test)
else:
    acc_noaug = -1


if args.aug == 'eda':
    from utils.eda import *

if args.aug == 'generate':
    ####################### generation setting ######################
    #if args.genm == 'gpt':
    from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
    #tokenizer_gpt2.padding_side = "left" 
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
    tokenizer_gpt2.sep_token = '<|sep|>'
    #tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
    print(tokenizer_gpt2)

    if args.genft == 'no':
        gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)

    elif args.genft == 'lambda' :
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
        gpt2 = GPT2LMHeadModel.from_pretrained(model_output_path)

    # elif args.genft == 'cc':
    #     gpt2 = GPT2LMHeadModel.from_pretrained(args.ft_model_path)

    # elif args.genft in ['tc', 'pp', 'ep']:
    #     gpt2 = GPT2LMHeadModel.from_pretrained('ft_model_{}_{}'.format(args.genm, args.genft) )

    else:
        raise KeyError("args.genft illegal!")
    gpt2.trainable = False
    gpt2.config.pad_token_id=50256
    gen_nlp_gpt2  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=len(gpus)-2, return_full_text=False)

    #elif args.genm == 't5':
    from transformers import T5Tokenizer, AutoModelWithLMHead
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
    print(tokenizer_t5)
    if args.genft == 'no':
        t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
    elif args.genft in ['tc', 'pp', 'ep']:
        ft_model_path = 'ft_model_{}_{}'.format(args.genm, args.genft)
        checkpoint_files = glob.glob(ft_model_path+"/checkpoint_loss_*")
        list.sort(checkpoint_files)
        t5 = AutoModelWithLMHead.from_pretrained(checkpoint_files[0])  
    else:
        raise KeyError("args.genft illegal!")

    gen_nlp_t5  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=len(gpus)-2)

    # elif args.genm == 'ctrl':
    #     from transformers import CTRLTokenizer, TFCTRLLMHeadModel
    #     tokenizer_ctrl = CTRLTokenizer.from_pretrained('ctrl', cache_dir='./cache', local_files_only=True)
    #     model_ctrl = TFCTRLLMHeadModel.from_pretrained('ctrl', cache_dir='./cache', local_files_only=True)
    #     print(tokenizer_ctrl)
    #     control_codes = tokenizer_ctrl.control_codes.keys()
    #     gen_nlp_ctrl  = pipeline("text-generation", model=model_ctrl, tokenizer=tokenizer_ctrl, device=len(gpus)-1, return_full_text=False)
 
    # elif args.genm == 'neo':
    #     gen_nlp_gptneo = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=0)

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

    #if 'clsembed' in args.filter or 'dvrl' in args.filter:
    enc = encoder('cmlm-base')
    enc_dic = {}
    for l in ds.df_train['label'].unique():
        contents_ = ds.df_train.loc[ds.df_train['label']==l]['content'].values
        embeds = enc.infer(contents_)
        centroid = embeds.mean(axis=0).reshape(1, -1) 
        enc_dic[l] = centroid

    if 'dvrl' in args.filter:
        os.makedirs('dvrl_np_array', exist_ok=True)
        
    print('filter==> {} model loaded'.format(args.filter))



if args.aug == 'bt':
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer_backward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", cache_dir="./cache", local_files_only=True)
    model_backward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en", cache_dir="./cache", local_files_only=True)
    tokenizer_forward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh", cache_dir="./cache", local_files_only=True)
    model_forward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh", cache_dir="./cache", local_files_only=True)
    nlp_backward = pipeline("translation", model=model_backward, tokenizer=tokenizer_backward, device=len(gpus)-1)
    nlp_forward = pipeline("translation", model=model_forward, tokenizer=tokenizer_forward, device=len(gpus)-1)
    print('bt model loaded')

# if args.aug == 'fillin':
#     from utils.aug_fillinmask import *

if args.aug == 'cbert':
    from utils.cbert_config import * 
    label_list_len = ds.df_test['label_name'].unique().shape[0]
    if label_list_len > 2:
        model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(label_list_len, 768)
        model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)
    model.to(torch.device("cpu"))

# if args.aug == 'cgpt':
#     from utils.cgpt_config import * 

# def nli_classify(generated_text, label_name, expand_label_nli):
#     assert label_name and  expand_label_nli
#     if not generated_text or len(generated_text) <= 10:
#         return 0, -99
#     labels_candidates = []
#     for label_expand in expand_label_nli.values():
#         labels_candidates.extend(label_expand)
#     result = nli_nlp(generated_text,  labels_candidates, multi_label=False, hypothesis_template="This text is about {}.")
#     if result['labels'][0] in expand_label_nli[label_name]:
#         return 1, result['scores'][0]
#     else:
#         return 0, result['scores'][0]

def run_dvrl_thread(dsn, ii, seed):
    os.system('python dvrl_iter.py --dsn {} --ite {} --seed {} '.format(dsn, ii, seed))


def nli_classify(generated_text, label_name, labels_candidates, ln_extend__rev, mode='max'):
    assert label_name and  labels_candidates
    if not generated_text or len(generated_text) <= 10:
        return 0, -99
    result = nli_nlp(generated_text,  list(labels_candidates), multi_label=True, hypothesis_template="This text is about {}.")
    accum_label = {l:[] for l in ds.df_test['label_name'].unique()}

    for l, score in zip(result['labels'], result['scores']):
        accum_label[ln_extend__rev[l]].append(score)

    for l in accum_label.keys():
        if mode == 'mean':
            accum_label[l] = sum(accum_label[l]) / len(accum_label[l])
        elif mode == 'max':
            accum_label[l] = max(accum_label[l])

    pred_label = max(accum_label.items(), key=operator.itemgetter(1))[0]

    score = accum_label[pred_label]
    if mode == 'mean':
        if pred_label == label_name:
            return 1, score 
        else:
            return 0, score
    elif mode == 'max':
        if pred_label == label_name and score > 0.5:
            return 1, score 
        else:
            return 0, score        

def enc_classify(content, ori_label, enc_dic):
    embed = enc.infer([content])
    result = {}
    for l, embeds in enc_dic.items():
        score = cosine_similarity(embed, embeds).mean()
        result[l] = score
    pred_label = max(result, key=result.get)
    if pred_label == ori_label:
        return 1,  result[pred_label]
    else:
        return 0,  result[pred_label]

def bertcls_classify(generated_text, label_name):
    pred = model_cls.predict([generated_text], batch_size=1, verbose=0)  

    pred_label_name = ixl[pred[0].argmax()]
    pred_label_score = float(pred[0].max())

    ori_label_score = float(pred[0][ixl_rev[label_name]])

    if pred_label_name == label_name and pred_label_score >= 0.8:
        assert ori_label_score == pred_label_score
        return 1, ori_label_score
    else:
        return 0, ori_label_score

def dvrl_inner_join(files):

    # files = glob.glob("./dvrl_np_array/csvs_{}/df_train_noise_{}_{}_*_0.9*.csv".format(5013, 'ag', 5013))

    df_retain_ll = []
    for file in files:
        df_tmp = pd.read_csv(file, usecols=['label','content','label_name','groudtruth', 'dve_out'])
        df_tmp.drop_duplicates(['content'], inplace=True)
        df_tmp.sort_values(by=['dve_out'], ascending=False, inplace=True) 

        for ix in range(df_tmp.shape[0]):
            df_block = df_tmp[0:ix]
            if df_block.shape[0] <= 10:
                continue

            recall_0 = df_block.loc[df_block['groudtruth']==0].shape[0] / df_tmp.loc[df_tmp['groudtruth']==0].shape[0]
            recall_1 = df_block.loc[df_block['groudtruth']==1].shape[0] / df_tmp.loc[df_tmp['groudtruth']==1].shape[0]
            if recall_0 >= 0.1 and recall_1 >= 0.5:
                print(ix, "recall 0:{} 1:{}".format(recall_0, recall_1) )
                break 
        df_cut_tmp = df_tmp[:ix]

        del df_cut_tmp['dve_out']

        df_retain_ll.append(df_cut_tmp.loc[df_cut_tmp['groudtruth']==9])


    df_merge = df_retain_ll[0].copy()
    for df_cut_tmp in df_retain_ll:
        print('before:', df_merge.shape[0]) 
        df_merge = pd.merge(df_merge, df_cut_tmp, on=['content','label','label_name','groudtruth'], how='inner')
        print('after:', df_merge.shape[0]) 

    return df_merge

# if not args.testbed:
#     with tf.distribute.MirroredStrategy().scope():
#         model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
#     model_cls.load_weights("./model_cls/model_full_{}.h5".format(args.dsn))   

def lambda_gen(row, gen_nlp, enc, model_cls):
    prompt = decorate_sent(row['content'], row['label_name'])
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
    torch.cuda.empty_cache()

    embeds_syn = enc.infer(contents_syn)
    embeds_score = cosine_similarity(embeds_syn, enc_dic[row['label']])

    preds = model_cls.predict(np.array(contents_syn),  batch_size=32, verbose=0)
    cls_score = preds[:, row['label'] ]

    df_tmp = pd.DataFrame(zip(contents_syn, list(embeds_score.reshape(-1)), list(cls_score)),\
                 columns=['content', 'embed_score', 'cls_score'])

    result_syn = {}
    result_syn['cls'] = df_tmp.sort_values(by=['cls_score'], ascending=False).head(1)['content'].tolist()[0] 
    result_syn['embed'] = df_tmp.sort_values(by=['embed_score'], ascending=False).head(1)['content'].tolist()[0] 
    return result_syn


def nlinsp_gen(row, gen_nlp, nli_nlp, bert_nsp):
    prompt = decorate_sent(row['content'], row['label_name'])
            
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
    torch.cuda.empty_cache()

    # get nli score
    #ners = get_ners(row['content'])

    # if args.genm == 't5' and args.dsn in ['ag','nyt']:
    #     fbs = 16 
    # else:
    #     fbs = 32
    # for ix in range(0, len(contents_syn), fbs):
    #nli_result = nli_nlp(contents_syn[ix:ix+fbs],  [row['label_name']], multi_label=True, hypothesis_template="This text is about {}.")
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
    result_syn['11'] = df_tmp.sort_values(by=['score'], ascending=False).head(1)['content'].tolist()[0] 
    result_syn['10'] = df_tmp.sort_values(by=['nli_score'], ascending=False).head(1)['content'].tolist()[0] 
    result_syn['01'] = df_tmp.sort_values(by=['nsp_score'], ascending=False).head(1)['content'].tolist()[0]  
    result_syn['00'] = df_tmp.sample(1)['content'].tolist()[0] 

    return result_syn

'''
def mc_nlinsp_gen(row, gen_nlp, nli_nlp, bert_nsp):
    # get mc scores
    df_future_sel_ll = []
    contents_syn = []
    fbs_gen = 32
    mc_candidates = args.candidates * 2
    for _ in range(0, mc_candidates //fbs_gen):
        torch.cuda.empty_cache()
        result_gpt = gen_nlp([row['content']], max_length=dsn_maxlen[args.dsn], \
                                        do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                        repetition_penalty=1.2, num_return_sequences= fbs_gen,\
                                        clean_up_tokenization_spaces=True)

        contents_syn_tmp = [remove_str(ii['generated_text']) for ii in result_gpt if ii]
        contents_syn.extend(contents_syn_tmp)

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
    preds = model_cls.predict(np.array(contents_syn_mc_trunk),  batch_size=32, verbose=0) 

    mc_scores_tmp = []
    for j in range(0, len(contents_syn_mc_trunk), args.test_beams):
        pred_mean = preds[j:j+args.test_beams, row['label']].mean()
        mc_scores_tmp.append(pred_mean)
    assert len(mc_scores_tmp) == len(contents_syn)

    df_future = pd.DataFrame(zip(contents_syn, mc_scores_tmp), columns=['content','mc_score'])
    df_future_sel_ll.append(df_future)

    df_future_all = pd.concat(df_future_sel_ll)
    df_future_sel = df_future_all.sort_values(by=['mc_score'], ascending=False).head(args.candidates)

    contents_syn = df_future_sel['content'].tolist()

    print("mc done:", len(contents_syn))
    return contents_syn
'''





def decorate_sent(content, label_name):
    if args.genm == 'gpt':
        if args.genft == 'lambda':
            prompt = '[{}]'.format(label_name) +  ' '.join(content.split(' ')[:3] )

        elif args.genft in ['tc', 'pp']:
            prompt = '{} {}'.format(content, tokenizer_gpt2.sep_token)

        elif args.genft in ['ep']:
            ners = get_ners(content)
            ners_join = '<=>'.join(ners)
            prompt = ners_join + tokenizer_gpt2.sep_token
        else:
            prompt = content

    elif args.genm == 'ctrl':
        prompt = "Links in {}. ".format(content)
        
    elif args.genm == 't5':
        if args.genft in ['ep']:
            ners = get_ners(content)
            ners_join = '<=>'.join(ners)
            prompt = ners_join + tokenizer_t5.eos_token
        else:
            prompt = content + tokenizer_t5.eos_token
    return prompt

def synthesize(ds, proper_len, syn_df_ll, seed):

    if args.aug == 'generate':
        # # nli config
        # if args.dsn in ['ag','uci', 'nyt']:
        #     ln_extend = {}
        #     for l in ds.df_test['label_name'].unique():
        #         ln_extend[l] = expand_label_nli[l]
        #     ln_extend__rev = {}
        #     for i, j in ln_extend.items():
        #         for jj in j:
        #             ln_extend__rev[jj] = i
        #     labels_candidates = set()
        #     for v in ln_extend.values():
        #         labels_candidates.update(v)   
        
        infos = []
        for ix, row in ds.df_train.reset_index().iterrows():
            torch.cuda.empty_cache()
            print(ix, "of", ds.df_train.shape[0], "ori====>", row['content'], "<===", row['label_name'])

            t0 = time.time()
            if args.filter == 'nlinsp':
                if args.genm == 'gpt':
                    result_syn = nlinsp_gen(row, gen_nlp_gpt2, nli_nlp, bert_nsp)
                elif args.genm == 't5':
                    result_syn = nlinsp_gen(row, gen_nlp_t5, nli_nlp, bert_nsp)
            elif args.filter == 'clsembed':
                result_syn = lambda_gen(row, gen_nlp_gpt2, enc, model_cls)

            print("gen===>")
            for fmark, content in result_syn.items():
                print("{} ==>{}".format(fmark, content) )
                infos.append((content, row['label_name'], row['label'], fmark))
            print('\n')
            t1 = time.time()
            print("timecost:", (t1-t0)/60 )

        df_synthesize = pd.DataFrame(infos, columns=['content','label_name','label', 'fmark'])

        print("final generated==>", df_synthesize.shape[0], ds.df_train.shape[0], df_synthesize.shape[0]/ds.df_train.shape[0])
        '''
        if 'dvrl' in args.filter:
            # trim to balance the samples
            #df_syn_tmp = sample_stratify(df_syn_tmp, args.samplecnt * args.abundance)

            # use dvrl to calculate score
            ds.df_train['groudtruth'] = 1
            df_syn_tmp['groudtruth'] = 9

            df_train_valid_noise = pd.concat([ds.df_train,  df_syn_tmp])

            embeds = enc.infer(df_train_valid_noise['content'].values)
            for j in range(embeds.shape[1]):
                df_train_valid_noise['embed_{}'.format(j)] = embeds[:, j]

            if os.path.exists("./dvrl_np_array/csvs_{}".format(seed)):
                shutil.rmtree("./dvrl_np_array/csvs_{}".format(seed))
            os.makedirs("./dvrl_np_array/csvs_{}".format(seed), exist_ok=False)

            df_train_valid_noise.to_csv("./dvrl_np_array/csvs_{}/df_train_valid_noise_{}_{}.csv".format(seed, args.dsn, seed), index=False)

            t0 = time.time()
            valid_files = []
            dvrl_iter = 0
            while True:
                threads = []
                for di in range(args.threads):
                    t = Thread(target=run_dvrl_thread, args=(args.dsn, di+dvrl_iter, seed))
                    t.start()
                    threads.append(t)

                # join all threads
                for t in threads:
                    t.join()
                print("dvrl after join")

                files = glob.glob("./dvrl_np_array/csvs_{}/df_train_noise_{}_{}_*_0.9*.csv".format(seed, args.dsn, seed))
                print("valid_output==>", len(files), files, 'dvrl_iter:', dvrl_iter)

                valid_files.extend(files)
                if len(valid_files) >= args.valid_files_cnt:
                    print("valid_files_cnt OK:", len(valid_files))
                    print("final_valid_output==>",  valid_files)
                    break 
                dvrl_iter += args.threads
            t1 = time.time()
            print("dvrl_cost_sec:", (t1-t0)/3600, "hour" )
            df_syn_tmp = dvrl_inner_join(random.sample(valid_files, args.valid_files_cnt) )
        '''
        assert df_synthesize.loc[df_synthesize['fmark']==df_synthesize['fmark'].unique()[0],'label_name'].value_counts().min() >= args.samplecnt
        print(df_synthesize.loc[df_synthesize['fmark']==df_synthesize['fmark'].unique()[0], 'label_name'].value_counts())




    elif args.aug == 'eda':
        aug_sentences = ds.df_train['content'].map(lambda x: eda(x, alpha_sr=0.2, alpha_ri=0.2, \
                                   alpha_rs=0.2, p_rd=0.2, num_aug=1)).tolist()
        assert len(aug_sentences) == ds.df_train.shape[0]
        contents_syn = [ii[0] for ii in aug_sentences]

    elif args.aug == 'bt':
        contents_syn = []
        fbs = 8
        for i in range(0, ds.df_train.shape[0], fbs):
            contents_trunk = ds.df_train['content'].tolist()[i:i+fbs]
            content_ =  nlp_forward(contents_trunk, truncation=True, \
                       do_sample=True, temperature=0.9, max_length=256, num_return_sequences=1)
            content__ =  nlp_backward([ii['translation_text'] for ii in content_], truncation=True, \
                        do_sample=True, max_length=256, temperature=0.9, num_return_sequences=1 )
            contents_syn.extend([ii['translation_text'] for ii in content__])
            print('translate trunk==>', i, i+fbs, 'of', ds.df_train.shape[0])

    elif args.aug == 'cbert':

        temp_path = "augf__{}_{}".format(args.dsn, args.aug)
        temp_path_ft = "augf__{}_{}_ft".format(args.dsn, args.aug)

        write_for_cbert(ds.df_train, ds.df_test, temp_path, 0)
        write_for_cbert(ds.df_train, ds.df_test, temp_path_ft, 0)

        processor = get_task_processor(args.dsn, temp_path)
        processor_ft = get_task_processor(args.dsn, temp_path_ft)

        label_list = processor.get_labels(args.dsn)
        # load train and dev data
        train_examples = processor.get_train_examples()
        train_examples_ft = processor_ft.get_train_examples()
        dev_examples = processor.get_dev_examples()   

        cbertgpt_batct_size = 8

        train_features =    convert_examples_to_features(train_examples, label_list, proper_len, tokenizer, seed)
        train_features_ft = convert_examples_to_features(train_examples_ft, label_list, proper_len, tokenizer, seed)
        dev_features =      convert_examples_to_features(dev_examples, label_list, proper_len, tokenizer, seed)

        # train data
        train_data = prepare_data(train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=cbertgpt_batct_size)

        train_data_ft = prepare_data(train_features_ft)
        train_sampler_ft = RandomSampler(train_data_ft)
        train_dataloader_ft = DataLoader(train_data_ft, sampler=train_sampler_ft, batch_size=cbertgpt_batct_size)

        #dev data
        dev_data = prepare_data(dev_features)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=cbertgpt_batct_size)

        print("***** Running training {} *****".format(args.aug))
        print("  Num examples = %d", len(train_features))
        print("  Batch size = %d", cbertgpt_batct_size)

        best_dev_loss = float('inf')
        #model_name = './{}_{}_best_{}.pt'.format(args.dsn, seed, args.aug)
        # finetune
        if not syn_df_ll:
            for epoch in trange(3, desc="Epoch"):
                avg_loss = 0.
                model.train()
                for step, batch in enumerate(train_dataloader_ft):
                    batch = tuple(t.to(device0) for t in batch)
                    inputs = {'input_ids': batch[1],
                              'attention_mask': batch[2],
                              'token_type_ids': batch[3],
                              'masked_lm_labels': batch[4] # yanan
                              }
                    outputs = model(**inputs)
                    loss = outputs[0]
                    optimizer.zero_grad()
                    loss.backward()
                    avg_loss += loss.item()
                    optimizer.step()
                    if (step + 1) % 50 == 0:
                        print("avg_loss: {}".format(avg_loss / 50))
                    avg_loss = 0.
                # eval on dev after every epoch
                dev_loss = compute_dev_loss(model, dev_dataloader)
                print("Epoch {}, Dev loss {}".format(epoch, dev_loss))
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    print("Saving model. Best dev so far {}".format(best_dev_loss))
                    #torch.save(model.state_dict(), model_name)

        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=cbertgpt_batct_size)
        #model.load_state_dict(torch.load(model_name))

        print("generate augmentated samples")
        MASK_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        cbert_sample_ratio = 0.3 # tune
        #tsv_writer = csv.writer(save_train_file, delimiter='\t')
        contents_syn = []
        for step, batch in enumerate(train_dataloader):
            model.eval()
            batch = tuple(t.to(device0) for t in batch)
            init_ids, _, input_mask, segment_ids, _ = batch
            input_lens = [sum(mask).item() for mask in input_mask]
            masked_idx = np.squeeze(
                [np.random.randint(0, l, max( int(l * cbert_sample_ratio), 1) ) for l in input_lens])
            for ids, idx in zip(init_ids, masked_idx):
                ids[idx] = MASK_id
            # print('mask tokens', [ii.shape[0] for ii in masked_idx])
            # print('ori tokens', input_lens )
            inputs = {'input_ids': init_ids,
                      'attention_mask': input_mask,
                      'token_type_ids': segment_ids}
            outputs = model(**inputs)
            predictions = outputs[0]  # model(init_ids, segment_ids, input_mask)
            predictions = F.softmax(predictions / 1.0, dim=2)
            for ids, idx, preds, seg in zip(init_ids, masked_idx, predictions, segment_ids):
                preds = torch.multinomial(preds, 1, replacement=True)[idx]
                if len(preds.size()) == 2:
                    preds = torch.transpose(preds, 0, 1)
                for pred in preds:
                    ids[idx] = pred
                    new_str = tokenizer.convert_ids_to_tokens(ids.cpu().numpy())
                    new_str = rev_wordpiece(new_str)
                    #contents_syn.append((new_str, int(label_list[seg[0].item()])  ))
                    contents_syn.append(new_str)
        
    else:
        raise KeyError("args.aug model illegal!")   

    if args.aug in ['eda','bt','cbert']:
        df_synthesize = ds.df_train[['label_name','label']]
        df_synthesize['content'] = contents_syn
        df_synthesize['fmark'] = aug

    return df_synthesize 



ds.df_train['fmark'] = 'ori'
filter_gems = {'clsembed':['gpt'], 'nlinsp':['gpt', 't5']}

for args.aug in ['generate']:
    for args.filter in ['nlinsp', 'clsembed']:
        for args.genm in filter_gems[args.filter]:
            print("=====>aug:{} filter:{} genm:{} <=====".format(args.aug, args.filter, args.genm))
            syn_df_ll = []
            for augi in range(args.max_aug_times):
                print("augi==>{}".format(augi))
                df_synthesize = synthesize(ds, proper_len, syn_df_ll, args.seed)
                syn_df_ll.append(df_synthesize)
      
            df_train_aug = pd.concat([ds.df_train] + syn_df_ll ).sample(frac=1)
            print("begin_to_test_aug")

            for fmark in df_synthesize['fmark'].unique():
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
