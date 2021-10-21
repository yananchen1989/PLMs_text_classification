import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math,time,operator,traceback,shutil
from sklearn import metrics
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# import GPUtil
# GPUtil.showUtilization()
# deviceIDs = [0,1,2,3]
# #deviceIDs = GPUtil.getAvailable(order = 'memory', limit = 4, maxLoad = 1, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
# print("deviceIDs ==> ", deviceIDs)
# assert len(deviceIDs) >= 2

parser = argparse.ArgumentParser()
parser.add_argument("--aug", default="eda", type=str)
parser.add_argument("--dsn", default="ag", type=str, choices=['uci','ag','nyt'])
parser.add_argument("--samplecnt", default=128, type=int)
parser.add_argument("--temp", default=1.0, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--model", default="albert", type=str)
parser.add_argument("--verbose", default=0, type=int)
parser.add_argument("--basemode", default="max", type=str) # rank or thres
parser.add_argument("--beams", default=1, type=int)
parser.add_argument("--abundance", default=1, type=int)

#parser.add_argument("--nlim", default="joeddav/bart-large-mnli-yahoo-answers", type=str)
parser.add_argument("--epochs_ft", default=3, type=int)
parser.add_argument("--trunk_size", default=32, type=int)
parser.add_argument("--epochs", default=100, type=int)
#parser.add_argument("--freq", default=25, type=int)
parser.add_argument("--testbed", default=1, type=int)
#parser.add_argument("--testvalid", default='valid', type=str)
parser.add_argument("--boost", default=0, type=int)

parser.add_argument("--seed", default=333, type=int)

parser.add_argument("--valid_files_cnt", default=16, type=int)
parser.add_argument("--threads", default=64, type=int)

parser.add_argument("--filter", default="dvrl", type=str)
# choices=['nli','cls','no','enc','nsp','dvrl','both']

parser.add_argument("--genm", default="gpt", type=str, choices=['gpt','ctrl', 't5'])
parser.add_argument("--genft", default='no', type=str, choices=['no','lambda','entire','tc','pp', 'ep'])

parser.add_argument("--max_aug_times", default=1, type=int)
#parser.add_argument("--basetry", default=3, type=int)
parser.add_argument("--num_return_sequences", default=4, type=int)
#parser.add_argument("--do_train_test_parallel", default=0, type=int)

parser.add_argument("--gpu", default="0", type=str)

parser.add_argument("--ddi", default=1, type=int)
parser.add_argument("--di", default=3, type=int)



# parser.add_argument("--encm", default='dan', type=str, \
#      choices=['dan', 'cmlm', \
#      'paraphrase-distilroberta-base-v2','paraphrase-mpnet-base-v2','paraphrase-TinyBERT-L6-v2',\
#      'paraphrase-MiniLM-L3-v2', 'average_word_embeddings_glove.6B.300d','nli-distilroberta-base-v2'])

args = parser.parse_args()
print('args==>', args)
filter_list = args.filter.split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


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

if args.aug not in ['cgpt','cbert']:
    assert gpus
device0 = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
assert device0.type=='cuda' and device1.type == 'cuda'

from utils.load_data import * 
from utils.transblock import * 
from utils.encoders import *
from utils.cbert_cgpt_config import * 
#from utils.dpp_model import * 

ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)
ds, proper_len = process_ds(ds, 256)
print(ds.df_train.sample(8))
print('proper_len==>', proper_len)
ixl = {ii[0]:ii[1] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
ixl_rev = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
#seed = random.sample(list(range(10000)), 1)[0]

testbed_func = {"test":do_train_test_thread, "valid":do_train_test_valid_thread}



def thread_testing(testvalid, df_train, df_test):
    best_test_accs = []
    models = []

    for ddi in range(args.ddi):
        threads = []
        for di in range(args.di):
            t = Thread(target=testbed_func[testvalid], args=(df_train, df_test, best_test_accs, models, di + ddi*2, \
                              args.epochs,  args.verbose))
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
    acc_noaug, model_cls = thread_testing('test',  ds.df_train, ds.df_test)
else:
    acc_noaug = -1






if args.aug == 'generate' and args.genft == 'ep':
    from utils.flair_ners import * 
    
if args.aug == 'eda':
    from utils.eda import *

if args.aug == 'generate':
    ####################### generation setting ######################
    if args.genm == 'gpt':
        from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
        tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
        #tokenizer_gpt2.padding_side = "left" 
        tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
        tokenizer_gpt2.sep_token = '<|sep|>'
        #tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
        print(tokenizer_gpt2)

        if args.genft == 'no':
            gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)

        elif args.genft in ['entire', 'lambda']:
            if not os.path.exists('ft_tmp'):
                os.makedirs('ft_tmp')

            train_file = './ft_tmp/{}_train_finetune_{}_{}.txt'.format(args.dsn, args.samplecnt, args.seed)
            validation_file = './ft_tmp/{}_test_finetune_{}_{}.txt'.format(args.dsn,  args.samplecnt, args.seed)

            df_train_ft = ds.df_train.copy()
            df_test_ft = ds.df_test.copy()

            if args.genft == 'lambda':
                df_train_ft['ctrl'] = df_train_ft['label_name'].map(lambda x: '[{}]'.format(x) )
                df_train_ft['text'] = df_train_ft['ctrl'] + df_train_ft['content']

                df_test_ft['ctrl'] = df_test_ft['label_name'].map(lambda x: '[{}]'.format(x) )
                df_test_ft['text'] = df_test_ft['ctrl'] + df_test_ft['content']

            elif args.genft == 'entire':
                df_train_ft['text'] = df_train_ft['content']
                df_test_ft['text'] = df_test_ft['content']

            with open (train_file, 'w') as f:
                f.write(" {} ".format(tokenizer_gpt2.eos_token).join(df_train_ft['text'].tolist()))

            with open (validation_file, 'w') as f:
                f.write(" {} ".format(tokenizer_gpt2.eos_token).join(df_test_ft['text'].tolist()))

            model_output_path = "./ft_tmp/{}_{}_{}".format(args.dsn, args.samplecnt, args.seed) 
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
                    --block_size {}".format(0, 20, train_file, validation_file, model_output_path, 128) ) 
            gpt2 = GPT2LMHeadModel.from_pretrained(model_output_path)

        # elif args.genft == 'cc':
        #     gpt2 = GPT2LMHeadModel.from_pretrained(args.ft_model_path)

        elif args.genft in ['tc', 'pp', 'ep']:
            gpt2 = GPT2LMHeadModel.from_pretrained('ft_model_{}_{}'.format(args.genm, args.genft) )

        gpt2.trainable = False
        gpt2.config.pad_token_id=50256
        gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=0, return_full_text=False)

    elif args.genm == 't5':
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
        gen_nlp  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=0)

    elif args.genm == 'ctrl':
        from transformers import CTRLTokenizer, TFCTRLLMHeadModel
        tokenizer_ctrl = CTRLTokenizer.from_pretrained('ctrl', cache_dir='./cache', local_files_only=True)
        model_ctrl = TFCTRLLMHeadModel.from_pretrained('ctrl', cache_dir='./cache', local_files_only=True)
        print(tokenizer_ctrl)
        control_codes = tokenizer_ctrl.control_codes.keys()
        gen_nlp  = pipeline("text-generation", model=model_ctrl, tokenizer=tokenizer_ctrl, device=0, return_full_text=False)
 
    # elif args.genm == 'neo':
    #     gen_nlp = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=0)

    print('generate model loaded ==>{}'.format(args.genm))

    dsn_maxlen = {'uci':256, 'ag':256, 'nyt':256}

    ####################### filter setting ######################
    if 'nli' in filter_list: 
        #nli_nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=1) #  1.8.1+cu102
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_nli = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli', cache_dir='./cache', local_files_only=True)
        tokenizer_nli = AutoTokenizer.from_pretrained('facebook/bart-large-mnli', cache_dir='./cache', local_files_only=True)
        nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=len(gpus)-1)

    if 'nsp' in filter_list:
        with tf.distribute.MirroredStrategy().scope():
            model_cls_pair  = get_model_nsp(512)
    
    if 'enc' in  filter_list or 'dvrl' in filter_list:
        enc = encoder('cmlm-large')
        enc_dic = {}
        for l in ds.df_train['label'].unique():
            contents_ = ds.df_train.loc[ds.df_train['label']==l]['content'].values
            embeds = enc.infer(contents_)
            enc_dic[l] = embeds

    if 'dvrl' in filter_list:
        os.makedirs('dvrl_np_array', exist_ok=True)
        
    print('filter==> {} model loaded'.format(args.filter))



if args.aug == 'bt':
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer_backward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", cache_dir="./cache", local_files_only=True)
    model_backward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en", cache_dir="./cache", local_files_only=True)
    tokenizer_forward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh", cache_dir="./cache", local_files_only=True)
    model_forward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh", cache_dir="./cache", local_files_only=True)
    nlp_backward = pipeline("translation", model=model_backward, tokenizer=tokenizer_backward, device=0)
    nlp_forward = pipeline("translation", model=model_forward, tokenizer=tokenizer_forward, device=0)
    print('bt model loaded')

if args.aug == 'fillin':
    from utils.aug_fillinmask import *

if args.aug == 'cbert':
    from utils.cbert_config import * 
    label_list_len = ds.df_test['label_name'].unique().shape[0]
    if label_list_len > 2:
        model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(label_list_len, 768)
        model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)
    model.to(device0)

if args.aug == 'cgpt':
    from utils.cgpt_config import * 

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

def nsp_classify(ds, generated_text, label_name):
    result = {}
    for l in ds.df_test['label_name'].unique():
        contents_ori = ds.df_train.loc[ds.df_train['label_name']==l].sample(32)['content'].tolist()
         
        pairs = [[sent, generated_text] for sent in contents_ori]
        pairs_ids = get_ids(pairs, 512, tokenizer_bert )
        preds = model_cls_pair.predict(pairs_ids, batch_size=128)

        nsp_score_reduce = preds[:,0].mean()
        result[l] = nsp_score_reduce

    pred_label = max(result, key=result.get)

    if pred_label == label_name:
        nsp_check, nsp_score = 1, result[pred_label]
    else:
        nsp_check, nsp_score = 0, result[pred_label]
    return nsp_check, nsp_score

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



def synthesize(ds, proper_len, syn_df_ll, seed):
    labels = ds.df_train['label'].tolist()
    if args.genm == 'gpt':
        if args.genft == 'lambda':
            prompts = (ds.df_train['label_name'].map(lambda x: '[{}]'.format(x) ) \
                        + ds.df_train['content'].map(lambda x: ' '.join(x.split(' ')[:3] )) ).tolist()

        elif args.genft in ['tc', 'pp']:
            prompts = ds.df_train['content'].map(lambda x: '{} {}'.format(x, tokenizer_gpt2.sep_token) ).tolist()
        elif args.genft in ['ep']:
            prompts = ds.df_train['content'].map(lambda x: get_ners(x))\
                                .map(lambda x: '{}{}'.format(x, tokenizer_gpt2.sep_token) ).tolist()
        else:
            prompts = ds.df_train['content'].tolist()

    elif args.genm == 'ctrl':
        prompts = ds.df_train['content'].map(lambda x: "Links in {}. ".format(x)).tolist()

    elif args.genm == 't5':
        if args.genft in ['ep']:
            prompts = ds.df_train['content'].map(lambda x: get_ners(x))\
                                .map(lambda x: '{}{}'.format(x, tokenizer_t5.eos_token) ).tolist()
        else:
            prompts = ds.df_train['content'].map(lambda x: '{} {}'.format(x, tokenizer_t5.eos_token)).tolist()

    label_names = ds.df_train['label_name'].tolist()
    
    if args.aug == 'generate':
        # nli config
        ln_extend = {}
        for l in ds.df_test['label_name'].unique():
            ln_extend[l] = expand_label_nli[l]
        ln_extend__rev = {}
        for i, j in ln_extend.items():
            for jj in j:
                ln_extend__rev[jj] = i
        labels_candidates = set()
        for v in ln_extend.values():
            labels_candidates.update(v)

        samples_syn_all = []
        for itr in range(100):     
            results = []
            for i in range(0, ds.df_train.shape[0], args.trunk_size):
                torch.cuda.empty_cache() 
                prompts_trunk = prompts[i:i+args.trunk_size]
                labels_trunk = labels[i:i+args.trunk_size] 
                # prompts_trunk: list of prompts
                if args.genm == 'gpt':
                    #return 32*8
                    results_trunk = gen_nlp(prompts_trunk, max_length=dsn_maxlen[args.dsn], do_sample=True, top_p=0.9, top_k=0, \
                        repetition_penalty=1.0, num_return_sequences=args.num_return_sequences, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                elif args.genm == 'ctrl':
                    #return 32*8
                    results_trunk = gen_nlp(prompts_trunk, max_length=dsn_maxlen[args.dsn], do_sample=True, top_p=0.9, top_k=0, temperature=1, \
                        repetition_penalty=1.2, num_return_sequences=args.num_return_sequences, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                elif args.genm == 't5':
                    results_trunk = []
                    for sent in prompts_trunk:
                        contents_trunk_ = gen_nlp(sent, max_length=dsn_maxlen[args.dsn], do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                            repetition_penalty=1.0, num_return_sequences=args.num_return_sequences, clean_up_tokenization_spaces=True) 
                        results_trunk.append(contents_trunk_)

                results.extend(results_trunk)

                print('generate trunk==>', i, i+args.trunk_size, 'of', ds.df_train.shape[0])
            assert len(results) == ds.df_train.shape[0]

            buffer = []
            for ii in range(ds.df_train.shape[0]):
                for s in results[ii]:
                    generated_text = s['generated_text']
                    if not generated_text or len(tokenizer_bert.encode(generated_text))<= 20 :
                        continue
                    label = labels[ii]
                    label_name = label_names[ii]
                    assert label_name in ds.df_test['label_name'].unique()


                    if 'no' in filter_list:
                        buffer.append((generated_text, label, label_name, 0))
                    else:                    
                        if 'nli' in filter_list:
                            nli_check, nli_score = nli_classify(generated_text, label_name, labels_candidates, ln_extend__rev)
                        else:
                            nli_check, nli_score = 1, 1

                        if 'cls' in filter_list:  
                            cls_check, cls_score =  bertcls_classify(generated_text, label_name)  
                        else:
                            cls_check, cls_score = 1, 1            
                                
                        if 'enc' in filter_list: # acc: 0.82
                            enc_check, enc_score = enc_classify(generated_text, label, enc_dic)
                        else:
                            enc_check, enc_score = 1, 1

                        if 'nsp' in filter_list:
                            nsp_check, nsp_score = nsp_classify(ds, generated_text, label_name)
                        else:
                            nsp_check, nsp_score = 1, 1

                        if nli_check and cls_check and enc_check and nsp_check:
                            buffer.append((generated_text, label, label_name, \
                                            nli_score * cls_score * enc_score * nsp_score ))
                        print("\nfiltering {}==>".format(ii), generated_text.replace('\n',' '), '\n', \
                            'label==>', label_name, '\n', \
                            'judge==>nli{}-cls{}-enc{}-nsp{}____{}-{}-{}-{}'\
                            .format(nli_check, cls_check, enc_check, nsp_check, nli_score, cls_score, enc_score, nsp_score),'\n' )


            samples_syn_all.extend(buffer)
            print('gen_itr:', itr , 'filter_ratio:', len(buffer) / (ds.df_train.shape[0]*args.num_return_sequences) )

            df_syn_tmp = pd.DataFrame(samples_syn_all, columns=['content','label','label_name','score'])
            print(df_syn_tmp['label_name'].value_counts())

            if df_syn_tmp['label_name'].value_counts().values.min() >= args.samplecnt * args.abundance:
                
                if 'dvrl' in filter_list:
                    # trim to balance the samples
                    df_syn_tmp = sample_stratify(df_syn_tmp, args.samplecnt * args.abundance)

                    # use dvrl to calculate score
                    ds.df_train['groudtruth'] = 1
                    df_syn_tmp['groudtruth'] = 9
                    del df_syn_tmp['score']
                    df_train_valid_noise = pd.concat([ds.df_train,  df_syn_tmp])

                    embeds = enc.infer(df_train_valid_noise['content'].values)
                    for j in range(embeds.shape[1]):
                        df_train_valid_noise['embed_{}'.format(j)] = embeds[:, j]

                    if os.path.exists("./dvrl_np_array/csvs_{}".format(seed)):
                        shutil.rmtree("./dvrl_np_array/csvs_{}".format(seed))
                    os.makedirs("./dvrl_np_array/csvs_{}".format(seed), exist_ok=False)

                    df_train_valid_noise.to_csv("./dvrl_np_array/csvs_{}/df_train_valid_noise_{}_{}.csv".format(seed, args.dsn, seed), index=False)

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

                    df_syn_tmp = dvrl_inner_join(random.sample(valid_files, args.valid_files_cnt) )

                df_syn_balance = sample_stratify(df_syn_tmp, min(df_syn_tmp['label'].value_counts().min(), args.samplecnt) )
                print("df_syn_balance ==> of {}".format(args.samplecnt) )
                print(df_syn_balance['label_name'].value_counts())
                samples_syn = df_syn_balance[['content','label']].values
                break
    

    elif args.aug == 'eda':
        aug_sentences = ds.df_train['content'].map(lambda x: eda(x, alpha_sr=0.2, alpha_ri=0.2, \
                                   alpha_rs=0.2, p_rd=0.2, num_aug=1)).tolist()
        assert len(aug_sentences) == ds.df_train.shape[0] and len(aug_sentences) == len(labels)
        samples_syn = []
        for ii in range(len(aug_sentences)):
            for sent in aug_sentences[ii]:
                samples_syn.append((sent, labels[ii]))

    # elif args.aug == 'fillin':
    #     augmentor = fillInmask()
    #     samples_syn = []
    #     for b in range(args.beams):
    #         sentences = ds.df_train['content'].map(lambda x: augmentor.augment(x)).tolist()
    #         samples_syn.extend(list(zip(sentences, labels)))
    #         print('beam:', b)

    elif args.aug == 'bt':
        samples_syn = []
        for i in range(0, ds.df_train.shape[0], args.trunk_size):
            contents_trunk = prompts[i:i+args.trunk_size]
            labels_trunk = labels[i:i+args.trunk_size]

            content_ =  nlp_forward(contents_trunk, truncation=True, \
                       do_sample=True, temperature=0.9, max_length=512, num_return_sequences=1)
            content__ =  nlp_backward([ii['translation_text'] for ii in content_], truncation=True, \
                        do_sample=True, max_length=512, temperature=0.9, num_return_sequences=1 )
            infos_trunk = list(zip([ii['translation_text'] for ii in content__], labels_trunk ))
            samples_syn.extend(infos_trunk)
            print('translate trunk==>', i, i+args.trunk_size, 'of', ds.df_train.shape[0])

    elif args.aug in ['cgpt','cbert']:

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

        if args.aug == 'cgpt':
            train_features =    convert_examples_to_features(train_examples, block_size, tokenizer, seed)
            train_features_ft = convert_examples_to_features(train_examples_ft, block_size, tokenizer, seed)
            dev_features =      convert_examples_to_features(dev_examples, block_size, tokenizer, seed)

        if args.aug == 'cbert':
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

        if args.aug == 'cgpt':
            # finetune
            if not syn_df_ll:
                for epoch in trange(args.epochs_ft, desc="Epoch"):
                    avg_loss = 0.
                    model.train()
                    for step, batch in enumerate(train_dataloader_ft):
                        batch = tuple(t.to(device0) for t in batch)

                        inputs = {'input_ids': batch[0],
                                  'labels': batch[1]}

                        outputs = model(**inputs)
                        loss = outputs[0]
                        # loss = model(input_ids, segment_ids, input_mask, masked_ids)
                        optimizer.zero_grad()
                        loss.backward()
                        avg_loss += loss.item()
                        optimizer.step()
                        model.zero_grad()
                        if (step + 1) % 50 == 0:
                            print("avg_loss: {}".format(avg_loss / 50))
                        # avg_loss = 0.

                    # eval on dev after every epoch
                    dev_loss = compute_dev_loss(model, dev_dataloader)
                    print("Epoch {}, Dev loss {}".format(epoch, dev_loss))
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        print("Saving model. Best dev so far {}".format(best_dev_loss))
                        #torch.save(model.state_dict(), model_name)     

            print("generate augmentated samples")
            samples_syn = [] 
            prefix_size = prefix      
            prefix_text = None
            for ex_index, example in enumerate(train_examples):
                model.eval()
                if prefix_size > 0:
                    prefix_text = " ".join(example.text_a.split(' ')[:prefix_size])
                    prompt = example.label + SEP_TOKEN + prefix_text
                else:
                    prompt = example.label + SEP_TOKEN
                # print('cgpt example.text_a ==>', example.text_a)
                # print('cgpt prompt==>', prompt)
                context_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device0)
                out = model.generate(
                    input_ids=context_tokens,
                    max_length=min(proper_len, tokenizer.model_max_length),
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=args.temp,
                    top_k=0,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    pad_token_id=50256
                )

                out = out[:, len(context_tokens):].tolist()
                #for o in out:
                text = tokenizer.decode(out[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                aug_text = text.split(SEP_TOKEN.lower() )[-1]
                # eosn_index = 128
                # for stop_token in STOP_TOKENS:
                #     idx = text.find(stop_token)
                #     if idx > 0:
                #         eosn_index = min(eosn_index, idx)
                # text = text[: eosn_index]
                # text = text.replace("\n", " ").replace(EOS_TOKEN, ' ').strip()
                # if prefix_size > 0:
                #     text = prefix_text + " " + text
                samples_syn.append((aug_text, int(example.label)) )
                #print('cgpt samples_syn==>', aug_text, '<==', example.label, '\n')


        if args.aug == 'cbert':

            # finetune
            if not syn_df_ll:
                for epoch in trange(args.epochs_ft, desc="Epoch"):
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
            samples_syn = []
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
                predictions = F.softmax(predictions / args.temp, dim=2)

                for ids, idx, preds, seg in zip(init_ids, masked_idx, predictions, segment_ids):
                    preds = torch.multinomial(preds, 1, replacement=True)[idx]
                    if len(preds.size()) == 2:
                        preds = torch.transpose(preds, 0, 1)
                    for pred in preds:
                        ids[idx] = pred
                        new_str = tokenizer.convert_ids_to_tokens(ids.cpu().numpy())
                        new_str = rev_wordpiece(new_str)
                        samples_syn.append((new_str, int(label_list[seg[0].item()])  ))
        
    else:
        raise KeyError("args.aug model illegal!")        
    print('samples_syn done...')
    df_synthesize = pd.DataFrame(samples_syn, columns = ['content','label'])
    df_synthesize['label'] = df_synthesize['label'].astype(int)
    aug_ratio_actual = df_synthesize.shape[0] / ds.df_train.shape[0] #* args.beams 
    print("aug_ratio_actual==>", aug_ratio_actual)
    for ix, row in df_synthesize.iterrows():
        print('final_sample {}==> {}'.format(ixl[row['label']], row['content'].strip().replace('\n',' ') ) )
    return df_synthesize 



# if not args.boost:
# print("augmentating... boost:", args.boost )

syn_df_ll = []
for augi in range(args.max_aug_times):
    print("augi==>{}".format(augi))
    df_synthesize = synthesize(ds, proper_len, syn_df_ll, args.seed)
    syn_df_ll.append(df_synthesize)

df_train_aug = pd.concat([ds.df_train] + syn_df_ll ).sample(frac=1)
print("begin_to_test_aug")


acc_aug, _ = thread_testing('test', df_train_aug, ds.df_test)







if acc_noaug > 0:
    gain = round((acc_aug - acc_noaug) / acc_noaug, 4)
else:
    gain = -1


summary = ['summary===>'] + ['{}:{}'.format(k, v) for k, v in vars(args).items() if not k.startswith('eda_')] + \
    ['acc_base:{} acc_aug:{} gain:{} '.format(acc_noaug, acc_aug, gain )]


record_log('logb', summary)
print('success', ' '.join(summary))
