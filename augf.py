import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math,time,operator,traceback
from sklearn import metrics
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
parser.add_argument("--freq", default=10, type=int)
parser.add_argument("--testbed", default=0, type=int)

parser.add_argument("--dpp", default=0, type=int)
parser.add_argument("--threads", default=64, type=int)

parser.add_argument("--filter", default='nli', type=str, choices=['nli','cls','no','enc','nsp','dvrl','both'])

parser.add_argument("--genm", default="gpt", type=str, choices=['gpt','ctrl', 't5'])
parser.add_argument("--genft", default='no', type=str, choices=['no','lambda','entire','tc','pp', 'ep'])
parser.add_argument("--ft_model_path", default="", type=str)

parser.add_argument("--max_aug_times", default=1, type=int)
parser.add_argument("--basetry", default=3, type=int)
parser.add_argument("--num_return_sequences", default=8, type=int)

parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--encm", default='dan', type=str, \
     choices=['dan', 'cmlm', \
     'paraphrase-distilroberta-base-v2','paraphrase-mpnet-base-v2','paraphrase-TinyBERT-L6-v2',\
     'paraphrase-MiniLM-L3-v2', 'average_word_embeddings_glove.6B.300d','nli-distilroberta-base-v2'])

args = parser.parse_args()
print('args==>', args)

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
import nltk 
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity
#nltk.download('wordnet')
gpus = tf.config.list_physical_devices('GPU')

import GPUtil
GPUtil.showUtilization()
deviceIDs = GPUtil.getAvailable(order = 'memory', limit = len(gpus), maxLoad = 1, maxMemory = 0.9, includeNan=False, excludeID=[], excludeUUID=[])
print("deviceIDs ==> ", deviceIDs)
assert len(deviceIDs) >= 2


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
device0 = torch.device("cuda:{}".format(deviceIDs[0]) if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:{}".format(deviceIDs[1]) if torch.cuda.is_available() else "cpu")
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
seed = random.sample(list(range(10000)), 1)[0]

if args.testbed or args.filter == 'cls':
    acc_noaug, model_cls = do_train_test(ds.df_train, ds.df_test, args.epochs, args.freq, args.verbose, \
               args.basetry, args.samplecnt, args.basemode, args.model)
else:
    acc_noaug = -1

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

            train_file = './ft_tmp/{}_train_finetune_{}_{}.txt'.format(args.dsn, args.samplecnt, seed)
            validation_file = './ft_tmp/{}_test_finetune_{}_{}.txt'.format(args.dsn,  args.samplecnt, seed)

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

            model_output_path = "./ft_tmp/{}_{}_{}".format(args.dsn, args.samplecnt, seed) 
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
                    --block_size {}".format(deviceIDs[0], 20, train_file, validation_file, model_output_path, 128) ) 
            gpt2 = GPT2LMHeadModel.from_pretrained(model_output_path)

        # elif args.genft == 'cc':
        #     gpt2 = GPT2LMHeadModel.from_pretrained(args.ft_model_path)

        elif args.genft in ['tc', 'pp']:
            gpt2 = GPT2LMHeadModel.from_pretrained('ft_model_{}_{}'.format(args.genm, args.genft) )

        gpt2.trainable = False
        gpt2.config.pad_token_id=50256
        gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=deviceIDs[0], return_full_text=False)

    elif args.genm == 't5':
        from transformers import T5Tokenizer, AutoModelWithLMHead
        tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
        print(tokenizer_t5)
        if args.genft == 'no':
            t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
        elif args.genft in ['tc', 'pp']:
            args.ft_model_path = 'ft_model_{}_{}'.format(args.genm, args.genft)
            checkpoint_files = glob.glob(args.ft_model_path+"/checkpoint_loss_*")
            list.sort(checkpoint_files)
            t5 = AutoModelWithLMHead.from_pretrained(checkpoint_files[0])  
        gen_nlp  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=deviceIDs[0])

    elif args.genm == 'ctrl':
        from transformers import CTRLTokenizer, TFCTRLLMHeadModel
        tokenizer_ctrl = CTRLTokenizer.from_pretrained('ctrl', cache_dir='./cache', local_files_only=True)
        model_ctrl = TFCTRLLMHeadModel.from_pretrained('ctrl', cache_dir='./cache', local_files_only=True)
        print(tokenizer_ctrl)
        control_codes = tokenizer_ctrl.control_codes.keys()
        gen_nlp  = pipeline("text-generation", model=model_ctrl, tokenizer=tokenizer_ctrl, device=deviceIDs[0], return_full_text=False)
 
    # elif args.genm == 'neo':
    #     gen_nlp = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=deviceIDs[0])

    print('generate model loaded ==>{}'.format(args.genm))

    dsn_maxlen = {'uci':50, 'ag':160, 'nyt':300}

    ####################### filter setting ######################
    if args.filter in ['nli', 'both']: 
        #nli_nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=deviceIDs[1]) #  1.8.1+cu102
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_nli = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli', cache_dir='./cache', local_files_only=True)
        tokenizer_nli = AutoTokenizer.from_pretrained('facebook/bart-large-mnli', cache_dir='./cache', local_files_only=True)
        nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=deviceIDs[1])

    if args.filter in ['nsp', 'both']:
        model_cls_pair  = get_model_nsp(min(512, dsn_maxlen[args.dsn]*2))
    
    if args.filter in ['enc','dvrl', 'both']:
        enc = encoder('cmlm-large')

    if args.filter in ['dvrl']:
        if not os.path.exists('dvrl_np_array'):
            os.makedirs('dvrl_np_array')
        from threading import Thread

    print('filter==> {} model loaded'.format(args.filter))



if args.aug == 'bt':
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer_backward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", cache_dir="./cache", local_files_only=True)
    model_backward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en", cache_dir="./cache", local_files_only=True)
    tokenizer_forward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh", cache_dir="./cache", local_files_only=True)
    model_forward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh", cache_dir="./cache", local_files_only=True)
    nlp_backward = pipeline("translation", model=model_backward, tokenizer=tokenizer_backward, device=deviceIDs[0])
    nlp_forward = pipeline("translation", model=model_forward, tokenizer=tokenizer_forward, device=deviceIDs[0])
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
    os.system('python dvrl_iter.py --dsn {} --seed {} --ite {}'.format(dsn, seed, ii))

def nli_classify(generated_text, label_name, labels_candidates, ln_extend__rev):
    assert label_name and  labels_candidates
    if not generated_text or len(generated_text) <= 10:
        return 0, -99
    result = nli_nlp(generated_text,  list(labels_candidates), multi_label=True, hypothesis_template="This text is about {}.")
    accum_label = {l:[] for l in ds.df_test['label_name'].unique()}

    for l, score in zip(result['labels'], result['scores']):
        accum_label[ln_extend__rev[l]].append(score)

    for l in accum_label.keys():
        accum_label[l] = sum(accum_label[l]) / len(accum_label[l])

    pred_label = max(accum_label.items(), key=operator.itemgetter(1))[0]

    score = accum_label[pred_label]
    if pred_label == label_name:
        return 1, score 
    else:
        return 0, score

def bertcls_classify(generated_text, label_name):
    pred = model_cls.predict([generated_text], batch_size=1, verbose=0)  
    pred_name = ds.df_train.loc[ds.df_train['label']==pred[0].argmax()]['label_name'].unique()[0]
    if pred_name == label_name:
        return 1, float(pred[0].max())
    else:
        return 0, float(pred[0].max())

def synthesize(ds, proper_len, syn_df_ll, seed):
    labels = ds.df_train['label'].tolist()
    if args.genm == 'gpt':
        if args.genft == 'lambda':
            contents = (ds.df_train['label_name'].map(lambda x: '[{}]'.format(x) ) \
                        + ds.df_train['content'].map(lambda x: ' '.join(x.split(' ')[:3] )) ).tolist()
        elif args.genft in ['tc', 'pp']:
            contents = ds.df_train['content'].map(lambda x: '{} {}'.format(x, tokenizer_gpt2.sep_token) ).tolist()
        else:
            contents = ds.df_train['content'].tolist()

    elif args.genm == 'ctrl':
        contents = ds.df_train['content'].map(lambda x: "Links in {}. ".format(x)).tolist()

    elif args.genm == 't5':
        contents = ds.df_train['content'].map(lambda x: '{} {}'.format(x, tokenizer_t5.eos_token)).tolist()

    label_names = ds.df_train['label_name'].tolist()
    
    if args.aug == 'generate' and args.genm not in ['neo']:
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
                contents_trunk = contents[i:i+args.trunk_size]
                labels_trunk = labels[i:i+args.trunk_size] 
                # contents_trunk: list of contents
                if args.genm == 'gpt':
                    #return 32*8
                    results_trunk = gen_nlp(contents_trunk, max_length=dsn_maxlen[args.dsn], do_sample=True, top_p=0.9, top_k=0, \
                        repetition_penalty=1.0, num_return_sequences=args.num_return_sequences, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                elif args.genm == 'ctrl':
                    #return 32*8
                    results_trunk = gen_nlp(contents_trunk, max_length=dsn_maxlen[args.dsn], do_sample=True, top_p=0.9, top_k=0, temperature=1, \
                        repetition_penalty=1.2, num_return_sequences=args.num_return_sequences, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                elif args.genm == 't5':
                    results_trunk = []
                    for sent in contents_trunk:
                        contents_trunk_ = gen_nlp(sent, max_length=dsn_maxlen[args.dsn], do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                            repetition_penalty=1.0, num_return_sequences=args.num_return_sequences, clean_up_tokenization_spaces=True) 
                        results_trunk.append(contents_trunk_)

                results.extend(results_trunk)

                print('generate trunk==>', i, i+args.trunk_size, 'of', ds.df_train.shape[0])
            assert len(results) == ds.df_train.shape[0]

            buffer = []
            buffer_both = []
            for ii in range(ds.df_train.shape[0]):
                for s in results[ii]:
                    generated_text = s['generated_text']
                    if not generated_text:
                        continue
                    label = labels[ii]
                    label_name = label_names[ii]
                    assert label_name in ds.df_test['label_name'].unique()
                    if args.filter in ['nli', 'both']:
                        nli_check, nli_score = nli_classify(generated_text, label_name, labels_candidates, ln_extend__rev)
                        if nli_check:
                            buffer.append((generated_text, label, label_name, nli_score))

                    if args.filter in [ 'cls', 'both']:  
                        cls_check, cls_score =  bertcls_classify(generated_text, label_name)  
                        if cls_check:
                            buffer.append((generated_text, label, label_name, cls_score))              
                            
                    if args.filter in [ 'enc', 'both']:
                        content_ori = ds.df_train['content'].tolist()[ii]
                        gen_enc = enc.infer([generated_text])
                        ori_enc = enc.infer([content_ori])
                        enc_score = cosine_similarity(gen_enc, ori_enc)[0][0]
                        if enc_score >= 0.7 :
                            buffer.append((generated_text, label, label_name, enc_score))

                    if args.filter in ['nsp', 'both']:
                        content_ori = ds.df_train['content'].tolist()[ii]
                        pairs = [[content_ori, generated_text]]
                        pairs_ids = get_ids(pairs, min(512, dsn_maxlen[args.dsn]*2), tokenizer_bert )
                        preds = model_cls_pair.predict(pairs_ids, batch_size=32)
                        nsp_score = preds[0][0]
                        if nsp_score >= 0.9:
                            buffer.append((generated_text, label, label_name, nsp_score))  

                    if args.filter == 'both': 
                        if nli_check and cls_check and enc_score>=0.7 and nsp_score >= 0.9:
                            buffer_both.append((generated_text, label, label_name, \
                                nli_score * cls_score * enc_score * nsp_score ))

                    if args.filter in ['no']:
                        buffer.append((generated_text, label, label_name, 0))


            if args.filter == 'both':
                samples_syn_all.extend(buffer_both)
                print('itr:', itr , 'filter ratio:', len(buffer_both) / (ds.df_train.shape[0]*args.num_return_sequences) )

            else:
                samples_syn_all.extend(buffer)
                print('itr:', itr , 'filter ratio:', len(buffer) / (ds.df_train.shape[0]*args.num_return_sequences) )


            df_syn_tmp = pd.DataFrame(samples_syn_all, columns=['content','label','label_name','score'])
            print(df_syn_tmp['label_name'].value_counts())

            if df_syn_tmp['label_name'].value_counts().values.min() >= args.samplecnt * args.abundance:
                
                if args.filter == 'dvrl':
                    # use dvrl to calculate score
                    #df_syn_tmp = dvrl_scoring(df_syn_tmp, ds.df_train, enc, args.dvrl_iter)
                    ds.df_train['groudtruth'] = 1
                    df_syn_tmp['groudtruth'] = 9
                    del df_syn_tmp['score']
                    df_train_valid_noise = pd.concat([ds.df_train,  df_syn_tmp])

                    embeds = enc.infer(df_train_valid_noise['content'].values)
                    for ii in range(embeds.shape[1]):
                        df_train_valid_noise['embed_{}'.format(ii)] = embeds[:, ii]
                    df_train_valid_noise.to_csv("./dvrl_np_array/df_train_valid_noise_{}_{}.csv".format(args.dsn, seed), index=False)

                    threads = []
                    for ii in range(args.threads):
                        t = Thread(target=run_dvrl_thread, args=(args.dsn, ii, seed))
                        t.start()
                        threads.append(t)

                    # join all threads
                    for t in threads:
                        t.join()
                    print("dvrl after join")

                    df_train_noise_files = glob.glob("./dvrl_np_array/df_train_noise_{}_{}_*.csv".format(args.dsn, seed))
                    print("valid output==>", len(df_train_noise_files), df_train_noise_files)

                    ll = []
                    for file in df_train_noise_files:
                        dfi = pd.read_csv(file)
                        auc = float(file.split('_')[-1].replace('.csv','')) 
                        if auc >= 0.8:
                            ll.append(dfi)
                            print(file, auc, dfi.shape[0], dfi['content'].unique().shape[0])

                    assert len(ll) >= 4
                    df_train_syn = pd.concat(ll)
                    df_syn = df_train_syn.loc[df_train_syn['groudtruth']==9]

                    df_syn_agg = df_syn.groupby(['content', 'label', 'label_name'])['dve_out'].mean().reset_index()
                    df_syn_tmp = df_syn_agg.rename(columns={"dve_out": "score"} )
                    
                df_syn_filter_ll = []
                for label_name in df_syn_tmp['label_name'].unique():
                    df_syn_tmp_l = df_syn_tmp.loc[df_syn_tmp['label_name']==label_name].copy()
                    if args.filter=='no':
                        df_syn_tmp_l = df_syn_tmp_l.sample(frac=1)      
                    else:
                        df_syn_tmp_l.sort_values(by=['score'], ascending=False, inplace=True) 
                    # if args.dpp:
                    #     df_syn_tmp_l = dpp_rerank(df_syn_tmp_l, enc)
                    df_syn_filter_ll.append(df_syn_tmp_l.head(args.samplecnt))

                df_syn_filter = pd.concat(df_syn_filter_ll)
                
                samples_syn = [(ii[0],ii[1]) for ii in df_syn_filter[['content','label']].values]
                break
    
    # elif args.aug == 'generate' and args.genm  in ['neo']:
    #     infos = []
    #     while True:
    #         prompt = ''
    #         for ix, row in ds.df_train.sample(16).iterrows():
    #             prompt += "label:{}\ncontent:{}\n###\n".format(row['label_name'], row['content']) 

    #         next_label = ds.df_train.sample(1)['label_name'].tolist()[0]
    #         prompt += "label:{}\ncontent:".format(next_label)
    #         #print(prompt)

    #         gen_text = gen_nlp(prompt, do_sample=True, max_length=1024, top_p=0.9,  \
    #                     clean_up_tokenization_spaces=True, return_full_text=False)
    #         #print('======>')
    #         #print(gen_text[0]['generated_text'])


    #         tokens = gen_text[0]['generated_text'].split('###')

    #         infos.append((next_label, tokens[0].strip()))
    #         for ii in tokens[1:]:
    #             tt = ii.strip().split("\n")
    #             if not tt or len(tt)!=2:
    #                 continue
    #             infos.append((tt[0].split(':')[-1] ,  tt[1].split(':')[-1])) 
    #         df_syn_tmp = pd.DataFrame(infos, columns=['label_name', 'content'])
    #         print('neo step==>')
    #         print(df_syn_tmp['label_name'].value_counts())

    #         if df_syn_tmp['label_name'].value_counts().values.min() >= args.samplecnt * args.abundance:
    #             df_syn_tmp['label'] = df_syn_tmp['label_name'].map(lambda x: ixl_rev[x])
    #             df_syn_filter_ll = []
    #             for label_name in df_syn_tmp['label_name'].unique():
    #                 df_syn_tmp_l = df_syn_tmp.loc[df_syn_tmp['label_name']==label_name].copy()
    #                 #if args.filter=='no':
    #                 df_syn_tmp_l = df_syn_tmp_l.sample(frac=1)
    #                 #else:
    #                 #    df_syn_tmp_l.sort_values(by=['score'], ascending=False, inplace=True) 
    #                 df_syn_filter_ll.append(df_syn_tmp_l.head(args.samplecnt))

    #             df_syn_filter = pd.concat(df_syn_filter_ll)
                
    #             samples_syn = [(ii[0],ii[1]) for ii in df_syn_filter[['content','label']].values]
    #             break
    

    elif args.aug == 'eda':
        aug_sentences = ds.df_train['content'].map(lambda x: eda(x, alpha_sr=0.2, alpha_ri=0.2, \
                                   alpha_rs=0.2, p_rd=0.2, num_aug=1)).tolist()
        assert len(aug_sentences) == ds.df_train.shape[0] and len(aug_sentences[1]) == args.beams \
                and len(aug_sentences) == len(labels)
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
            contents_trunk = contents[i:i+args.trunk_size]
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
    assert df_synthesize.shape[0] == ds.df_train.shape[0] #* args.beams 
    for ix, row in df_synthesize.iterrows():
        print('final_sample {}==> {}'.format(ixl[row['label']], row['content'].strip().replace('\n',' ') ) )
    return df_synthesize 




print("augmentating...")

syn_df_ll = []
for _ in range(args.max_aug_times):
    df_synthesize = synthesize(ds, proper_len, syn_df_ll, seed)
    syn_df_ll.append(df_synthesize)

df_train_aug = pd.concat([ds.df_train] + syn_df_ll )
acc_aug, _ = do_train_test(df_train_aug, ds.df_test, args.epochs, args.freq, args.verbose, \
                        args.basetry, args.samplecnt, args.basemode, args.model)

if acc_noaug > 0:
    gain = round((acc_aug - acc_noaug) / acc_noaug, 4)
else:
    gain = -1


summary = ['summary===>'] + ['{}:{}'.format(k, v) for k, v in vars(args).items() if not k.startswith('eda_')] + \
    ['acc_base:{} acc_aug:{} gain:{}'.format(acc_noaug, acc_aug, gain )]


record_log('logb', summary)
print('success', ' '.join(summary))