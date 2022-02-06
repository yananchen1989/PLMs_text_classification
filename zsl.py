import pandas as pd
import time,argparse
import os,math,itertools
import numpy as np
import re,operator,joblib
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity 
import joblib,gensim,transformers
#assert gensim.__version__ == '4.1.2'

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str)
parser.add_argument("--fbs_gpt", default=256, type=int)
parser.add_argument("--fbs_para", default=32, type=int)
parser.add_argument("--acc_topn", default=1, type=int)
parser.add_argument("--expand", default='gpt_filter', type=str)
parser.add_argument("--softmax_score", default=0, type=int)

parser.add_argument("--backbone", default='nli', type=str, choices=['nli','tars','roberta','nspbert','simi'])
parser.add_argument("--seed_sample", default=8, type=int)
parser.add_argument("--gpu", default="", type=str)
parser.add_argument("--param", default='t5', type=str)

args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

transformers.logging.set_verbosity_error()
#from utils.flair_ners import *

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

from utils.load_data import * 
from utils.transblock import * 
from utils.encoders import *


ds = load_data(dataset=args.dsn, samplecnt= 8)
labels_candidates = ds.df_train['label_name'].unique().tolist()
print(labels_candidates)
if args.dsn in ['nyt','yahoo']:
    ds, proper_len = process_ds(ds, 256, True)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))

from transformers import pipeline
import torch
device0 = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

if args.backbone == 'nli':
    # nli model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    nli_model_name = 'vicgalle/xlm-roberta-large-xnli-anli' #"facebook/bart-large-mnli"
    model_nli = AutoModelForSequenceClassification.from_pretrained(nli_model_name, cache_dir='./cache', local_files_only=True)
    tokenizer_nli = AutoTokenizer.from_pretrained(nli_model_name, cache_dir='./cache', local_files_only=True)
    nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=len(gpus)-1)

elif args.backbone == 'roberta':
    from transformers import AutoTokenizer, AutoModelWithLMHead
    tokenizer = AutoTokenizer.from_pretrained("roberta-large",cache_dir="./cache",local_files_only=True) 
    model = AutoModelWithLMHead.from_pretrained("roberta-large",cache_dir="./cache",local_files_only=True)
    nlp_fill = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=len(gpus)-1, top_k = 2048 ) 
    id_token = {ix:token for token, ix in tokenizer.vocab.items()}

    stopwords = joblib.load("./utils/stopwords")
    stopwords = set(stopwords)

elif args.backbone == 'tars':
    import flair
    from flair.models import TARSClassifier
    from flair.data import Sentence
    flair.device = torch.device('cuda:{}'.format(args.gpu))
    # 1. Load our pre-trained TARS model for English
    tars = TARSClassifier.load("./resource/my-tars.pt")

elif args.backbone == 'simi':
    enc = encoder('cmlm-base', 'cpu')
    embed_label = enc.infer(labels_candidates)

#nsp model
from transformers import BertTokenizer, BertForNextSentencePrediction
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache', local_files_only=True)
bert_nsp = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', cache_dir='./cache', local_files_only=True)
bert_nsp.to(device0)


if args.param == 't5':
    from transformers import T5Tokenizer, AutoModelWithLMHead
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
    print(tokenizer_t5)
    t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)    
    gen_nlp_t5  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=len(gpus)-1)

elif args.param == 'bt':
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer_backward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", cache_dir="./cache", local_files_only=True)
    model_backward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en", cache_dir="./cache", local_files_only=True)
    tokenizer_forward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh", cache_dir="./cache", local_files_only=True)
    model_forward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh", cache_dir="./cache", local_files_only=True)
    nlp_backward = pipeline("translation", model=model_backward, tokenizer=tokenizer_backward, device=len(gpus)-1)
    nlp_forward = pipeline("translation", model=model_forward, tokenizer=tokenizer_forward, device=len(gpus)-1)
    print('bt model loaded')    


elif args.param == 'bart':
    from transformers import BartForConditionalGeneration, BartTokenizer
    model_bart = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase', cache_dir="./cache", local_files_only=True).to(device0)
    tokenizer_bart = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase', cache_dir="./cache", local_files_only=True)

elif args.param == 'peg':
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    tokenizer_peg = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase', cache_dir="./cache", local_files_only=True)
    model_peg = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase', cache_dir="./cache", local_files_only=True).to(device0)

elif args.param == 't5paws':
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer_t5paws = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", cache_dir="./cache", local_files_only=True)  
    model_t5paws = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws", cache_dir="./cache", local_files_only=True).to(device0)


def para_bt(content):
    content_ =  nlp_forward([content], truncation=True, \
                       do_sample=True, temperature=0.9, max_length=128, num_return_sequences=8)
    content_ = list(set([ii['translation_text'] for ii in content_]))
    content__ =  nlp_backward(content_, truncation=True, \
                        do_sample=True, max_length=128, temperature=0.9, num_return_sequences=8 )
    content__ = list(set([ii['translation_text'] for ii in content__]))
    return random.sample(content__, min(args.fbs_para, len(content__)))


def para_bart(content):
    result = set()
    while 1:
        batch = tokenizer_bart(content, return_tensors='pt', truncation=True, padding='longest',max_length=128)
        generated_ids = model_bart.generate(batch['input_ids'].to(device0), max_length=128,
                                do_sample=True, num_return_sequences=8)
        generated_sentence = tokenizer_bart.batch_decode(generated_ids, skip_special_tokens=True)

        result.update([sent for sent in generated_sentence if sent != content])
        if len(result) >= args.fbs_para:
            break 
    return random.sample(list(result), args.fbs_para)


def para_peg(content):
    result = set()
    while 1:
        batch = tokenizer_peg([content], truncation=True,padding='longest', max_length=128, return_tensors="pt").to(device0)
        translated = model_peg.generate(**batch, do_sample=True, max_length=128, num_beams=8, num_return_sequences=8, temperature=1.5)
        tgt_text = tokenizer_peg.batch_decode(translated, skip_special_tokens=True)
        result.update([sent for sent in list(set(tgt_text)) if sent != content])
        if len(result) >= args.fbs_para:
            break 
    return random.sample(list(result), args.fbs_para)

def para_t5paws(content):
    text =  "paraphrase: " + content + " {}".format(tokenizer_t5paws.eos_token)
    encoding = tokenizer_t5paws.encode_plus(text, pad_to_max_length=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device0), encoding["attention_mask"].to(device0)

    outputs = model_t5paws.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=128,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=args.fbs_para)

    results = []
    for output in outputs:
        line = tokenizer_t5paws.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if line != content:
            results.append(line)
    return results



def para_t5(content):
    dsn_maxlen = {'uci':64, 'agt':64, 'ag':128, 'yahoo':128, 'nyt':128, 'amazon2':128, 'yelp2':128, 'imdb':128}
    result_gpt = gen_nlp_t5([content], max_length=dsn_maxlen[args.dsn], \
                                        do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                        repetition_penalty=1.2, num_return_sequences= args.fbs_para,\
                                        clean_up_tokenization_spaces=True)
    assert len(result_gpt) == args.fbs_para
    ori_gen_contents = [ii['generated_text'] for ii in result_gpt if ii['generated_text']] 
    return ori_gen_contents

para_func = {'t5':para_t5, 'bt':para_bt, 'bart':para_bart, 'peg':para_peg, 't5paws':para_t5paws}

def para_ranking(content_ori):
    contents_para = para_func[args.param](content_ori)

    ls = {l:0 for l in labels_candidates}

    for sent in contents_para:
        dfi = ZSL_FUNC[args.backbone](sent)
        for ix, row in dfi.iterrows():
            ls[row['label']] += row['score_noexpand']

    #ls_sort = sorted(ls.items(), key=operator.itemgetter(1), reverse=True)
    df_t5 = pd.DataFrame(ls.items(), columns=['label','score_t5'])
    df_t5['score_t5'] = df_t5['score_t5'] / len(contents_para)
    return df_t5


def continuation_ranking(content_ori):
    infos = []
    for l in labels_candidates:
        scores_tmp = []
        # gpt generation
        contents_syn = df_contents_arxiv.loc[df_contents_arxiv['label_name']==l].sample(args.fbs_gpt)['content'].tolist()
        pairs = list(itertools.product([content_ori], contents_syn ))
        #for ii in range(0, args.fbs, 8):
            #pairs = [[row['content'], sent] for sent in contents_syn[ii:ii+8]]
        for j in range(0, len(pairs), 8):
            score_nsp = nsp_infer_pairs(pairs[j:j+8], bert_nsp, bert_tokenizer, device0)[:,0]
            scores_tmp.extend(list(score_nsp)) 

        score_nsp_reduce = np.array(scores_tmp).mean()
        infos.append((l, score_nsp_reduce))

    df_nsp = pd.DataFrame(infos, columns=['label','score_nsp'])
    return df_nsp



'''
from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)
gpt2.trainable = False
gpt2.config.pad_token_id=50256
gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=0, return_full_text=False)

def gen_gpt_expansions():
    infos = []
    while True:
        #prompts = ["topic {} source strait stimes title".format(label) for label in labels_candidates]
        
        if args.expand == 'seeds':
            prompts = ["This is {} News: ".format(" ".join(random.sample(label_expands_auto[label], args.seed_sample))) for label in labels_candidates] # gpt-seed words
        elif args.expand == 'gpt_vanilla':
            prompts = ["This is {} News: ".format(label) for label in labels_candidates] # gpt
        #prompts = ["Links In {} : ".format(label) for label in labels_candidates] # ctrl
        result_gpt = gen_nlp(prompts, max_length=128, \
                                                    do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                                    repetition_penalty=1.2, num_return_sequences= 16,\
                                                    clean_up_tokenization_spaces=True)

        for label, rg in zip(labels_candidates, result_gpt):
            contents = [ ii['generated_text'] for ii in rg if len(ii['generated_text'])>=20 ] 
            for sent in contents:
                infos.append((remove_str(sent) , label ))

            result_nli = nli_nlp(contents, labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
            for ii in  result_nli:
                df_tmp = pd.DataFrame(ii)
                df_tmp_sel = df_tmp.loc[df_tmp['scores']>=0.9]
                if df_tmp_sel.shape[0] == 0:
                    continue
                if label in df_tmp_sel['labels'].tolist():
                    infos.append((remove_str(ii['sequence']), label))
        if len(infos) > 0 and len(infos) % 100:
            df = pd.DataFrame(infos, columns = ['content','label_name'])
        if df['label_name'].value_counts().min() >= args.fbs_para * 4  :
            break 

    #df.to_csv("df_gen_{}_{}_{}.csv".format(args.dsn, args.expand, args.seed_sample), index=False)
    return df

'''

def zsl_roberta(content):

    template1 = "{}. This News is about {}".format(content, nlp_fill.tokenizer.mask_token)
    template2 = "{} News: {}.".format(nlp_fill.tokenizer.mask_token, content)
    template3 = "[Category: {} ] {}.".format(nlp_fill.tokenizer.mask_token, content)

    ls = {l:0 for l in labels_candidates}

    filled_results = nlp_fill([template1, template2, template3])
    
    for filled_result in filled_results:

        for r in filled_result :
            token = r['token_str'].lower().strip()

            if token in stopwords or token in string.punctuation or token.isdigit() :
                continue

            for l in ls.keys():
                if token in l.lower():
                    ls[l] += r['score']  

    df_noexpand = pd.DataFrame(ls.items(), columns=['label','score_noexpand'])
    return df_noexpand


def zsl_nli(content):
    nli_result = nli_nlp(content,  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
    nli_result.pop('sequence')
    df_noexpand = pd.DataFrame(nli_result)
    df_noexpand = df_noexpand.rename(columns={'labels': 'label', 'scores':'score_noexpand'})
    return df_noexpand


def zsl_tars(content):
    sentence = Sentence(content)
    tars.predict_zero_shot(sentence, labels_candidates)
    result = sentence.to_dict()
    print(l, '===>', result['labels'])

def zsl_simi(content):
    embed_contents = enc.infer([content], batch_size=1)
    embeds_score = cosine_similarity(embed_contents, embed_label)
    df_noexpand = pd.DataFrame(zip(labels_candidates, list(embeds_score[0])), columns=['label', 'score_noexpand'])
    df_noexpand['score_noexpand'] = df_noexpand['score_noexpand'] + 1
    return df_noexpand

def zsl_nspbert(content):
    prompt_1 = [["This document is about {}".format(l), content] for l in labels_candidates]
    prompt_2 = [["It is {} News".format(l), content] for l in labels_candidates]
    score_nsp_1 = nsp_infer_pairs(prompt_1, bert_nsp, bert_tokenizer, device0)[:,0]    
    score_nsp_2 = nsp_infer_pairs(prompt_2, bert_nsp, bert_tokenizer, device0)[:,0]    
    
    score_nsp = (score_nsp_1  + score_nsp_2 ) / 2
    df_noexpand = pd.DataFrame(zip(labels_candidates, list(score_nsp)), columns=['label', 'score_noexpand'])
    return df_noexpand


ZSL_FUNC = {'nli':zsl_nli, 'roberta':zsl_roberta, 'tars':zsl_tars, 'simi':zsl_simi, 'nspbert':zsl_nspbert}


if args.expand == 'gpt_filter':
    df_contents_arxiv = pd.read_csv("df_gen_{}.csv".format(args.dsn))
elif args.expand == 'gpt_nofilter':
    df_contents_arxiv_ = pd.read_csv("df_gen_{}_nofil.csv".format(args.dsn))
    df_contents_arxiv = df_contents_arxiv_.loc[(~df_contents_arxiv_['content'].isnull()) & (df_contents_arxiv_['content']!='')]
elif args.expand == 'pplm':
    df_contents_arxiv = pd.read_csv("df_gen_pplm_{}.csv".format(args.dsn))
elif args.expand == 'seeds':
    df_contents_arxiv = gen_gpt_expansions()



acc = {}

for ix, row in ds.df_test.sample(frac=1).reset_index().iterrows():
    torch.cuda.empty_cache() 

    try:
        df_noexpand = ZSL_FUNC[args.backbone](row['content'])
        df_t5 = para_ranking(row['content'])

        df_nsp = continuation_ranking(row['content'])

        df_fuse_ = pd.merge(df_noexpand, df_t5, on=['label'], how='inner')
        df_fuse  = pd.merge(df_fuse_, df_nsp, on=['label'], how='inner')


        if args.softmax_score:
            for col in df_fuse.columns:
                if col.startswith('score_'):
                    softmax_v = np.exp(df_fuse[col].values) / np.sum(np.exp(df_fuse[col].values))
                    df_fuse[col] = softmax_v

        df_fuse['score_w_nsp'] =  df_fuse['score_noexpand'].map(lambda x: math.log(x)) \
                                + df_fuse['score_nsp'].map(lambda x: math.log(x)) 

        df_fuse['score_w_t5'] =  df_fuse['score_noexpand'].map(lambda x: math.log(x)) \
                                + df_fuse['score_t5'].map(lambda x: math.log(x)) 

        df_fuse['score_w_t5_nsp'] =  df_fuse['score_noexpand'].map(lambda x: math.log(x)) \
                                + df_fuse['score_t5'].map(lambda x: math.log(x))    \
                                + df_fuse['score_nsp'].map(lambda x: math.log(x))   
    except:
        print("error==>")
        print(df_noexpand)
        print(df_t5)
        print(df_nsp)
        continue  

    for col in ['score_noexpand','score_w_t5', 'score_w_nsp', 'score_w_t5_nsp']:
        if col not in acc.keys():
            acc[col] = []

        pred_label = df_fuse.sort_values(by=[col], ascending=False).head(args.acc_topn)['label'].tolist()
        if row['label_name'] in pred_label:
            acc[col].append(1)
        else:
            acc[col].append(0)

    if (ix % 64 == 0 and ix > 0) or (ix == ds.df_test.shape[0]-1):
        print(ix)
        for col in ['score_noexpand','score_w_t5', 'score_w_nsp', 'score_w_t5_nsp']:
            print(col, round(np.array(acc[col]).mean(), 4))
        print()

print("final_summary==>", ' '.join(['{}:{}'.format(k, v) for k, v in vars(args).items()]))