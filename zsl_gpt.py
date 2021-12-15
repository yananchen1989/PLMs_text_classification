import operator
import pandas as pd
import time,argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str, choices=['uci','ag','agt','nyt','yelp2','amazon2','stsa'])
parser.add_argument("--fbs_gen", default=32, type=int)
parser.add_argument("--maxlen", default=64, type=int)
parser.add_argument("--genm", default='gpt2', type=str)
parser.add_argument("--gpu", default="0,1", type=str)
args = parser.parse_args()

dsn_maxlen = {'uci':64, 'agt':64, 'ag':128, 'nyt':128, 'amazon2':128, 'yelp2':128}

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#from utils.flair_ners import *
import tensorflow as tf
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
if args.dsn == 'nyt':
    samplecnt = 256 
else:
    samplecnt = 2048

ds = load_data(dataset=args.dsn, samplecnt= samplecnt)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))
ds, proper_len = process_ds(ds, 128)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))


labels_candidates = ds.df_train['label_name'].unique().tolist()
print("labels_candidates==>", labels_candidates)

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_nli = AutoModelForSequenceClassification.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
tokenizer_nli = AutoTokenizer.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=1)


from transformers import pipeline

if args.genm == 'gpt2':
    from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
    #tokenizer_gpt2.padding_side = "left" 
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
    tokenizer_gpt2.sep_token = '<|sep|>'
    #tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
    print(tokenizer_gpt2)
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)

    gpt2.trainable = False
    gpt2.config.pad_token_id=50256
    gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=0, return_full_text=False)

elif args.genm == 't5':
    from transformers import T5Tokenizer, AutoModelWithLMHead
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
    print(tokenizer_t5)
    t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)    
    gen_nlp  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=0)



# gpt expand
accs_noexpand = []
accs_expand = []
for ix, row in ds.df_train.reset_index().iterrows():
    content = row['content']

    nli_result = nli_nlp([content],  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")

    pred_label =  nli_result['labels'][0]
    if pred_label == row['label_name']:
        accs_noexpand.append(1)
    else:
        accs_noexpand.append(0)


    result_gpt = gen_nlp([remove_str(content)], max_length=dsn_maxlen[args.dsn], \
                                        do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                        repetition_penalty=1.2, num_return_sequences= args.fbs_gen,\
                                        clean_up_tokenization_spaces=True)
    
    ori_gen_contents = [ii['generated_text'] for ii in result_gpt if ii['generated_text']] + [remove_str(content)]
    nli_result = nli_nlp(ori_gen_contents,  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
    ls = {l:0 for l in labels_candidates}
    for r in nli_result:
        for l,s in zip(r['labels'], r['scores']):
            ls[l] += s

    ls_sort = sorted(ls.items(), key=operator.itemgetter(1), reverse=True)

    if ls_sort[0][0] == row['label_name']:
        accs_expand.append(1)
    else:
        accs_expand.append(0)

    if ix % 256 == 0 and ix > 0:
        print(ix, sum(accs_noexpand) / len(accs_noexpand), sum(accs_expand)/len(accs_expand))

print("summary==>", ' '.join(['{}:{}'.format(k, v) for k, v in vars(args).items()]) ,
             sum(accs_noexpand) / len(accs_noexpand), sum(accs_expand)/len(accs_expand) )












