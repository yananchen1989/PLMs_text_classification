import os, argparse,random
parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="nyt", type=str)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--ft_epochs", default=1, type=int)
parser.add_argument("--block_size", default=256, type=int)
parser.add_argument("--onlyft", default=0, type=int)

args = parser.parse_args()
#os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
from utils.transblock import * 
from utils.load_data import * 
from transformers import pipeline

from collections import Counter
import numpy as np

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel#TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
print('gpt2 tokenizer:', tokenizer_gpt2.unk_token, tokenizer_gpt2.bos_token, tokenizer_gpt2.eos_token)
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
gpt2.trainable = False
gpt2.config.pad_token_id=50256

gen_nlp_ori  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=args.gpu, return_full_text=False)

ds = load_data(dataset=args.dsn, samplecnt= 32)
ds, proper_len = process_ds(ds, 256, True)


seed = random.sample(list(range(10000)), 1)[0]
# train_file = './fintune_csvs/{}_train_finetune_32_{}.csv'.format(args.dsn, seed)
# validation_file = './fintune_csvs/{}_test_finetune_32_{}.csv'.format(args.dsn, seed)

# ds.df_train['text'] = ds.df_train['content'] + tokenizer_gpt2.eos_token
# ds.df_test['text'] = ds.df_test['content'] + tokenizer_gpt2.eos_token

# ds.df_train[['text']].sample(frac=1).to_csv(train_file, index=False)
# ds.df_test[['text']].sample(frac=1).to_csv(validation_file, index=False)

train_file = './fintune_csvs/{}_train_finetune_32_{}.txt'.format(args.dsn, seed)
validation_file = './fintune_csvs/{}_test_finetune_32_{}.txt'.format(args.dsn, seed)


with open (train_file, 'w') as f:
    f.write(" {} ".format(tokenizer_gpt2.eos_token).join(ds.df_train['content'].tolist()))

with open (validation_file, 'w') as f:
    f.write(" {} ".format(tokenizer_gpt2.eos_token).join(ds.df_test['content'].tolist()))




dsn_maxlen = {'uci':64, 'ag':256, 'nyt':512}


def distinct(seqs):
    """ Calculate intra/inter distinct 1/2. """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


def sents2tokensL(contents):
    list_of_references = []
    for sent in contents:
        tokens = []
        for ii in sent.strip().replace('\n',' ').split(' '):
            if ii:
                tokens.append(ii.lower())
        list_of_references.append(tokens)
    return list_of_references

# from fast_bleu import BLEU, SelfBLEU
# weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}



def cal_selfBleu(gen_contents):
    list_of_references = sents2tokensL(gen_contents)
    self_bleu = SelfBLEU(list_of_references, weights)
    slefbleu_result = self_bleu.get_score()
    print('bigram:', np.array(slefbleu_result['bigram']).mean(), \
          'trigram:', np.array(slefbleu_result['trigram']).mean() )


def cal_Bleu(gen_contents, ref_contents):
    list_of_references = sents2tokensL(ref_contents)
    hypotheses = sents2tokensL(gen_contents)
    bleu = BLEU(list_of_references, weights)
    bleu_result = bleu.get_score(hypotheses)
    print('bigram:', np.array(bleu_result['bigram']).mean(), \
          'trigram:', np.array(bleu_result['trigram']).mean() )


# ref score
df_test_sample = ds.df_test.sample(2048)
seqs_ref = sents2tokensL(df_test_sample['content'].tolist())
intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(seqs_ref)
print('ref: distinct ngrams', intra_dist1, intra_dist2)




for args.ft_epochs in [1,3,7,12]:
    model_output_path = "./finetune_gpt2/{}_32_{}".format(args.dsn, seed) 
    os.system(
    "CUDA_VISIBLE_DEVICES=1 python -u ./run_clm_no_trainer.py \
            --num_train_epochs {} \
            --train_file {} \
            --validation_file {} \
            --model_name_or_path gpt2 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --output_dir {} \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --block_size {}".format(args.ft_epochs, train_file, validation_file, model_output_path, args.block_size) ) 


    gpt2_ft = GPT2LMHeadModel.from_pretrained(model_output_path)
    gpt2_ft.trainable = False
    gpt2_ft.config.pad_token_id=50256
    gen_nlp_ft  = pipeline("text-generation", model=gpt2_ft, tokenizer=tokenizer_gpt2, device=args.gpu, return_full_text=False)



    gen_contents_ori = []
    gen_contents_ft = []


    for i in range(0, df_test_sample.shape[0], 32):
        print('==>', i)
        contents_trunk = df_test_sample['content'].tolist()[i:i+32]
        results_trunk_ft = gen_nlp_ft (contents_trunk, max_length=dsn_maxlen[args.dsn], do_sample=True, top_p=0.9, top_k=0, \
                            repetition_penalty=1.0, num_return_sequences=1, clean_up_tokenization_spaces=True)
        gen_contents_ft.extend([ii[0]['generated_text'] for ii in results_trunk_ft])

        
        if args.ft_epochs == 1:
            results_trunk_ori = gen_nlp_ori(contents_trunk, max_length=dsn_maxlen[args.dsn], do_sample=True, top_p=0.9, top_k=0, \
                               repetition_penalty=1.0, num_return_sequences=1, clean_up_tokenization_spaces=True)
            gen_contents_ori.extend([ii[0]['generated_text'] for ii in results_trunk_ori]) 

    
    
    seqs_ft = sents2tokensL(gen_contents_ft)
    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(seqs_ft)
    print('ft: distinct ngrams', intra_dist1, intra_dist2)

    if args.ft_epochs == 1:
        seqs_ori = sents2tokensL(gen_contents_ori)
        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(seqs_ori)
        print('ori: distinct ngrams', intra_dist1, intra_dist2)




# print('ori==>')
# cal_Bleu(gen_contents_ori, contents_trunk)
# print('ft==>')
# cal_Bleu(gen_contents_ft, contents_trunk)





'''
# 1-gram individual BLEU
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu

def get_bleu(seqs_ref, seqs_):
    infos = []
    for candidate, ref  in zip(seqs_, seqs_ref):
        score_uni = sentence_bleu([ref], candidate, weights=(1, 0, 0, 0))
        score_bi = sentence_bleu([ref], candidate, weights=(0, 1, 0, 0))
        score_tri = sentence_bleu([ref], candidate, weights=(0, 0, 1, 0))
        infos.append((score_uni, score_bi, score_tri))

    df_bleu = pd.DataFrame(infos, columns=['score_uni','score_bi','score_tri'])
    print(df_bleu['score_uni'].mean(), df_bleu['score_bi'].mean(), df_bleu['score_tri'].mean())


get_bleu(seqs_ref, seqs_ori)

get_bleu(seqs_ref, seqs_ft)
'''




# print('ori==>')
# cal_selfBleu(gen_contents_ori)

# print('ft==>')
# cal_selfBleu(gen_contents_ft)



# ori: 0.41300412594489094 0.17545364068562297
# ft: 0.39453614839268614 0.16453152919661795






# cal_Bleu(gen_contents_ori, df_test_sample['content'].tolist())
# cal_Bleu(gen_contents_ft, df_test_sample['content'].tolist())







'''
A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0.
'''



