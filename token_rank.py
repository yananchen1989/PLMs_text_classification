import argparse,os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--future_steps", default=32, type=int)
parser.add_argument("--beams", default=128, type=int)
parser.add_argument("--alpha", default=0.5, type=float)
parser.add_argument("--gpu", default="0,1", type=str)
args = parser.parse_args()
print('args==>', args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf 
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
    
from transformers import top_k_top_p_filtering
from torch.nn import functional as F
import os,string,torch,math,time

from transformers import pipeline
from utils.load_data import * 
ds = load_data(dataset='uci', samplecnt= 128)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: x.strip(string.punctuation))

# gpt2
from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)
# no
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
# tc pp
#gpt2 = GPT2LMHeadModel.from_pretrained('ft_model_{}_{}'.format('t5', 'ep') )
gpt2.trainable = False
gpt2.config.pad_token_id = 50256

gen_nlp_gpt2  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=1, return_full_text=True)

device_i = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
gpt2.to(device_i)

from sklearn.metrics.pairwise import cosine_distances,cosine_similarity
from utils.transblock import * 

with tf.distribute.MirroredStrategy().scope():
    model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
model_cls.load_weights("./model_cls/model_full_uci.h5")          



################# gpt token generation ############  

def vs_mc(sent, label, candidate_ids, future_steps=32, test_beams=512):
    # ori
    ori_ids = tokenizer_gpt2.encode(sent, return_tensors="pt")
    tokens_len_ori = ori_ids.shape[1]
    result = gen_nlp_gpt2([sent], max_length=tokens_len_ori+future_steps, do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                            repetition_penalty=1.0, num_return_sequences=test_beams, clean_up_tokenization_spaces=True)
    
    sents_future = np.array([ ii['generated_text'].strip() for ii in result ])
    x = sents_future.reshape(-1,1)
    y = np.array([label] * sents_future.shape[0])
    eval_result = model_cls.evaluate(x, y, batch_size=64, verbose=0)  
    loss_ori =  eval_result[0]

    candidate_sents = []
    for idd in candidate_ids: 
        candidate_sent_ids = torch.cat([ori_ids, torch.tensor(idd).reshape(1,-1)], dim=-1) 
        candidate_sent = tokenizer_gpt2.decode(candidate_sent_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
        candidate_sents.append(candidate_sent)

    result_1 = gen_nlp_gpt2(candidate_sents, max_length=tokens_len_ori+1+future_steps, do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                            repetition_penalty=1.0, num_return_sequences=test_beams, clean_up_tokenization_spaces=True)

    # scores = []
    # for i in range(len(result_1)): #  32 * 256
    #     x = np.array([ii['generated_text'] for ii in result_1[i]])
    #     y = np.array([label] * x.shape[0])
    #     eval_result = model_cls.evaluate(x, y, batch_size=64, verbose=0) 
    #     scores.append(loss_ori-eval_result[0])
    
    assert len(result_1) * len(result_1[0]) == len(candidate_sents) * test_beams
    all_results = []
    for r in result_1:
        all_results.extend( [ii['generated_text'] for ii in r] )

    assert len(all_results) == len(candidate_sents) * test_beams

    x = np.array(all_results)
    #y = np.array([label] * x.shape[0])
    preds = model_cls.predict(x,  batch_size=args.batch_size, verbose=0) 

    scores = []
    for j in range(0, len(all_results), test_beams):
        preds_j = preds[j:j+test_beams]
        y_j = np.array([label] * preds_j.shape[0])
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_j, preds_j)
        loss_mean = loss.numpy().mean()
        score = loss_ori - loss_mean
        scores.append(score)

    return scores



def add_cls_score(sent, label, next_logits, next_values):
    torch.cuda.empty_cache()
    pred_ori = model_cls.predict([sent], steps=1)
    label_score = pred_ori[0][label]
    ori_ids = tokenizer_gpt2.encode(sent, return_tensors="pt")

    candidate_sents = []
    for idd in next_logits: 
        candidate_sent_ids = torch.cat([ori_ids, torch.tensor(idd).reshape(1,-1)], dim=-1) 
        candidate_sent = tokenizer_gpt2.decode(candidate_sent_ids[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
        candidate_sents.append(candidate_sent)


    pred_candidate = model_cls.predict(candidate_sents, batch_size=64)

    score_diff = pred_candidate[:,label] - label_score

    # for ii in range(score_diff.shape[0]):
    #     print(candidate_sents[ii], score_diff[ii])

    # next_values_modify = []    
    # for s, i, j in zip(candidate_sents, score_diff, next_values):
    #     if i < 0:
    #         jj = -math.inf
    #     elif i == 0:
    #         jj = j 
    #     elif i > 0:
    #         jj = j / i
    #     next_values_modify.append(jj)

    return score_diff


# from utils.encoders import *
# enc = encoder('cmlm-large')

# enc_dic = {}
# for l in ds.df_train['label'].unique():
#     contents_ = ds.df_train.loc[ds.df_train['label']==l]['content'].values
#     embeds = enc.infer(contents_)
#     enc_dic[l] = embeds


# embed_candidates = enc.infer(candidate_sents)
# result = {}
# for l, embeds in enc_dic.items():
#     scores = cosine_similarity(embed_candidates, embeds).mean()
#     break 
#     print(l, score) 
#     result[l] = score



while 1:
    row = ds.df_train.sample(1)
    sent = row['content'].tolist()[0]
    label = row['label'].tolist()[0]
    label_name = row['label_name'].tolist()[0]
    print("ori===>", sent, "<===", label_name)

    t0 = time.time()
    for step in range(64):
        input_ids = tokenizer_gpt2.encode(sent, return_tensors="pt").to(device_i)

        # get logits of last hidden state
        next_token_logits = gpt2(input_ids).logits[:, -1, :] / 1.0

        # filter
        filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_p=0.9) # , top_p=1
        
        #next_logits = [ii for ii in range(filtered_next_token_logits.cpu().detach().numpy()[0].shape[0]) if not math.isinf(ii)]
        #print("valid next logits:", len(next_logits)) 

        filtered_next_token_logits_arry = filtered_next_token_logits.cpu().detach().numpy()[0] 

        next_values = filtered_next_token_logits_arry[filtered_next_token_logits_arry!=(-math.inf)]
        next_logits = np.where(filtered_next_token_logits_arry!=(-math.inf))[0]

        print("valid after top_p:", next_logits.shape[0])

        next_values_modify = add_cls_score(sent, label, next_logits, next_values)

        for idd, v in zip(next_logits, next_values_modify):
            filtered_next_token_logits[0][idd] = float(v)

        probs = F.softmax(filtered_next_token_logits, dim=-1) # 50257

        probs_ = probs.detach().cpu().numpy()[0]

        candidate_ids = np.where(probs_>0)[0] 

        #print(probs_[probs_>0])

        # scores = vs_mc(sent, label, candidate_ids, args.future_steps, args.beams)

        # # modify probs
        # for idd, score in zip(candidate_ids, scores):
        #     probs[0][idd] =  max( (1-args.alpha) * probs[0][idd] + args.alpha * score, 0)

        if probs.cpu().detach().numpy().max()==0:
            probs += 1

        next_token = torch.multinomial(probs, num_samples=1)
        gen_ids = torch.cat([input_ids, next_token], dim=-1)
        sent = tokenizer_gpt2.decode(gen_ids.tolist()[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
        #print(candidate_ids, "sample==>", next_token)
        print("===>", sent.replace('\n', ''))
        # if step % 32 ==0:
        #     print("sent_tmp==>", sent.replace('\n',' '))
    t1 = time.time()

    print("sent_final==>", sent)
    print("time cost:", (t1-t0) / 3600)
    print('\n\n\n')