import argparse,os
parser = argparse.ArgumentParser()
parser.add_argument("--future_steps", default=32, type=int)
parser.add_argument("--beams", default=128, type=int)
parser.add_argument("--alpha", default=128, type=float)
parser.add_argument("--gpu", default="0,1", type=str)
args = parser.parse_args()
print('args==>', args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


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


from utils.transblock import * 
model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
model_cls.load_weights("./model_cls/model_uci.h5")          



################# gpt token generation ############  

def vs(sent, label, candidate_ids, future_steps=32, beams=512):
    # ori
    ori_ids = tokenizer_gpt2.encode(sent, return_tensors="pt")
    tokens_len_ori = ori_ids.shape[1]
    result = gen_nlp_gpt2([sent], max_length=tokens_len_ori+future_steps, do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                            repetition_penalty=1.0, num_return_sequences=beams, clean_up_tokenization_spaces=True)
    
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
                            repetition_penalty=1.0, num_return_sequences=beams, clean_up_tokenization_spaces=True)

    scores = []
    for i in range(len(result_1)): #  32 * 256
        x = np.array([ii['generated_text'] for ii in result_1[i]])
        y = np.array([label] * x.shape[0])
        eval_result = model_cls.evaluate(x, y, batch_size=64, verbose=0) 
        #print('\n' ,candidate_sents[i])
        #print("loss_diff:", loss_ori-eval_result[0] )
        scores.append(loss_ori-eval_result[0])
    return scores


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
        filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=16) # , top_p=1
        #next_logits = [ii for ii in filtered_next_token_logits.cpu().detach().numpy()[0] if not math.isinf(ii)]
        #print("valid next logits:", len(next_logits)) 

        probs = F.softmax(filtered_next_token_logits, dim=-1) # 50257

        probs_ = probs.detach().cpu().numpy()[0]

        candidate_ids = np.where(probs_>0)[0] 

        #print(probs_[probs_>0])

        scores = vs(sent, label, candidate_ids, args.future_steps, args.beams)

        # modify probs
        for idd, score in zip(candidate_ids, scores):
            probs[0][idd] =  max( (1-args.alpha) * probs[0][idd] + args.alpha * score, 0)

        # for _ in range(50):
        #     next_token = torch.multinomial(probs, num_samples=1)
        #     next_word = tokenizer_gpt2.convert_ids_to_tokens([next_token.detach().cpu().numpy()[0][0]], skip_special_tokens=True)
        #     print(next_word)
        if probs.cpu().detach().numpy().max()==0:
            probs += 1
        next_token = torch.multinomial(probs, num_samples=1)
        gen_ids = torch.cat([input_ids, next_token], dim=-1)
        sent = tokenizer_gpt2.decode(gen_ids.tolist()[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
        
        # if step % 32 ==0:
        #     print("sent_tmp==>", sent.replace('\n',' '))
    t1 = time.time()

    print("sent_final==>", sent)
    print("time cost:", (t1-t0) / 3600)
    print('\n\n\n')