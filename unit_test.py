sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

sent = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

sent = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''
sent = "Federal jury orders tech giant Samsung to pay"

sent = 'FDA gives green light to migraine prevention tool'

sent = "Obamacare offers health insurance, not health care"

sent = "Why BlackBerry ( BBRY ) Stock Is Up Today"

sent = "Doctor warns Arizonans about colorectal cancer"


sent = "Facebook acquires video ad company LiveRail for between US $ 500m"

sent = "Autism wave keeps growing"

sent = "Virus to cause spike in pork prices"



# gpt neo
# from transformers import GPT2Tokenizer, GPTNeoForCausalLM
# tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir="./cache")
# gpt2 = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir="./cache")














import transformers
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

for dsn in ['agp','uci','yahoo']:
    ds = load_data(dataset=dsn, samplecnt= 32)
    labels_candidates = ds.df_train['label_name'].unique().tolist()
    print(labels_candidates)
    infos = []
    while True:
        #prompts = ["topic {} source strait stimes title".format(label) for label in labels_candidates]
        prompts = ["This is {} News: ".format(label) for label in labels_candidates]
        result_gpt = gen_nlp(prompts, max_length=64, \
                                                    do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                                    repetition_penalty=1.2, num_return_sequences= 64,\
                                                    clean_up_tokenization_spaces=True)

        for label, rg in zip(labels_candidates, result_gpt):
            contents = [ ii['generated_text'] for ii in rg if len(ii['generated_text'])>=20 ] 

            result_nlp = nli_nlp(contents, labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
            for ii in  result_nlp:
                df_tmp = pd.DataFrame(ii)
                df_tmp_sel = df_tmp.loc[df_tmp['scores']>=0.95]
                if df_tmp_sel.shape[0] == 0:
                    continue
                if label in df_tmp_sel['labels'].tolist():
                    infos.append((remove_str(ii['sequence']), label))

        if len(infos) > 0 and len(infos) % 1000:
            df = pd.DataFrame(infos, columns = ['content','label_name'])
            print(df['label_name'].value_counts())
            df.to_csv("df_gen_{}.csv".format(dsn), index=False)
            if df['label_name'].value_counts().min() >= 512:
                break 









from sklearn.metrics.pairwise import cosine_distances,cosine_similarity
from utils.load_data import * 

ds = load_data(dataset='yahoo', samplecnt= 2048)
df_gen = pd.read_csv("df_gen_yahoo.csv")

labels_candidates = ds.df_train['label_name'].unique().tolist()




from sklearn.metrics import accuracy_score

infos = []
for ix, row in ds.df_train.reset_index().iterrows():

    pairs0 = [[row['content'], "this text is about {}".format(l)] for l in labels_candidates]
    pairs1 = [["this text is about {}".format(l) ,row['content']] for l in labels_candidates]

    nsp_logits0 = nsp_infer_pairs(pairs0, bert_nsp, bert_tokenizer)
    preds0 = nsp_logits0[:,0]

    nsp_logits1 = nsp_infer_pairs(pairs1, bert_nsp, bert_tokenizer)
    preds1 = nsp_logits1[:,0]

    preds = preds0 + preds1
    pred_ix = preds.argmax()
    pred_label_name = labels_candidates[pred_ix]

    pred_label_name0 = labels_candidates[preds0.argmax()]
    pred_label_name1 = labels_candidates[preds1.argmax()]

    #scores = [nsp_infer(row['content'], "this text is about {}".format(l), bert_nsp, bert_tokenizer ) for l in labels_candidates]
    #pred_label_name_ = labels_candidates[np.array(scores).argmax()]

    infos.append((pred_label_name, pred_label_name0, pred_label_name1, row['label_name']))


    if len(infos) % 100 ==0:

        df = pd.DataFrame(infos, columns=['p','p0','p1','p_','l']) 

        acc_p = accuracy_score(df['l'].values, df['p'].values)
        acc_p0 = accuracy_score(df['l'].values, df['p0'].values)
        acc_p1 = accuracy_score(df['l'].values, df['p1'].values)
        acc_p_ = accuracy_score(df['l'].values, df['p_'].values)
        print(acc_p, acc_p0, acc_p1, '===', acc_p_)


0.47846153846153844 0.46615384615384614 0.49 === 0.47846153846153844


'''
this is a science news:
The world's leading research institution is on a mission to create safe, 
renewable energy for all of us without compromising our health. 
Since 1981, the United Nations has implemented its Goal of Emissions Neutrality and is


'''

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs, labels=inputs["input_ids"])
# loss = outputs.loss
# logits = outputs.logits

import tensorflow as tf 
gpus = tf.config.list_physical_devices('GPU')
print('======>',gpus,'<=======')

############## whole token generation ############  

from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
try:
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
except:
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=False)

#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)
try:
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
except:
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=False)
#gpt2 = GPT2LMHeadModel.from_pretrained('ft_model_ft_ep')
from transformers import pipeline
gpt2.trainable = False
gpt2.config.pad_token_id = 50256
gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=-1, return_full_text=False)


sent = '''
Wizards welcome Bobcats to NBA Basketball returned to Charlotte as the Bobcats opened their first season Thursday night with a 103 - 96 loss to the Wizards. Replacing the Hornets after they moved to New Orleans
'''
result_gpt = gen_nlp([sent], max_length=128, \
                                            do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                            repetition_penalty=1.2, num_return_sequences= 36,\
                                            clean_up_tokenization_spaces=True)





from transformers import GPT2Tokenizer, TFGPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache_tfgpt")
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)
# no
gpt2 = TFGPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache_tfgpt")




input_ids = tokenizer_gpt2.encode(sent, return_tensors="tf")
# get logits of last hidden state
next_token_logits = gpt2(input_ids).logits[:, -1, :] / 1.0










# x_ori = np.array([sent]).reshape(-1,1)
# y_ori = np.array([label] * 1)
# eval_result_ori = model.evaluate(x_ori, y_ori, batch_size=1)

# t5
import glob
from transformers import pipeline
from transformers import T5Tokenizer, AutoModelWithLMHead
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
print(tokenizer_t5)
# no ft
t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)

# ft:tc pp
# ft_model_path = 'ft_model_{}_{}'.format('t5', 'tc')
# checkpoint_files = glob.glob(ft_model_path+"/checkpoint_loss_*")
# list.sort(checkpoint_files)
# t5 = AutoModelWithLMHead.from_pretrained(checkpoint_files[0])  

########
gen_nlp_t5  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=0)

prompts = ds.df_train['content'].map(lambda x: '{} {}'.format(x, tokenizer_t5.eos_token)).tolist()
labels =  ds.df_train['label'].tolist()

contents_trunk_ = gen_nlp_t5([prompts[11]], max_length=256, do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                          repetition_penalty=1.2, num_return_sequences=1, clean_up_tokenization_spaces=True)


################ t5 token generation ###################
# create ids of encoded input vectors

temperature=1.2

sent = 'FDA gives green light to migraine prevention tool'
input_ids = tokenizer_t5(sent, return_tensors="pt").input_ids

# create BOS token
decoder_input_ids = tokenizer_t5(tokenizer_t5.pad_token, add_special_tokens=False, return_tensors="pt").input_ids

assert decoder_input_ids[0, 0].item() == t5.config.decoder_start_token_id, "`decoder_input_ids` should correspond to `model.config.decoder_start_token_id`"

# pass input_ids to encoder and to decoder and pass BOS token to decoder to retrieve first logit
outputs = t5(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)

# get encoded sequence
encoded_sequence = (outputs.encoder_last_hidden_state,)
# get logits
lm_logits = outputs.logits

# sample last token with highest prob
#next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
next_token_logits = top_k_top_p_filtering(lm_logits[:, -1, :]/temperature, top_k=0, top_p=0.9)
probs = F.softmax(next_token_logits, dim=-1)
next_decoder_input_ids = torch.multinomial(probs, num_samples=1)

# concat
decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)


for _ in range(64):
    # reuse encoded_inputs and pass BOS + "Ich" to decoder to second logit
    lm_logits = t5(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True).logits

    # sample last token with highest prob again
    #next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
    next_token_logits = top_k_top_p_filtering(lm_logits[:, -1, :]/temperature, top_k=0, top_p=0.9)
    probs = F.softmax(next_token_logits, dim=-1)
    next_decoder_input_ids = torch.multinomial(probs, num_samples=1)

    # concat again
    decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)


print(tokenizer_t5.decode(decoder_input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))








