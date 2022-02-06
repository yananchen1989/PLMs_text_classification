sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

sent = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''
sent = "Federal jury orders tech giant Samsung to pay"

sent = 'FDA gives green light to migraine prevention tool'

sent = "Obamacare offers health insurance, not health care"

sent = "Why BlackBerry ( BBRY ) Stock Is Up Today"

sent = "Doctor warns Arizonans about colorectal cancer"


content = "Facebook acquires video ad company LiveRail for between US $ 500m"

sent = "Autism wave keeps growing"

sent = "Virus to cause spike in pork prices"

content = "Grand Budapest Hotel'not grand, but still stylish"


tokenizer("Why BlackBerry ( BBRY ) Stock Is Up Today", return_special_tokens_mask=True)

tokenizer_gpt2("Why BlackBerry ( BBRY ) Stock Is Up Today \n<|endoftext|>", return_special_tokens_mask=True)

tokenizer_t5("Why BlackBerry ( BBRY ) Stock Is Up Today"+tokenizer_t5.eos_token, max_length=100,\
         padding=True, truncation=True, return_special_tokens_mask=True)


tokenizer_t5.convert_ids_to_tokens([2])
tokenizer_gpt2.convert_ids_to_tokens([6288])
tokenizer_t5.convert_tokens_to_ids(['up'])


# gpt neo
# from transformers import GPT2Tokenizer, GPTNeoForCausalLM
# tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir="./cache")
# gpt2 = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir="./cache")









import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from transformers import pipeline
from transformers import T5Tokenizer, AutoModelWithLMHead
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir='./cache', local_files_only=True)
print(tokenizer_t5)
t5 = AutoModelWithLMHead.from_pretrained("./finetunes/t5_natcat/epoch_1")    

t5_noft = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir='./cache', local_files_only=True)    


from transformers import BartTokenizer, AutoModelWithLMHead
tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-base", cache_dir='./cache', local_files_only=True)
print(tokenizer_bart)
bart = AutoModelWithLMHead.from_pretrained("./finetunes/bart_natcat_label2content/epoch_0")    

bart_noft = AutoModelWithLMHead.from_pretrained("facebook/bart-base", cache_dir='./cache', local_files_only=True)    


gen_nlp_t5  = pipeline("text2text-generation", model=t5_noft, tokenizer=tokenizer_t5, device=0)
gen_nlp_bart  = pipeline("text2text-generation", model=bart_noft, tokenizer=tokenizer_bart, device=0)



sent = "Why BlackBerry ( BBRY ) Stock Is Up Today"



sent = "Technology BlackBerry Stock "

sent = "Health HIV vaccine"


result = gen_nlp_t5([sent], max_length=128, \
                                        do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                        repetition_penalty=1.2, num_return_sequences= 8,\
                                        clean_up_tokenization_spaces=True)

result = gen_nlp_bart([sent.lower()], max_length=128, \
                                        do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                        repetition_penalty=1.2, num_return_sequences= 8,\
                                        clean_up_tokenization_spaces=True)

for ii in result:
    print(ii['generated_text'], '\n')





ds.df_train.loc[ds.df_train['label_name']=='Sports'].sample(1)['content'].tolist()[0]



















import os 
os.environ['CUDA_VISIBLE_DEVICES'] = ""


from utils.load_data import * 
ds = load_data(dataset='ag', samplecnt= 128)


from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
from transformers import pipeline
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)
gpt2.trainable = False
gpt2.config.pad_token_id=50256
gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=-1, return_full_text=False)

labels_candidates = ds.df_train['label_name'].unique().tolist()


infos = []
while True:
    #prompts = ["topic {} source strait stimes title".format(label) for label in labels_candidates]
    
    prompts = ["This is {} News: ".format(label) for label in labels_candidates] # gpt
    #prompts = ["Links In {} : ".format(label) for label in labels_candidates] # ctrl
    result_gpt = gen_nlp(prompts, max_length=128, \
                                                do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                                repetition_penalty=1.2, num_return_sequences= 64,\
                                                clean_up_tokenization_spaces=True)

    for label, rg in zip(labels_candidates, result_gpt):
        contents = [ ii['generated_text'] for ii in rg if len(ii['generated_text'])>=20 ] 
        for sent in contents:
            infos.append((remove_str(sent) , label ))


    if len(infos) > 0 and len(infos) % 128 == 0:
        df = pd.DataFrame(infos, columns = ['content','label_name'])
        print(len(infos))
        df.to_csv("df_gen_ag_nofil.csv", index=False)

    if df['label_name'].value_counts().min() >= 1024  :
        break 



























input_ids = tokenizer_gpt2.encode(sent, return_tensors="tf")
# get logits of last hidden state
next_token_logits = gpt2(input_ids).logits[:, -1, :] / 1.0








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




#############  计算指标

import glob
import pandas as pd 


files = glob.glob("./log_lambda/uci.128.64.*.log")

infos = []
for file in files:
    with open(file,'r') as f: 
        for line in f:
            if 'success summary===>' in line:
                #print(line)
                for ii in tokens:
                    if ii.startswith("samplecnt:"):
                        samplecnt = int(ii.split(":")[-1])
                    if ii.startswith("candidates:"):
                        candidates = int(ii.split(":")[-1])
                tokens = line.strip().split('summary==')[-1].split(' ')
                fmark = tokens[-3].split(':')[-1]
                acc_base = float(tokens[-2].split(':')[-1])
                acc_aug = float(tokens[-1].split(':')[-1])
                infos.append((fmark, acc_base, acc_aug))

df = pd.DataFrame(infos, columns=['fmark','acc_base','acc_aug'])



for fmark in df['fmark'].unique():
    dfi = df.loc[df['fmark']==fmark]
    print( fmark, dfi.shape[0], dfi['acc_base'].mean(), dfi['acc_aug'].mean()) 






import torch
device0 = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

from transformers import CTRLTokenizer, CTRLLMHeadModel, pipeline
tokenizer_ctrl = CTRLTokenizer.from_pretrained('ctrl', cache_dir='./cache')
model_ctrl = CTRLLMHeadModel.from_pretrained('ctrl', cache_dir='./cache')
print(tokenizer_ctrl)
control_codes = tokenizer_ctrl.control_codes.keys()
gen_nlp_ctrl  = pipeline("text-generation", model=model_ctrl, tokenizer=tokenizer_ctrl, device=0, return_full_text=False)


'Sports', 'boxing', 'gymnastics', 'boxer'

prompts = ["Links In Sports boxing gymnastics boxer : "] # ctrl
result_gpt = gen_nlp_ctrl(prompts, max_length=128, \
                                do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                repetition_penalty=1.2, num_return_sequences= 16,\
                                clean_up_tokenization_spaces=True)




