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

#from utils.flair_ners import * 

# gpt neo
# from transformers import GPT2Tokenizer, GPTNeoForCausalLM
# tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir="./cache")
# model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir="./cache",  local_files_only=True)

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs, labels=inputs["input_ids"])
# loss = outputs.loss
# logits = outputs.logits

contents_syn_ix = [' that allow company managers scheduling 30 full days without security clearances!', ' review.. Talks began years ago through Editorial Board Chairman Sullivan Longweatherhill (I-La.). Blanket announced at teleconference Friday he wlf know Senator Edmund Day while being accompanied there by White House Communications Director Richard Worden Called Lockhart TE during his long public Eureka impeachment tirade Thursday evening Calls first panel head Holder later Tuesday JR took part in panels sponsored by former', ' requirements. Democrats started backjumping trying many times last year long causing major edge because some employees were half paid (and unemployment rate was at 4 percent as I wrote yesterday) #wisconsin Fires Group Surprises Warnings.. YES it Feels That Way Time Attendred Done Count 9 out for Ave night 5 fleewatch bay clinic talks stoppek 66 FI llor COVER FAST', ' until they remove such details from holiday bonuses instituted amid fighting over Iraq aid -- a shift pursued by Norris escalation procslowed at Saratoga Naval Base this summer following battles in Sacramento district seven caused severe losses. PD became assistant chief when Dan Gallagher approached Former Reposauce Marcos-Pres Lindquist wanted security \u200bseparated... The Kellogg Sodag usiing', ' raises rue aggressive conservative in-state politics attempt something sexy local politicians stand up their fists!!!!! -- Shed Avianious Carrying Chicken To Nowbeans Chicken greens... Andy Hitchcock looked at segregation protection from Jeff Bell until he moved; they should be opened anyway Phil Tully sliced chicken pizza hole Neil Mitchell talked Net $100 JakaClinicLathering Gen Hank Huh talks', ' days even as Republicans tilted GOP while Holder asserted intense skepticism). but attack Coloradans were ag! h5 sp </th><Section Id="1173">fi God Linson Believes Palin Can Honor HERO Act 1961</s> Products in trouble Lolz 3 20383 10 -->FOLLOW Mike Printi --@uangelo printish Prostitutes Spread Degenerate Selling', ' relief during extramarital affairs into Sunday, according just c h sons timers May 27 16:42 Proprietary bills w others indicate cloture victories BBC Deawissa Schaible 24 07 pages 2001 tight forecast Similar error 3 mean 94 Albert Sorel Chicago National Shooting Ce conf np(7) 1435706 30004 ( 34 ) Brunswick 788733 13312 359', ' that threaten interstate workers fighting retribution after striking at work Sept 22." In other words GOP candidate Govudd Douglas Ducey took credit out in hot butter speech a 12-inning delayed debate coinaded \'Jerry Neiberg credited away Joe wants wheels into weld breaker.\' Unfortunately rich South Dakota Legislatureive only gets money now several years later throughout UCA./JERSEYG Nader']
labels_candidates = ['Business', 'Democratic', 'Bush', 'Wis', 'John Edwards', 'Wisconsin', 'Edwards Hits Bush on Overtime Pay Rules RACINE']


t0 = time.time()
nli_result = nli_nlp([s.replace('</s>','')for s in contents_syn_ix],  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
t1 = time.time()
print(t1 - t0)



############## whole token generation ############  


from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#gpt2 = GPT2LMHeadModel.from_pretrained('ft_model_ft_ep')

gpt2.trainable = False
gpt2.config.pad_token_id = 50256
gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=len(gpus)-1, return_full_text=False)


ds = load_data(dataset='uci', samplecnt= -1)







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








