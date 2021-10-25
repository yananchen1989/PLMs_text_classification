
sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

sent = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''


sent = "Federal jury orders tech giant Samsung to pay"

sent = 'FDA gives green light to migraine prevention tool'

generated_text = "Editorial cartoon, Sun Jun 6: Asian Innovation, how unfriendly of you as $ 40 makes China true hank hacking 1] recently    You talk about personal electronic interaction with someone by surflinking their history into a post-doc and you walk into the office, and as an e-mailer you are literally texting me, making tons of calls, walking into my place in two hours and"

sent = '''
The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com) SPACE.com - TORONTO, Canada -- A second\team of rocketeers competing for the  #36;10 million Ansari X Prize, a contest for\privately funded suborbital space flight, has officially announced the first\launch date for its manned rocket.
'''

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
from transformers import pipeline
from utils.load_data import * 
ds = load_data(dataset='ag', samplecnt= 128)

from utils.flair_ners import * 


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
gpt2 = GPT2LMHeadModel.from_pretrained('ft_model_{}_{}'.format('t5', 'ep') )

gpt2.trainable = False
gpt2.config.pad_token_id=50256
gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=0, return_full_text=False)


for ix, row in ds.df_train.sample(frac=1).iterrows():
    content = row['content']
    sent_ners = get_ners(content)
    if not sent_ners:
        print(content)
         


gen_nlp(content+tokenizer_gpt2.sep_token, max_length=256, do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                            repetition_penalty=1.0, num_return_sequences=4, clean_up_tokenization_spaces=True)



# t5
import glob
from transformers import pipeline
from transformers import T5Tokenizer, AutoModelWithLMHead
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
print(tokenizer_t5)
# no ft
t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)

# ft:tc pp
ft_model_path = 'ft_model_{}_{}'.format('t5', 'tc')
checkpoint_files = glob.glob(ft_model_path+"/checkpoint_loss_*")
list.sort(checkpoint_files)
t5 = AutoModelWithLMHead.from_pretrained(checkpoint_files[0])  

########
gen_nlp  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=0)





sent = ds.df_train.sample(1)['content'].tolist()[0]

# t5
gen_nlp(sent + tokenizer_t5.eos_token, max_length=256, do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                            repetition_penalty=1.0, num_return_sequences=8, clean_up_tokenization_spaces=True)




'''
t5:
no: enough
tc: enough
pp: insufficient


gpt2:
no:
tc: enough
pp: insufficient
'''

