
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


sent = "Facebook acquires video ad company LiveRail for between US $ 500m"

sent = "Autism wave keeps growing"

sent = "Virus to cause spike in pork prices"



import os,string
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
from transformers import pipeline
from utils.load_data import * 
ds = load_data(dataset='uci', samplecnt= 128)
ds, proper_len = process_ds(ds, 256)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: x.strip(string.punctuation))

#from utils.flair_ners import * 


import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir="./cache")
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir="./cache",  local_files_only=True)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits


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
gpt2.config.pad_token_id=50256
gen_nlp_gpt2  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=0, return_full_text=False)


from utils.transblock import * 
model = get_model_bert(ds.df_test.label.unique().shape[0])
model.load_weights("./model_cls/model_uci.h5")          

         
while 1:
row = ds.df_train.sample(1)
sent = row['content'].tolist()[0]
label = row['label'].tolist()[0]
label_name = row['label_name'].tolist()[0]



result = gen_nlp_gpt2([sent], max_length=32, do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                            repetition_penalty=1.0, num_return_sequences=512, clean_up_tokenization_spaces=True)

sents_future = np.array([ii['generated_text'].strip() for ii in result ])
complete_r = sents_future.shape[0] / len(result)

print(sent)
print("completion rate:", complete_r)
print('\n')



preds = model.predict(sents_future, batch_size=32)
pred_labels = preds.argmax(axis=1)
pred_scores = preds.max(axis=1)






x = sents_future.reshape(-1,1)
y = np.array([label] * sents_future.shape[0])

eval_result = model.evaluate(x, y, batch_size=32)

print("future loss:", eval_result[0], "future acc:", eval_result[1])
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






from transformers import GPT2Tokenizer, top_k_top_p_filtering, GPT2LMHeadModel
import torch
from torch.nn import functional as F

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="./cache", local_files_only=True)
model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="./cache", local_files_only=True)

device0 = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
model.to(device0)
sent = "Federal jury orders tech giant Samsung to pay"

sent = 'FDA gives green light to migraine prevention tool'


temperature=1.0

for step in range(10):
    row = ds.df_train.sample(1)
    sent = row['content'].tolist()[0]
    label_name = row['label_name'].tolist()[0]
    print("label==>", label_name)
    print("ori==>", sent)
    
    for _ in range(64):
        input_ids = tokenizer.encode(sent, return_tensors="pt").to(device0)

        # get logits of last hidden state
        next_token_logits = model(input_ids).logits[:, -1, :] / temperature

        # filter
        filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=0, top_p=0.9)
        filtered_next_token_logits_ = filtered_next_token_logits.cpu().detach().numpy()[0]
        next_logits = [ii for ii in filtered_next_token_logits_ if not math.isinf(ii)]
        print("valid next logits:", len(next_logits))
        # sample
        probs = F.softmax(filtered_next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([input_ids, next_token], dim=-1)

        sent = tokenizer.decode(generated.tolist()[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)

    print("gen==>", sent.replace('\n',' '))
    print('\n')


