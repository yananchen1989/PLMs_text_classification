
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


############## whole token generation ############  





import glob
files = glob.glob("./dpfuture*.log")

infos_ori = []
infos_aug = []
for file in files:

    with open(file, 'r') as f:

        lines = f.readlines()
    start_ix = []
    for i, line in enumerate(lines):
        if "reduce rate ===>" in line:
            start_ix.append(i)

    for i, j in zip(start_ix[0:len(start_ix)-1], start_ix[1:len(start_ix)] ):
        content_gens = []
        for line  in lines[i:j]:
            
            if " ==> " in line:
                label = line.split(" ==> ")[0]
                content = line.split(" ==> ")[1].strip()
            if line.startswith("gen==> "):
                content_gens.append(line.replace("gen==> ", "").strip())
        infos_ori.append((label, content ))

        for at in range(min(4, len(content_gens))):
            infos_aug.append((label, content_gens[at]))


df_ori = pd.DataFrame(infos_ori, columns=['label_name', 'content'])
df_ori['label'] = df_ori['label_name'].map(lambda x: ixl_rev[x])

df_aug = pd.DataFrame(infos_aug, columns=['label_name', 'content'])
df_aug['label'] = df_aug['label_name'].map(lambda x: ixl_rev[x])

acc_noaug, model_cls = thread_testing('test', df_ori, ds.df_test)



acc_noaug, model_cls = thread_testing('test', pd.concat([df_ori,df_aug]),  ds.df_test) 

  

















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






for dsn in ['uci','ag','stsa']:
    ds = load_data(dataset=dsn, samplecnt= -1)

    x_train, y_train = get_keras_data(ds.df_train)
    x_test, y_test = get_keras_data(ds.df_test)

    with tf.distribute.MirroredStrategy().scope():
    #with tf.device('/GPU:0'):
        model = get_model_bert(ds.df_test.label.unique().shape[0])


    history = model.fit(
        x_train, y_train, batch_size=32, epochs=12, \
        validation_data=(x_test, y_test), verbose=1, validation_batch_size=64, validation_freq=1,
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=4, mode='max',restore_best_weights=True)]
    )
    model.save_weights("./model_cls/model_full_uci.h5")


infos = []
infos.append(('aaa',1, 'sent1', ['aaa','bbb','ccc']))
infos.append(('bbb',2, 'sent2', ['rrr','fff','ggg']))
infos.append(('bbb',3, 'sent3', ['fff','nnn','bbb']))

