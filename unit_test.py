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




# gpt neo
# from transformers import GPT2Tokenizer, GPTNeoForCausalLM
# tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir="./cache")
# gpt2 = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir="./cache")
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import pandas as pd 
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained('roberta-large',cache_dir="./cache",local_files_only=True) 
model = AutoModelWithLMHead.from_pretrained('roberta-large',cache_dir="./cache",local_files_only=True)
nlp_fill = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0)


from utils.load_data import * 
ds = load_data(dataset='ag', samplecnt= 1024)



row = ds.df_train.sample(1)
print(row['content'].tolist()[0], row['label_name'].tolist()[0])

filled_result = nlp_fill("{} are melting faster than before in some regions from the Arctic to the Alps \
    but others are getting bigger, scientists said on Friday.".format(nlp_fill.tokenizer.mask_token))
df = pd.DataFrame(filled_result)
print(df)




with tf.distribute.MirroredStrategy().scope():
    model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
    model_cls.load_weights("./model_cls/model_full_{}.h5".format('ag'))   




sent = "Emergence of community - acquired infections due to ESBL" # health
sent = "100 bodies found at the scene of plane disaster" # health

Castle'Season 6 Spoilers : Season Finale Sees'Trouble'With Beckett's Past <=== entertainment
Fitch Publishes Sector Credit Factors for Japanese Insurers <=== business
Obama announces new sanctions on Russia <=== business

sent = "U. S. senator demands compensation fund for recalled GM cars" # science and technology
sent = "Federal jury orders tech giant Samsung to pay"

sent = 'FDA gives green light to migraine prevention tool'

filled_result = nlp_fill("Glaciers Shrink, But Some Resist Global Warming (Reuters) Reuters - Glaciers are melting faster than before in\some regions from the Arctic to the Alps but others are getting\bigger, scientists said on Friday.".format(nlp_fill.tokenizer.mask_token))

filled_result = nlp_fill("Federal jury orders business giant {} to pay".format(nlp_fill.tokenizer.mask_token))

filled_result = nlp_fill("Federal jury orders military giant {} to pay".format(nlp_fill.tokenizer.mask_token))


filled_result = nlp_fill("{} 6 Spoilers : Season Finale Sees'Trouble'With Beckett's Past".format(nlp_fill.tokenizer.mask_token), top_k=10)

df = pd.DataFrame(filled_result)
print(df)
















input_ids = tokenizer_gpt2.encode(sent, return_tensors="tf")
# get logits of last hidden state
next_token_logits = gpt2(input_ids).logits[:, -1, :] / 1.0


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








