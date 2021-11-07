
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
import argparse,os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default='uci', type=str)
parser.add_argument("--future_steps", default=50, type=int)
parser.add_argument("--test_beams", default=256, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--candidates", default=64, type=int)
parser.add_argument("--cls_score_thres", default=0.95, type=float)


parser.add_argument("--gpu", default="0,1,2,3", type=str)
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
ds = load_data(dataset=args.dsn, samplecnt= 128)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: x.strip(string.punctuation))

from utils.transblock import * 
with tf.distribute.MirroredStrategy().scope():
    model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
model_cls.load_weights("./model_cls/model_{}.h5".format(dsn))   


# gpt2
from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)
# no

gen_d = {}
for g in args.gpu.split(","):
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
    # tc pp
    #gpt2 = GPT2LMHeadModel.from_pretrained('ft_model_{}_{}'.format('t5', 'ep') )
    gpt2.trainable = False
    gpt2.config.pad_token_id = 50256

    gen_d[int(g)]  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, \
                    device=int(g), return_full_text=False)




from threading import Thread
def gen_vs(sent, future_steps, test_beams, model_cls, di):
    tokens_len_ori = tokenizer_gpt2.encode(sent, return_tensors="pt").shape[1]
    result_ = gen_d[di]([sent], max_length=tokens_len_ori + future_steps, \
                                  do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                                repetition_penalty=1.0, num_return_sequences=test_beams, clean_up_tokenization_spaces=True)
    x = np.array([ii['generated_text'] for ii in result_])
    y = np.array([label] * x.shape[0])
    eval_result_ori = model_cls.evaluate(x, y, batch_size=args.batch_size, verbose=0)    
    eval_result_oris.append(eval_result_ori[0])
    print(eval_result_ori[0])

def gengen_vs(sent, future_steps, candidates, test_beams, model_cls, di):
    tokens_len_ori = tokenizer_gpt2.encode(sent, return_tensors="pt").shape[1]
    result_0 = gen_d[di]([sent], max_length=tokens_len_ori + future_steps, do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                                repetition_penalty=1.0, num_return_sequences=candidates, clean_up_tokenization_spaces=True)
    #print("result_0 generated")
    result_1 = gen_d[di]([ ii['generated_text'].strip().replace('\n',' ') for ii in result_0], max_length=future_steps+future_steps, \
                                  do_sample=True, top_p=0.9, top_k=0, temperature=1,\
                                repetition_penalty=1.0, num_return_sequences=test_beams, clean_up_tokenization_spaces=True)
    #print("result_1 generated")

    scores = []
    for i in range(len(result_1)): #  32 * 256
        x = np.array([ii['generated_text'] for ii in result_1[i]])
        y = np.array([label] * x.shape[0])
        eval_result = model_cls.evaluate(x, y, batch_size=args.batch_size, verbose=0) 
        #print('\n' , result_0[i]['generated_text'])
        score = loss_ori - eval_result[0] 
        #print("loss_diff:", score)
        scores.append(score)

    df_future = pd.DataFrame(zip([ ii['generated_text'].strip().replace('\n', ' ') for ii in result_0], scores), \
                                        columns=['content','score'])
    df_future_ll.append(df_future)



while 1:    
    row = ds.df_train.sample(1)
    sent = row['content'].tolist()[0]
    label = row['label'].tolist()[0]
    label_name = row['label_name'].tolist()[0]
    torch.cuda.empty_cache()
    eval_result_oris = []
    threads = []
    for di in range(len(args.gpu.split(","))):
        t = Thread(target=gen_vs, args=(sent, args.future_steps, args.test_beams, model_cls, di))
        t.start()
        threads.append(t)

    # join all threads
    for t in threads:
        t.join()

    loss_ori = sum(eval_result_oris) / len(eval_result_oris)

    torch.cuda.empty_cache()

    df_future_ll = []
    threads = []
    for di in range(len(args.gpu.split(","))):
        t = Thread(target=gengen_vs, args=(sent, args.future_steps, args.candidates, args.test_beams, model_cls, di))
        t.start()
        threads.append(t)

    # join all threads
    for t in threads:
        t.join()

    df_future_threds = pd.concat(df_future_ll)

    df_future_threds.sort_values(by=['score'], ascending=False, inplace=True)
    #sents = df_future.head(8)['content'].tolist()

    preds = model_cls.predict(df_future_threds['content'].values, batch_size= args.batch_size, verbose=0)

    df_future_threds['cls_score'] = preds[:, label] 
    df_future_threds['cls_label'] = preds.argmax(axis=1)
    dfaug = df_future_threds.loc[(df_future_threds['cls_label']==label) & (df_future_threds['cls_score']>=args.cls_score_thres)]
    print("reduce rate ===>", dfaug.shape[0] / df_future_threds.shape[0] )
    assert dfaug.shape[0] >= 8

    print(label_name, "==>", sent)
    print('\n')
    for s in dfaug.head(8)['content'].tolist():
        print("gen==>", s.replace(sent, ''))
    print('\n\n')













































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





'''
In [25]: sent
Out[25]: 'Apple, other tech firms formally agree to US $ 325m hiring accord'

In [26]: loss_future = vs(sent + ' FDA', label, 32, 128)
4/4 [==============================] - 1s 150ms/step - loss: 3.1841 - acc: 0.1875
4/4 [==============================] - 1s 151ms/step - loss: 3.2459 - acc: 0.1719
4/4 [==============================] - 1s 154ms/step - loss: 3.4739 - acc: 0.1250
4/4 [==============================] - 1s 150ms/step - loss: 3.4979 - acc: 0.1406
4/4 [==============================] - 1s 149ms/step - loss: 3.1692 - acc: 0.2188

In [27]: loss_future
Out[27]: 0.025892172008752823

In [28]: loss_future = vs(sent + ' FDA', label, 32, 128)
4/4 [==============================] - 1s 157ms/step - loss: 3.5690 - acc: 0.1094
4/4 [==============================] - 1s 156ms/step - loss: 3.4631 - acc: 0.1641
4/4 [==============================] - 1s 152ms/step - loss: 3.6237 - acc: 0.0859
4/4 [==============================] - 1s 154ms/step - loss: 3.0837 - acc: 0.1797
4/4 [==============================] - 1s 162ms/step - loss: 3.6443 - acc: 0.1406

In [29]: loss_future
Out[29]: 0.02716214470565319

In [30]: vs(sent + ' cyber security', label, 32, 128)
4/4 [==============================] - 1s 157ms/step - loss: 2.0523 - acc: 0.4375
4/4 [==============================] - 1s 153ms/step - loss: 2.0199 - acc: 0.4766
4/4 [==============================] - 1s 153ms/step - loss: 2.2970 - acc: 0.3906
4/4 [==============================] - 1s 152ms/step - loss: 2.1723 - acc: 0.3984
4/4 [==============================] - 1s 156ms/step - loss: 2.0566 - acc: 0.4141
Out[30]: 0.01655937284231186

In [31]: vs(sent + ' to', label, 32, 128)
4/4 [==============================] - 1s 156ms/step - loss: 2.4899 - acc: 0.3359
4/4 [==============================] - 1s 151ms/step - loss: 2.2010 - acc: 0.4375
4/4 [==============================] - 1s 152ms/step - loss: 1.9590 - acc: 0.4141
4/4 [==============================] - 1s 151ms/step - loss: 2.2989 - acc: 0.3672
4/4 [==============================] - 1s 152ms/step - loss: 2.2012 - acc: 0.4453
Out[31]: 0.017421722412109375
'''

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



