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
import argparse,os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default='uci', type=str)
parser.add_argument("--samplecnt", default=-1, type=int)
parser.add_argument("--future_steps", default=64, type=int)
parser.add_argument("--test_beams", default=256, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--candidates", default=64, type=int)
parser.add_argument("--cls_score_thres", default=0.8, type=float)
parser.add_argument("--max_aug_times", default=1, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gpu", default="7", type=str)
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

from utils.load_data import * 
from utils.transblock import * 
ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)
#ds, proper_len = process_ds(ds, 64)

ixl = {ii[0]:ii[1] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
ixl_rev = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
#seed = random.sample(list(range(10000)), 1)[0]

from threading import Thread
testbed_func = {"test":do_train_test_thread, "valid":do_train_test_valid_thread}


def thread_testing(testvalid, df_train, df_test):
    best_test_accs = []
    models = []

    for ddi in range(3):
        threads = []
        for di in range(1):
            t = Thread(target=testbed_func[testvalid], \
                        args=(df_train, df_test, best_test_accs, models, di + ddi*2, 100,  0))
            t.start()
            threads.append(t)

        # join all threads
        for t in threads:
            t.join()

    acc = round(np.array(best_test_accs).max(), 4)

    #model_best = models[np.array(best_test_accs).argmax()]
    return  acc


import glob
files = glob.glob("./log_dpfuture/dpfuture.{}.samplecnt_128.max_aug_times_1.candidates_128.test_beams_128.*.log".format(args.uci))

infos = []
for file in files:

    with open(file, 'r') as f:
        lines = f.readlines()
     
    start_ix = []
    for i, line in enumerate(lines):
        if 'eval_result_oris==>' in  line:
            start_ix.append(i)

    for i, j in zip(start_ix[0:len(start_ix)-1], start_ix[1:len(start_ix)] ):

        label_name = lines[i:j][2].split("==>")[0].strip()
        content_ori = lines[i:j][2].split("==>")[1].strip()
        content_syn = lines[i:j][4].replace("gen==>","").strip()
        infos.append((label_name, content_ori, content_syn))


df = pd.DataFrame(infos, columns=['label_name', 'content', 'content_syn'])
df['label'] = df['label_name'].map(lambda x: ixl_rev[x])

ds = load_data(dataset='uci', samplecnt= -1)
ds, proper_len = process_ds(ds, 64)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: x.strip(string.punctuation).strip())
ds.df_test['content'] = ds.df_test['content'].map(lambda x: x.strip(string.punctuation).strip())

df_all = pd.concat([ds.df_train, ds.df_test])

df_test = df_all.loc[~df_all['content'].isin(df['content'].tolist())]


for samplecnt in [32, 64, 128]:
    for trial in range(7):
        df_ = df.sample(frac=1).drop_duplicates(['content','label_name'])
        df_train = sample_stratify(df_, samplecnt)

        acc_noaug, _ = thread_testing('test', df_train[['label_name','label', 'content']], df_test)

        acc_aug, _ = thread_testing('test', \
            pd.concat([df_train[['label_name','label', 'content']], \
                      df_train[['label_name','label', 'content_syn']].rename(columns={'content_syn': 'content'}) ]), \
            df_test) 

        gain = (acc_aug - acc_noaug) / acc_noaug
        print("samplecnt:", samplecnt, "trial:", trial, 'gain:', gain, "<==", acc_noaug, acc_aug)

















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










