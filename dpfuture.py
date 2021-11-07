import argparse,os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default='uci', type=str)
parser.add_argument("--samplecnt", default=128, type=int)
parser.add_argument("--future_steps", default=64, type=int)
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
ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)


def remove_str(sent):
    rml = ['(AP)', '(Reuters)', '(Canadian Press)', '&lt;b&gt;...&lt;/b&gt', '(AFP)', '(washingtonpost.com)', \
                '(NewsFactor)', '(USATODAY.com)', '(Ziff Davis)', '#39;' ]
    for word in rml:
        sent = sent.replace(word,'')

    sent.replace(' #39;', "'")
    return sent.strip(string.punctuation).strip()

if args.dsn =='agt':
    ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))

# for ix, row in ds.df_train.iterrows():
#     print(row['label_name'])
#     print(row['content'], '\n')

ds.df_train['content'] = ds.df_train['content'].map(lambda x: x.strip(string.punctuation).strip())
#ds, proper_len = process_ds(ds, 32)

from utils.transblock import * 
with tf.distribute.MirroredStrategy().scope():
    model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
model_cls.load_weights("./model_cls/model_full_{}.h5".format(args.dsn))   


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



for ix, row in ds.df_train.iterrows():
    t0 = time.time()
    sent = row['content']
    label = row['label']
    label_name = row['label_name']
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
    dfaug = df_future_threds.loc[(df_future_threds['cls_label']==label) & \
                 (df_future_threds['cls_score']>=args.cls_score_thres)  & \
                 (df_future_threds['score']>0)]
    print("reduce rate ===>", dfaug.shape[0] / df_future_threds.shape[0], dfaug.shape[0] )
    
    print(label_name, "==>", sent)
    print('\n')
    t1 = time.time()
    print("time cost:", (t1-t0) / 60 )

    if dfaug.shape[0] == 0:
        print("reduct_empty")  
        continue 

    for s in dfaug.head(8)['content'].tolist():
        print("gen==>", s.replace(sent, ''))
    print('\n\n')