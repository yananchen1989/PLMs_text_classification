import copy,argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers,os,nltk,random
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForNextSentencePrediction
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity
from utils.load_data import * 
from utils.transblock import * 
from utils.encoders import *

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--max_length", default=64, type=int)
parser.add_argument("--samplecnt", default=0.7, type=float)
parser.add_argument("--writedata", default=0, type=int)
parser.add_argument("--epochs", default=12, type=int)
parser.add_argument("--ft", default=0, type=int)
parser.add_argument("--init_kl_coef", default=0.2, type=float)
parser.add_argument("--fbs", default=8, type=int)
parser.add_argument("--w_simi", default=0.5, type=float)
parser.add_argument("--w_cce", default=1.0, type=float)
parser.add_argument("--ppo_batchsize", default=32, type=int)
args = parser.parse_args()

#os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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
assert gpus

def get_ds(file, shuffle=True, epochs=1):
    ds = tf.data.experimental.make_csv_dataset(
        file_pattern = file, select_columns=['content'],
        batch_size=args.ppo_batchsize, num_epochs=epochs, prefetch_buffer_size=12800, num_parallel_reads=4,
        #label_name='label',
        shuffle=shuffle,
        shuffle_buffer_size=12800)
    return ds

def get_pairs(samples, label=2):
    ixs = []
    infos = []
    for ix in range(samples.shape[0]):
        para = samples[ix]
        sents = nltk.sent_tokenize(para)
        if len(sents) < 2:
            continue 
        for sent in sents:
            if len(sent.split(' ')) <= 10:
                continue
            infos.append((sent, ix))
            ixs.append(ix)

    ixs = list(set(ixs))

    pairs = []
    for i in range(len(infos)-1):
        if infos[i][-1] == infos[i+1][-1]:
            pairs.append((infos[i][0], infos[i+1][0], 1))
            candidates = [j for j in ixs if j != infos[i][-1]]
            ix_ = random.sample(candidates, 1)[0]
            candidates_e = [j for j in infos if j[-1]==ix_]
            ee = random.sample(candidates_e, 1)[0]
            pairs.append((infos[i][0], ee[0], 0))
        else:
            continue
    random.shuffle(pairs)
    return pairs

if args.dsn  in ['yahoo','yelp2','imdb','ag']:
    ds_train = get_ds("./torch_ds/{}_train.csv".format(args.dsn), True)
    ds_test = get_ds("./torch_ds/{}_test.csv".format(args.dsn), False)

elif args.dsn =='news':
    ds_train_cc = get_ds("./torch_ds/cc_train.csv", True)
    ds_train_cnndm = get_ds("./torch_ds/cnndm.csv", True)

    for cc_step, trunk in ds_train_cc.enumerate():
        continue
    for cnndm_step, trunk in ds_train_cnndm.enumerate():
        continue

    datasets_list = [ds_train_cnndm, ds_train_cc]
    choice_dataset = tf.data.Dataset.range(2).repeat(cc_step+cnndm_step)
    ds_train = tf.data.experimental.choose_from_datasets(datasets_list, choice_dataset)
    #ds_train = ds_train_cnndm.concatenate(ds_train_cc)
    ds_test = get_ds("./torch_ds/cc_test.csv", False)     

elif args.dsn == 'snli':
    df_train = pd.read_csv("./torch_ds/snli_1.0_train.csv")
    df_test = pd.read_csv("./torch_ds/snli_1.0_test.csv")

    df_train = df_train.loc[(df_train['similarity']!='-') & (df_train['similarity']!='neutral')]
    df_test = df_test.loc[(df_test['similarity']!='-') & (df_test['similarity']!='neutral')]

    df_train['label'] = df_train["similarity"].apply(
        lambda x: {"contradiction":0, "entailment":1}.get(x)
    )
    df_test['label'] = df_test["similarity"].apply(
        lambda x: {"contradiction":0, "entailment":1}.get(x)
    )

    ds_train = tf.data.Dataset.from_tensor_slices((df_train[["sentence1", "sentence2"]].values.astype("str"), df_train['label'].values))
    ds_train = ds_train.shuffle(buffer_size=12800).batch(64).prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((df_test[["sentence1", "sentence2"]].values.astype("str"), df_test['label'].values))
    ds_test = ds_test.batch(64)
#


for train_step, trunk in ds_train.enumerate():
    continue
for test_step, trunk in ds_test.enumerate():
    continue

print('train_step:', train_step.numpy(), 'test_step:', test_step.numpy())








model_cls_pair  = get_model(128)


ds = load_data(dataset='ag', samplecnt= -1)

classes = ds.df_test['label_name'].unique()

accs = []

for ix, row in ds.df_test.iterrows():

    train_x = get_ids_(row['content'], classes,  128)
    preds = model_cls_pair.predict(train_x)
    pred_ix = np.argmax(preds[:,0])
    if classes[pred_ix] == row['label_name']:
        accs.append(1)
    else:
        accs.append(0)
    print(sum(accs) / len(accs))



df_cc = get_cc_news(0.3)

from sklearn.metrics import accuracy_score

while 1:
    df_batch = df_cc.sample(512)
    pairs = get_pairs(df_batch['content'].values)
    df_pairs = pd.DataFrame(pairs, columns=['text1', 'text2', 'label'])
    df_pairs['label'] = df_pairs['label'].map(lambda x: 0 if x==1 else 1)
    train_x = get_ids(df_pairs, 128)
    preds = model_cls_pair.predict(train_x, batch_size=64)
    pred_ix = np.argmax(preds, axis=1)    
    acc_tmp = accuracy_score(df_pairs['label'].values, pred_ix)
    print(acc_tmp)





# df_pair = pd.read_csv("df_content_prompt_test_one_shot.csv")
# df_train, df_test = train_test_split(df_pair, test_size=0.15)


# train_x = get_ids(df_train, 64)
# test_x = get_ids(df_test, 64)

# model_cls_pair.fit(train_x, df_train['label'].values , \
#              batch_size=32, epochs=10, \
#             validation_data=(test_x, df_test['label'].values), verbose=1)


# model_cls_pair.load_weights("./cls_pairs/bert_{}.h5".format(args.dsn))

###### train cls
'''
lr = 4e-5
base_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

@tf.function
def train_step_base(input_ids, attention_masks, token_type_ids, labels):
    # generator_ral update
    with tf.GradientTape() as tape:
        predictions = model_cls([input_ids, attention_masks, token_type_ids])
        #loss = tf.keras.losses.SparseCategoricalCrossentropy()(labels, predictions)
        loss = tf.keras.losses.BinaryCrossentropy()(labels, predictions)
    grads = tape.gradient(loss, model_cls.trainable_weights)
    base_optimizer.apply_gradients(zip(grads, model_cls.trainable_weights))
    return loss
if args.dsn == 'snli':
    for epoch in range(args.epochs):
        #m_ = tf.keras.metrics.SparseCategoricalAccuracy()
        m_ = tf.keras.metrics.BinaryAccuracy()
        for step, trunk in ds_train.enumerate():
            
            sentence_pairs = [[ii[0].decode(), ii[1].decode()] for ii in trunk[0].numpy()]
            input_ids , attention_masks, token_type_ids = get_ids(sentence_pairs)
            labels = trunk[1].numpy()

            preds = model_cls([input_ids, attention_masks, token_type_ids], training=False)
            m_.update_state(labels, preds)
            loss = train_step_base(input_ids, attention_masks, token_type_ids, labels) 
            print("epoch:", epoch, "step:", step.numpy(), 'of', train_step.numpy(), \
                'acc:', m_.result().numpy(), 'loss:', loss.numpy()) 
            if step % 500 == 0 and step > 0:
                print('test acc:')
                #m_test = tf.keras.metrics.SparseCategoricalAccuracy()
                m_test = tf.keras.metrics.BinaryAccuracy()
                for _, trunk_ in ds_test.enumerate():
                    sentence_pairs = [[ii[0].decode(), ii[1].decode()] for ii in trunk_[0].numpy()]
                    input_ids , attention_masks, token_type_ids = get_ids(sentence_pairs)
                    labels = trunk_[1].numpy()
                    preds = model_cls([input_ids, attention_masks, token_type_ids], training=False)
                    m_test.update_state(labels, preds)
                print('test acc:', m_test.result().numpy() )


else:
    cur_test_acc = 0
    for epoch in range(args.epochs):
        #m_ = tf.keras.metrics.SparseCategoricalAccuracy()
        m_ = tf.keras.metrics.BinaryAccuracy()
        for step, trunk in ds_train.enumerate():
            pairs = get_pairs(trunk['content'].numpy())
            pairs_ = random.sample(pairs, int(args.samplecnt * len(pairs)))
            ix = 0
            while ix < len(pairs_):
                pairs_batch = pairs_[ix:ix+32] 
                sentence_pairs = [[ii[0], ii[1]] for ii in pairs_batch]
                labels =  tf.convert_to_tensor([ii[-1] for ii in pairs_batch])
                input_ids, attention_masks, token_type_ids = get_ids(sentence_pairs)
                #predict
                preds = model_cls([input_ids, attention_masks, token_type_ids], training=False)
                m_.update_state(labels, preds)
                
                # learn
                loss = train_step_base(input_ids, attention_masks, token_type_ids, labels) 
                print("epoch:", epoch, "step:", step.numpy(), 'of', train_step.numpy(), \
                    'acc:', m_.result().numpy(), 'loss:', loss.numpy())   
                ix += 32


        print('begin to test')
        #m_test = tf.keras.metrics.SparseCategoricalAccuracy()
        m_test = tf.keras.metrics.BinaryAccuracy()
        for step_, trunk_ in enumerate(ds_test):
            pairs = get_pairs(trunk_['content'].numpy())   
            pairs_batch = random.sample(pairs, 64)            
            sentence_pairs = [[ii[0], ii[1]] for ii in pairs_batch]
            labels =  tf.convert_to_tensor([ii[-1] for ii in pairs_batch])                      
            input_ids, attention_masks, token_type_ids = get_ids(sentence_pairs)
            preds = model_cls([input_ids, attention_masks, token_type_ids], training=False)
            m_test.update_state(labels, preds)

        print('epoch:', epoch, 'test acc:', m_test.result().numpy())   
        if m_test.result().numpy() > cur_test_acc:
            model_cls.save_weights("./cls_pairs/bert_{}.h5".format(args.dsn))
            cur_test_acc = m_test.result().numpy()
'''
ds = load_data(dataset=args.dsn, samplecnt= -1)
num_classes = ds.df_test['label'].unique().shape[0]
model_full = get_model_bert(num_classes)
model_full.load_weights("./cls/model_full_{}.h5".format(args.dsn) )     


################## 
from utils.encoders import *
enc = encoder('dan', 'cuda')

import torch
from sklearn.metrics.pairwise import cosine_distances
from transformers import GPT2Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='./cache', local_files_only=True)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type=='cuda'


from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt

config = {
    # "lm_name": "lvwerra/gpt2-imdb",
    # "ref_lm_name": "lvwerra/gpt2-imdb",
     "cls_model_name": "lvwerra/bert-imdb",
    #"tk_name": "gpt2",
    #"steps": 25600,
    "forward_batch_size": 16,
    "ppo_epochs": 4,   
    #"txt_in_len": 5,
    #"txt_out_len": 15,
    "batch_size":args.ppo_batchsize,
    "lr": 1.41e-5,
    "init_kl_coef":args.init_kl_coef,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": 0.2,
    "cliprange_value": 0.2,
    "vf_coef":.1, 
}

if args.ft :
    model_output_path = "./finetune_gpt2_unspervised/{}".format(args.dsn)
    train_file = './fintune_csvs/{}_train_finetune_unspervised.csv'.format(args.dsn)
    validation_file = './fintune_csvs/{}_test_finetune_unspervised.csv'.format(args.dsn)
    #if args.dsn in ['yahoo','yelp2','imdb','ag']:
    ds = load_data(dataset=args.dsn, samplecnt= -1)
    df_train_ft = ds.df_train
    df_test_ft = ds.df_test
    # elif args.dsn == 'news':
    #     df_cc = pd.read_csv("./torch_ds/cc_news.csv", nrows=100000)
    #     df_cnndm = pd.read_csv("./torch_ds/cnndm.csv", nrows=100000)
    #     df = pd.concat([df_cnndm, df_cc])
    #     del df['title']
    #     df_train_ft, df_test_ft = train_test_split(df, test_size=0.1)

    df_train_ft['text'] = df_train_ft['content'].replace('<br />',' ') + gpt2_tokenizer.eos_token
    df_test_ft['text'] = df_test_ft['content'].replace('<br />',' ') + gpt2_tokenizer.eos_token
    df_train_ft.to_csv(train_file, index=False)
    df_test_ft.to_csv(validation_file, index=False)       
    os.system(
    "CUDA_VISIBLE_DEVICES={} python -u ./run_clm_no_trainer.py \
            --num_train_epochs 1 \
            --train_file {} \
            --validation_file {} \
            --model_name_or_path gpt2 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --output_dir {} \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --block_size 256".format(args.gpu,  train_file, validation_file, model_output_path) ) 

else:
    model_output_path = 'gpt2'

gpt2_model_ref_trl = GPT2HeadWithValueModel.from_pretrained(model_output_path)
gpt2_model_ref_trl.to(device)

gpt2_model_trl = GPT2HeadWithValueModel.from_pretrained(model_output_path)
gpt2_model_trl.to(device)

ppo_trainer = PPOTrainer(gpt2_model_trl, gpt2_model_ref_trl, **config)



bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
even_loss = bce(np.ones(1).reshape(-1,1), np.array(0.5).reshape(-1,1 )).numpy()[0]
even_loss_cce = cce(np.array([1] + [0]*(num_classes-1)), np.array([1/num_classes]*num_classes)).numpy()

os.makedirs("./gpt_ppo_from_pairs", exist_ok=True) 

from collections import deque
# next sentence rewards
for epoch in range(args.epochs):
    df_batch_ll = deque([], maxlen=100)
    for step, trunk in ds_train.enumerate():
        contents = [content.decode().replace('<br />',' ') for content in trunk['content'].numpy()]

        df_batch = pd.DataFrame(contents, columns=['content'])

        #ix = 0
        #while ix < df_train.shape[0]:
        torch.cuda.empty_cache()    

        #df_batch = df_train[ix:ix+ppo_batchsize].copy()
    
        # get responses
        # min_tokens_len = min([ii.shape[1] for ii in df_batch['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt")).tolist()])
        # assert min_tokens_len >= 2
        # print( 'min_tokens_len:', min_tokens_len)

        # df_batch['query_tokens'] = df_batch['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt", \
        #                         truncation=True, padding=True, max_length=min_tokens_len)\
        #                         .to(device)[0, :min_tokens_len])


        # df_batch['query_tokens'] = df_batch['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt", \
        #                         truncation=True, padding=True, max_length=min_tokens_len+100)\
        #                         .to(device)[0, :]).tolist()

        contents_tokens_lens = [ii.shape[1] for ii in df_batch['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt")).tolist()]    
        maxlen = min(int(sum(contents_tokens_lens) / len(contents_tokens_lens)), 60)
        print('step', step.numpy(), 'of', train_step.numpy(), 'tokens:', maxlen )

        query_tensors = gpt2_tokenizer(contents, return_tensors="pt", pad_to_max_length =True, \
                                truncation=True, padding=True, max_length=maxlen)['input_ids'].to(device)

        #df_batch['query'] =  df_batch['query_tokens'].map(lambda x: gpt2_tokenizer.decode(x))

        df_batch['query'] =  [gpt2_tokenizer.decode(t,clean_up_tokenization_spaces=True, skip_special_tokens=True) for t in query_tensors]
        #query_tensors = torch.stack(df_batch['query_tokens'].tolist())

        response_tensors_ll = []
        for i in range(int(args.ppo_batchsize/args.fbs)):
            response  = respond_to_batch(gpt2_model_trl, query_tensors[i*args.fbs:(i+1)*args.fbs],
                                          txt_len = maxlen, top_p=0.9, \
                                          temperature=1.2, min_tokens_to_keep=1)
            response_tensors_ll.append(response)
        response_tensors = torch.cat(response_tensors_ll)


        df_batch['responses'] = [gpt2_tokenizer.decode(response_tensor, clean_up_tokenization_spaces=True, skip_special_tokens=True) \
                            for response_tensor in response_tensors]


        # sentence_pairs = [[ii[0], ii[1]] for ii in df_batch[['query','responses']].values]
        # input_ids, attention_masks, token_type_ids = get_ids(sentence_pairs)
        # preds = model_cls_pair([input_ids, attention_masks, token_type_ids], training=False)
        
        # loss = bce(np.ones(preds.shape[0]).reshape(-1,1),  preds.numpy()).numpy()
        
        # rewards = even_loss - loss

        #rewards_binary_pair = preds.numpy().reshape(-1) - 0.5 
        #if ds.df_test['label'].unique().shape[0]==2:
        pred_content = model_full.predict(df_batch['content'].values, batch_size=64)
        pred_response = model_full.predict(df_batch['responses'].values, batch_size=64)  
        loss = cce(pred_content, pred_response)
        df_batch['ccer'] = even_loss_cce - loss
        
        embeds_ori_contents = enc.infer(df_batch['content'].tolist(), batch_size=64 )
        embeds_syn_contents = enc.infer(df_batch['responses'].tolist(), batch_size=64 )
        assert embeds_ori_contents.shape[0] == embeds_syn_contents.shape[0]
        cos_matrix = cosine_similarity(embeds_ori_contents, embeds_syn_contents)
        simis = [cos_matrix[j][j] for j in range(df_batch.shape[0])]
        df_batch['cosine_simi'] = simis

        df_batch['reward'] = df_batch['cosine_simi'] * args.w_simi + df_batch['ccer'] * args.w_cce

        stats = ppo_trainer.step(query_tensors, response_tensors, torch.tensor(df_batch['reward'].values).to(device))

        df_batch_ll.append(df_batch)
        

        if step.numpy() % 20 == 0 and step.numpy()> 0:
            df_epoch = pd.concat(df_batch_ll)
            print('epoch:',epoch, \
                    'rewards:', df_epoch['reward'].mean(), \
                    'simi:', df_epoch['cosine_simi'].mean(),\
                    'ccer:', df_epoch['ccer'].mean() )

            for ix, row in df_epoch.sample(8).iterrows():
                print('epoch:',epoch, 'step:',step.numpy(), \
                     'reward:', round(row['reward'],4), \
                       'simi:', round(row['cosine_simi'],4), \
                       'ccer:', round(row['ccer'],4) )
                print('ori==>', row['content'].replace('\n',' ') )
                print('syn==>', row['responses'].replace('\n',' ') )
    gpt2_model_trl.save_pretrained("./gpt_ppo_from_pairs/{}_gptppo_epoch{}".format(args.dsn, epoch)) 




