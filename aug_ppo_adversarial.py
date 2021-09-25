import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math,random, time
import numpy as np
import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split
from tensorflow import keras
from transformers import pipeline

from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="uci", type=str)
parser.add_argument("--samplecnt", default=128, type=int)
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--gpu", default=1, type=int)
parser.add_argument("--reward", default='hard', type=str) 

# ppo 
parser.add_argument("--temperature", default=1.0, type=float) 
parser.add_argument("--min_tokens_to_keep", default=1, type=int) 
parser.add_argument("--fbs", default=16, type=int)
parser.add_argument("--ppo_batchsize", default=32, type=int)
parser.add_argument("--init_kl_coef", default=0.2, type=float) 
parser.add_argument("--cliprange", default=0.2, type=float) 
parser.add_argument("--cliprange_value", default=0.2, type=float) 
parser.add_argument("--ref_ft", default=0, type=int)
parser.add_argument("--gpt_ft", default=0, type=int)

args = parser.parse_args()
print('args==>', args)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
assert device.type=='cuda'
from utils.load_data import * 
from utils.transblock import * 
from utils.gan_config import * 

assert gpus


@tf.function
def train_step_gan(prompts_tensor, prompts_syn_tensor, labels_tensor, labels_syn_tensor):
    combined_prompts = tf.concat([prompts_tensor, prompts_syn_tensor], axis=0)
    combined_labels = tf.concat([labels_tensor, labels_syn_tensor], axis=0)
    # generator_ral update
    with tf.GradientTape() as tape:
        predictions = model_gan(combined_prompts)
        loss = keras.losses.SparseCategoricalCrossentropy()(combined_labels, predictions)
    grads = tape.gradient(loss, model_gan.trainable_weights)
    gan_optimizer.apply_gradients(zip(grads, model_gan.trainable_weights))
    return loss

#. ppo
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
    "batch_size": args.ppo_batchsize ,
    "lr": 1.41e-5,
    "init_kl_coef":args.init_kl_coef,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": args.cliprange,
    "cliprange_value":args.cliprange_value,
    "vf_coef":.1, 
}

from transformers import GPT2Tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='./cache', local_files_only=True)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token 

if args.ref_ft:
    gpt2_model_ref_trl = GPT2HeadWithValueModel.from_pretrained(model_output_path)
else:
    gpt2_model_ref_trl = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_model_ref_trl.to(device)

if args.gpt_ft:
    gpt2_model_trl = GPT2HeadWithValueModel.from_pretrained(model_output_path)
else:
    gpt2_model_trl = GPT2HeadWithValueModel.from_pretrained('gpt2')
gpt2_model_trl.to(device)

ppo_trainer = PPOTrainer(gpt2_model_trl, gpt2_model_ref_trl, **config)


####### prepare data
ds = load_data(dataset=args.dsn, samplecnt=args.samplecnt)
#ds, max_len = process_ds(ds)

num_classes = ds.df_test.label.unique().shape[0]

def get_model_bert_for_gan(num_classes):

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string

    encoder = hub.KerasLayer("./resource/albert_en_base_2", trainable=True)

    encoder_inputs = preprocessor_layer(text_input)
    outputs = encoder(encoder_inputs)
    embed = outputs["pooled_output"]  

    if num_classes == 2:
        out = layers.Dense(1, activation='sigmoid')(embed)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        #model.compile(Adam(lr=2e-5), "binary_crossentropy", metrics=["binary_accuracy"])
    else:
        out = layers.Dense(num_classes, activation="softmax")(embed)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        #model.compile(Adam(lr=2e-5), "sparse_categorical_crossentropy", metrics=["acc"])
    return model

model_gan  = get_model_bert_for_gan(num_classes*2)
lr = 4e-5
gan_optimizer = keras.optimizers.Adam(learning_rate=lr)

#val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)


def reponse(df_batch):
    contents_tokens_lens = [ii.shape[1] for ii in df_batch['content'].map(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt")).tolist()]    

    maxlen = min(int(sum(contents_tokens_lens) / len(contents_tokens_lens)), 64)

    df_batch['query'] =  df_batch['content']
    query_tensors = gpt2_tokenizer(df_batch['query'].tolist(), return_tensors="pt", pad_to_max_length =True, \
                                truncation=True, padding=True, max_length=maxlen)['input_ids'].to(device)
        
    response_tensors_ll = []
    for i in range(int(df_batch.shape[0]/args.fbs)):
        response  = respond_to_batch(gpt2_model_trl, query_tensors[i*args.fbs:(i+1)*args.fbs],
                                      txt_len = maxlen, top_p=0.9, \
                                      temperature=args.temperature, min_tokens_to_keep=args.min_tokens_to_keep)
        response_tensors_ll.append(response)
    response_tensors = torch.cat(response_tensors_ll)

    df_batch['response'] = [gpt2_tokenizer.decode(response_tensor, clean_up_tokenization_spaces=True, skip_special_tokens=True).strip() \
                                for response_tensor in response_tensors]
    return df_batch, query_tensors, response_tensors


# row = df_batch.sample(1)
# print(row['label_name'])
# print(row['content'].tolist()[0])
# print(row['response'].tolist()[0])

# model_gan.predict(row['content'].values, batch_size=32)


for epoch in range(args.epoch):
    ds.df_train = ds.df_train.sample(frac=1)
    ix = 0
    while ix < ds.df_train.shape[0]:

        df_batch = ds.df_train[ix:ix+args.ppo_batchsize].copy()

        torch.cuda.empty_cache()   

        df_batch, query_tensors, response_tensors = reponse(df_batch)

        prompts = tf.convert_to_tensor(df_batch['content'].values)
        labels = tf.convert_to_tensor(df_batch['label'].values)

        prompts_syn = tf.convert_to_tensor(df_batch['response'].values)
        labels_syn = tf.convert_to_tensor(df_batch['label'].values+ num_classes )  

        preds_syn = model_gan.predict(prompts_syn, batch_size=32) 
        preds_ori = model_gan.predict(prompts, batch_size=32) 

        preds_ori_labels = preds_ori.argmax(axis=1)
        preds_syn_labels = preds_syn.argmax(axis=1)

        rewards = []
        for i in range(args.ppo_batchsize):
            if args.reward == 'hard':
                if preds_ori_labels[i] == preds_syn_labels[i]:
                    rewards.append(1)
                else:
                    rewards.append(-1)
            elif args.reward == 'soft':
                diff = np.abs(preds_ori[i] - preds_syn[i]).sum()
                rewards.append(1-diff)

        # train ppo               
        stats = ppo_trainer.step(query_tensors, response_tensors, torch.tensor(rewards).to(device))    

        loss_gan = train_step_gan(prompts, prompts_syn,  \
                            tf.cast(labels, tf.float32), tf.cast(labels_syn, tf.float32))

        print(ix, 'of', args.samplecnt*num_classes, loss_gan.numpy())
        ix += args.ppo_batchsize


    preds = model_gan.predict(ds.df_test['content'].values, batch_size=32)  
    preds_accum =  preds[:,:num_classes] + preds[:,num_classes:]
    acc_2class = accuracy_score(ds.df_test['label'].values, preds_accum.argmax(axis=1))  

    df_test_batch = ds.df_test.sample(1024)
    df_test_batch, _, _  = reponse(df_test_batch)
    preds = model_gan.predict(df_test_batch['content'].values, batch_size=32)
    preds_syn = model_gan.predict(df_test_batch['response'].values, batch_size=32)

    acc_all = accuracy_score(np.concatenate((df_test_batch['label'].values, df_test_batch['label'].values+num_classes)),
      np.concatenate((preds.argmax(axis=1), preds_syn.argmax(axis=1))) )

    df_batch['reward'] =rewards

    print('epoch:',epoch, acc_2class, acc_all, 'rewards==>', round(df_batch['reward'].mean(), 4))
     






