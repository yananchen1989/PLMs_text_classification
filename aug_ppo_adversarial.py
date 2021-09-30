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
parser.add_argument("--epoch", default=12, type=int)
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
parser.add_argument("--ft_pattern", default='pp', type=str, choices=['pp', 'tc', 'no'])
parser.add_argument("--ppo_train_epoch", default=1, type=int)


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
from utils.ppo_config import * 
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


# row = df_batch.sample(1)
# print(row['label_name'])
# print(row['content'].tolist()[0])
# print(row['response'].tolist()[0])

# model_gan.predict(row['content'].values, batch_size=32)
if args.dsn == 'uci':
    maxlen = 32 
elif args.dsn == 'ag':
    maxlen = 128

for epoch in range(args.epoch):
    ds.df_train = ds.df_train.sample(frac=1)
    ix = 0
    while ix < ds.df_train.shape[0]:

        df_batch = ds.df_train[ix:ix+args.ppo_batchsize].copy()

        torch.cuda.empty_cache()   

        df_batch, query_tensors, response_tensors = reponse_(df_batch, gpt2_model_trl, maxlen)

        prompts = tf.convert_to_tensor(df_batch['content'].values)
        labels = tf.convert_to_tensor(df_batch['label'].values)

        prompts_syn = tf.convert_to_tensor(df_batch['response'].values)
        labels_syn = tf.convert_to_tensor(df_batch['label'].values+ num_classes )  

        preds_syn = model_gan.predict(prompts_syn, batch_size=32) 
        preds_ori = model_gan.predict(prompts, batch_size=32) 

        preds_ori_labels = preds_ori.argmax(axis=1)
        preds_syn_labels = preds_syn.argmax(axis=1)

        acc_all = accuracy_score(np.concatenate((df_batch['label'].values, df_batch['label'].values+num_classes)),
                    np.concatenate((preds_ori_labels, preds_syn_labels)) )
        acc_half = accuracy_score(df_batch['label'].values, \
                      (preds_ori[:,:num_classes] + preds_ori[:,num_classes:]).argmax(axis=1))  

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
        if epoch >= args.ppo_train_epoch :          
            stats = ppo_trainer.step(query_tensors, response_tensors, torch.tensor(rewards).to(device))    

        # loss_gan = train_step_gan(prompts, prompts_syn,  \
        #                     tf.cast(labels, tf.float32), tf.cast(labels_syn, tf.float32))

        loss_gan = train_step_gan(prompts, prompts_syn, labels, labels_syn)

        print(ix, 'of', args.samplecnt*num_classes, 'epoch:', epoch, \
               'acc_half:', acc_half, 'acc_all:', acc_all,  \
              'loss:', loss_gan.numpy(), 'rewards:', np.array(rewards).mean() )
        ix += args.ppo_batchsize


    preds = model_gan.predict(ds.df_test['content'].values, batch_size=32)  
    preds_accum =  preds[:,:num_classes] + preds[:,num_classes:]
    acc_half = accuracy_score(ds.df_test['label'].values, preds_accum.argmax(axis=1))  

    df_test_batch = ds.df_test.sample(256)
    df_test_batch, _, _  = reponse_(df_test_batch, gpt2_model_trl, maxlen)
    preds = model_gan.predict(df_test_batch['content'].values, batch_size=32)
    preds_syn = model_gan.predict(df_test_batch['response'].values, batch_size=32)

    acc_all = accuracy_score(np.concatenate((df_test_batch['label'].values, df_test_batch['label'].values+num_classes)),
      np.concatenate((preds.argmax(axis=1), preds_syn.argmax(axis=1))) )

    df_batch['reward'] =rewards

    print('summary epoch:',epoch, acc_half, acc_all, 'rewards==>', round(df_batch['reward'].mean(), 4))
     




