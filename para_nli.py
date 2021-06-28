import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math,random, time
import numpy as np
import tensorflow as tf
import pandas as pd 
import datasets
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split
from tensorflow import keras
#from transformers import pipeline
gpus = tf.config.experimental.list_physical_devices('GPU')
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'  
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type=='cuda'
from load_data import * 
from transblock import * 


assert gpus

# cc_news = datasets.load_dataset('cc_news', split="train")
# dfcc = pd.DataFrame(cc_news['text'], columns=['content'])
# dfcc.to_csv('cc_news.csv', index=False)

dfcc = pd.read_csv("cc_news.csv")
text_input1 = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string
text_input2 = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string

encoder = hub.KerasLayer('albert_en_base_2', trainable=False)
embed = []
for textin in [text_input1, text_input2]:
    encoder_inputs = preprocessor_layer(textin)
    outputs = encoder(encoder_inputs)
    embed.append(outputs["pooled_output"])
embed_all = tf.concat(embed, axis=1)
x = layers.Dense(512, activation="relu")(embed_all)
out = layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=[text_input1, text_input2], outputs=out)

@tf.function
def train_step_base(prompts, labels):
    # generator_ral update
    with tf.GradientTape() as tape:
        predictions = model(prompts)
        loss = keras.losses.BinaryCrossentropy(from_logits=True)(labels, predictions)
    grads = tape.gradient(loss, model.trainable_weights)
    base_optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")



lr = 1e-5
base_optimizer = keras.optimizers.Adam(learning_rate=lr)


df_train, df_test = train_test_split(dfcc, test_size=0.05)

ds_train = tf.data.Dataset.from_tensor_slices(df_train['content'].values)
ds_train = ds_train.shuffle(buffer_size=12800).batch(64)

ds_test = tf.data.Dataset.from_tensor_slices(df_test['content'].values)
ds_test = ds_test.batch(64)

def get_pairs(trunk):
    pos_pairs = []
    neg_pairs = []
    ll = []
    for para in trunk.numpy():
        elements = []
        for sent in para.decode().replace('. ','\n').split('\n'):
            if len(tokenizer.tokenize(sent)) <= 20:
                continue
            elements.append(sent)
        if len(elements) <= 2:
            continue 
        ll.append(elements)

    if len(ll) <= 4:
        return []

    for ii in ll:
        pos_pairs.append(tuple(random.sample(ii, 2)))
        rnd_paras = random.sample(ll, 2)
        neg_pairs.append((random.sample(rnd_paras[0], 1)[0], random.sample(rnd_paras[1], 1)[0]))

    df_trunk = pd.DataFrame(pos_pairs+neg_pairs, columns=['sent1','sent2'])
    df_trunk['label'] = [1] * len(pos_pairs)+ [0] * len(pos_pairs)
    if df_trunk.shape[0] > 64:
        df_trunk = df_trunk.sample(64)
    return df_trunk



for trainable in [False, True]:
    encoder.trainable = trainable
    if not trainable:
        epochs = 7 
    else:
        epochs = 100
    for epoch in range(epochs):
        print("\nStart epoch", epoch)
        for step, trunk in enumerate(ds_train):
            df_trunk = get_pairs(trunk)
            if len(df_trunk) == 0:
                print("not enough ll")
                continue
            loss = train_step_base([df_trunk['sent1'].values, df_trunk['sent2'].values], df_trunk['label'].values)        
            if step % 100 == 0:
                print(loss.numpy())

        results_pred = []
        results_label = []
        for step, trunk in enumerate(ds_test):
            df_trunk = get_pairs(trunk)
            preds = model([df_trunk['sent1'].values, df_trunk['sent2'].values], training=False)  
            results_pred.append(tf.reshape(preds, -1))
            results_label.append(tf.convert_to_tensor(df_trunk['label'].values))
        pred_val = tf.concat(results_pred, axis=0)
        label_val = tf.concat(results_label, axis=0)
        val_loss = keras.losses.BinaryCrossentropy(from_logits=True)(label_val, pred_val)
        print('epoch:', epoch, 'val loss:', val_loss.numpy())





