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
from transformers import pipeline
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
from gan_config import * 

assert gpus

cc_news = datasets.load_dataset('cc_news', split="train")
dfcc = pd.DataFrame(cc_news['text'], columns=['content'])

model = get_model_bert_pair()

@tf.function
def train_step_base(prompts, labels):
    # generator_ral update
    with tf.GradientTape() as tape:
        predictions = model(prompts)
        loss = keras.losses.BinaryCrossentropy()(labels, predictions)
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
        return tf.convert_to_tensor([]), tf.convert_to_tensor([])

    for ii in ll:
        pos_pairs.append(tuple(random.sample(ii, 2)))
        rnd_paras = random.sample(ll, 2)
        neg_pairs.append((random.sample(rnd_paras[0], 1)[0], random.sample(rnd_paras[1], 1)[0]))

    return tf.convert_to_tensor(pos_pairs), tf.convert_to_tensor(neg_pairs)


val_acc_metric = tf.keras.metrics.BinaryAccuracy()


for epoch in range(100):
    print("\nStart epoch", epoch)
    for step, trunk in enumerate(ds_train):
        pos_pairs_tf, neg_pairs_tf = get_pairs(trunk)
        if pos_pairs_tf.shape[0] == 0:
            print("not enough ll")
            continue
        labels_pos = tf.convert_to_tensor([1.0]*pos_pairs_tf.shape[0])
        labels_neg = tf.convert_to_tensor([0.0]*neg_pairs_tf.shape[0])
        combined_images = tf.concat([pos_pairs_tf, neg_pairs_tf], axis=0)
        combined_labels = tf.concat([labels_pos, labels_neg], axis=0)
        loss = train_step_base(combined_images, combined_labels)
        print(loss)

    for x_batch_val, y_batch_val in ds_test:
        preds = model(x_batch_val, training=False)  
        val_acc_metric.update_state(y_batch_val, preds)
    print("gan Validation acc: %.4f" % (float(val_acc_metric.result()),))
    val_acc_metric.reset_states()





