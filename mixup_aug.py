import sys,os,logging,glob,pickle,torch
import numpy as np
import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

import datetime,argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", "-dsn", default="", type=str)
parser.add_argument("--alpha", "-alpha", default=0.5, type=float)
args = parser.parse_args()

from load_data import * 

ds = load_data(dataset=args.dsn, samplecnt=-1)
num_classes = ds.df_test['label'].unique().shape[0]
batch_size = 64


def generator(ds, batch_size, alpha_seed):
    epoch = 0
    while 1:
        print('epoch:', epoch)
        dfx = ds.df_train.sample(frac=1)
        dfy = ds.df_train.sample(frac=1)
        alpha = np.random.beta(alpha_seed, alpha_seed, ds.df_train.shape[0])
        oht_x = tf.keras.utils.to_categorical(dfx['label']-1, num_classes=num_classes, dtype='int' )
        oht_y = tf.keras.utils.to_categorical(dfy['label']-1, num_classes=num_classes, dtype='int' )
        label = np.multiply(oht_x.T, alpha).T + np.multiply(oht_y.T, (1-alpha)).T
        for ix in range(ds.df_train.shape[0]):
            eni = ix + batch_size
            if eni > ds.df_train.shape[0]:
                epoch += 1
                break 
            yield (dfx[ix:eni]['content'].values, dfy[ix:eni]['content'].values, alpha[ix:eni]), label[ix:eni]

test_x = (ds.df_test['content'].values, ds.df_test['content'].values, np.ones(ds.df_test.shape[0]))
test_y = tf.keras.utils.to_categorical(ds.df_test['label']-1, num_classes=num_classes, dtype='int' )
       


def get_model(num_classes):
    # https://tfhub.dev/tensorflow/albert_en_base/2
    encoder = hub.KerasLayer('./albert_en_base_2', trainable=True)
    # https://tfhub.dev/tensorflow/albert_en_preprocess/3
    preprocessor = hub.KerasLayer("./albert_en_preprocess_3")

    text_input_x = tf.keras.layers.Input(shape=(), dtype=tf.string, name='sentx') # shape=(None,) dtype=string
    text_input_y = tf.keras.layers.Input(shape=(), dtype=tf.string, name='senty') # shape=(None,) dtype=string
    alpha = tf.keras.layers.Input(shape=(1,), name='alpha')
    ids_y = preprocessor(text_input_y)
    outputs_y = encoder(ids_y)
    ids_x = preprocessor(text_input_x)
    outputs_x = encoder(ids_x)
    outputsl = [outputs_x["pooled_output"]*alpha,  outputs_y["pooled_output"]*(1-alpha) ]
    pooled_output_xy_mixup = tf.keras.layers.add(outputsl)
    out = layers.Dense(num_classes, activation="softmax")(pooled_output_xy_mixup)
    model = tf.keras.Model(inputs=[text_input_x, text_input_y, alpha], outputs=out)
    model.compile(Adam(lr=1e-5), "categorical_crossentropy", metrics=["acc"])
    return model

for args.alpha in [0.2, 0.5, 0.7]:
    model = get_model(num_classes)
    history = model.fit(
                generator(ds,batch_size=batch_size,alpha_seed=args.alpha),  epochs=64, validation_data=(test_x, test_y),\
                verbose=1, steps_per_epoch= ds.df_train.shape[0] // batch_size, 
                callbacks = [EarlyStopping(monitor='val_acc', patience=4, mode='max')]
            ) 
    print("ds:{} alpha:{} acc:{}".format(args.dsn, args.alpha, max(history.history['val_acc'])))
























