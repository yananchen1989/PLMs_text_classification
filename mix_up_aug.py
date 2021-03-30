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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
logging.Formatter.converter = beijing


parser = argparse.ArgumentParser()
parser.add_argument("--ds", "-ds", default="", type=str)
parser.add_argument("--samplecnt", "-samplecnt", default=1000, type=int)
args = parser.parse_args()


logging.basicConfig(
    filename='log_mixup_{}'.format(args.ds),
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO, filemode='w'
)

logger = logging.getLogger()

from load_data import * 



def get_keras_data_mixup(ds):
    labels = ds.df_test['label'].unique().tolist()
    label_idx = {l:ix for ix, l in enumerate(labels)}
    num_classes = len(label_idx)

    # train
    #ds.df_train['key'] = 0
    #del ds.df_train['title']
    df_train_pair = pd.merge(ds.df_train, ds.df_train, on='label')
    #del df_train_pair['key']

    df_train_pair['content_x_y'] = list(zip(df_train_pair.content_x, df_train_pair.content_y))

    df_train_pair['content_x_y_sort'] = df_train_pair['content_x_y'].map(lambda x: ' '.join(sorted(list(x))))

    df_train_pair.drop_duplicates(subset=['content_x_y_sort'], inplace=True)
    del  df_train_pair['content_x_y_sort'], df_train_pair['content_x_y']

    df_train_pair = df_train_pair.sample(frac=1)

    train_y = tf.keras.utils.to_categorical(\
                             df_train_pair['label'].map(lambda x: label_idx.get(x)).values, \
                             num_classes = num_classes, dtype='int' )
    # label_y_oht = tf.keras.utils.to_categorical(\
    #                          df_train_pair['label_y'].map(lambda x: label_idx.get(x)).values, \
    #                          num_classes = num_classes, dtype='int' )
    # label_xy =label_x_oht + label_y_oht
    # label_xy_clip = np.clip(label_xy,0,1)

    train_x0 = df_train_pair['content_x'].values.reshape(-1,1) 
    train_x1 = df_train_pair['content_y'].values.reshape(-1,1) 

    # test 
    test_x = ds.df_test['content']
    test_y = tf.keras.utils.to_categorical(\
                         ds.df_test['label'].map(lambda x: label_idx.get(x)).values, \
                         num_classes = num_classes, dtype='int' )  
    return ((train_x0, train_x1), train_y), ((test_x,test_x), test_y), num_classes


def get_model(num_classes):
    # https://tfhub.dev/tensorflow/albert_en_base/2
    encoder = hub.KerasLayer('./albert_en_base_2', trainable=True)
    # https://tfhub.dev/tensorflow/albert_en_preprocess/3
    preprocessor = hub.KerasLayer("./albert_en_preprocess_3")

    text_input_x = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string
    text_input_y = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string

    pooled_output_xy = []
    for text_input in [text_input_x, text_input_y]:
        encoder_inputs = preprocessor(text_input)
        outputs = encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]   # (None, 768)
        sequence_output = outputs["sequence_output"] # (None, 128, 768)
        #pooled_output_ = tf.keras.layers.Dense(256, activation="relu")(pooled_output)
        pooled_output_xy.append(pooled_output*0.5)

    pooled_output_xy_mixup = tf.keras.layers.add(pooled_output_xy)
    out = layers.Dense(num_classes, activation="softmax")(pooled_output_xy_mixup)
    model = tf.keras.Model(inputs=[text_input_x, text_input_y], outputs=out)
    model.compile(Adam(lr=1e-5), "categorical_crossentropy", metrics=["acc"])
    return model


for i in range(5):
    ds = load_data(dataset=args.ds, samplecnt=args.samplecnt)
    ((train_x0, train_x1), train_y), ((test_x,test_x), test_y), num_classes = get_keras_data_mixup(ds)
    model = get_model(num_classes)
    history = model.fit(
                (train_x0, train_x1), train_y, batch_size=32, epochs=12, validation_data=((test_x,test_x), test_y),\
                verbose=1,
                callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
            )
    best_val_acc = max(history.history['val_acc'])
    print("best_val_acc==>", best_val_acc)
    logger.info("iter:{} acc:{}".format(i, best_val_acc))








