import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math
import numpy as np
import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
from transformers import pipeline
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

from load_data import * 
from transblock import * 
from gan_config import * 
parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str)
parser.add_argument("--samplecnt", default=100, type=int)
args = parser.parse_args()
print('args==>', args)

ds = load_data(dataset=args.dsn, samplecnt=args.samplecnt)
label_unique = ds.df_test.label.unique()
label_ix = {label_unique[i]:i for i in range(label_unique.shape[0])}
ix_label = {i:label_unique[i] for i in range(label_unique.shape[0])}
ds.df_train['label'] = ds.df_train['label'].map(lambda x: label_ix[x])
ds.df_test['label'] = ds.df_test['label'].map(lambda x: label_ix[x])


max_len = get_tokens_len(ds, 0.99) 
num_classes = label_unique.shape[0]

ds_train = tf.data.Dataset.from_tensor_slices((ds.df_train['content'].values, ds.df_train['label'].values))
ds_train = ds_train.shuffle(buffer_size=12800).batch(64)

ds_test = tf.data.Dataset.from_tensor_slices((ds.df_test['content'].values, ds.df_test['label'].values))
ds_test = ds_test.batch(64)

# def parser(x):
#     inputs = tokenizer([ii.decode() for ii in xx], padding='max_length', add_prefix_space=True, truncation=True, max_length=max_len, return_tensors="tf")
#     return inputs

# for mm in dstf.map(lambda x, y: (x, y) ).take(5):
#     print(mm)
#     print(sent)
#     print(label)
#     break 

generator = get_generator_bert()
#generator = get_generator_former()
generator_real = tf.keras.models.clone_model(generator)

discriminator = get_discriminator(num_classes*2)

d_optimizer = keras.optimizers.Adam(learning_rate=1e-5)
g_optimizer = keras.optimizers.Adam(learning_rate=1e-5)
gr_optimizer = keras.optimizers.Adam(learning_rate=1e-5)



@tf.function
def train_step(prompts_tensor, prompts_syn_tensor, labels_tensor, labels_syn_tensor):

    generated_images = generator(prompts_syn_tensor )
    real_images = generator_real(prompts_tensor)

    labels_tensor += 0.05 * tf.random.uniform(labels_tensor.shape)
    labels_syn_tensor += 0.05 * tf.random.uniform(labels_syn_tensor.shape)

    combined_images = tf.concat([generated_images, real_images], axis=0)
    combined_labels = tf.concat([labels_syn_tensor, labels_tensor], axis=0)
    # discriminator update 
    with tf.GradientTape() as tape:
        predictions = discriminator(combined_images)
        d_loss = keras.losses.SparseCategoricalCrossentropy()(combined_labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

    # generator update
    with tf.GradientTape() as tape:
        predictions = discriminator(generator(prompts_syn_tensor))
        g_loss = keras.losses.SparseCategoricalCrossentropy()(labels_tensor, predictions)
    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

    # generator_ral update
    with tf.GradientTape() as tape:
        predictions = discriminator(generator_real(prompts_tensor))
        gr_loss = keras.losses.SparseCategoricalCrossentropy()(labels_tensor, predictions)
    grads = tape.gradient(gr_loss, generator_real.trainable_weights)
    gr_optimizer.apply_gradients(zip(grads, generator_real.trainable_weights))
    return d_loss, g_loss, gr_loss

m_ = tf.keras.metrics.SparseCategoricalAccuracy()


# def loss_fn(output_sequences, labels):

#     preds = model(np.array(syn_sents_pure))

#     assert preds.shape[0] == len(prompts) and preds.shape[1] == num_classes

#     label_oht = tf.keras.utils.to_categorical( np.array([label_idx[l] for l in labels]), num_classes = num_classes, dtype='int' ) 
#     label_oht_tf = tf.convert_to_tensor(label_oht)
#     assert label_oht.shape == preds.shape

#     loss_value = cce(label_oht_tf, preds)#.numpy()
#     return loss_value
accs = []
for epoch in range(100):
    print("\nStart epoch", epoch)
    for step, trunk in enumerate(ds_train):
        sents = trunk[0].numpy()
        labels = trunk[1].numpy()
        prompts = [s.decode() for s in sents]

        prompts_syn = synthesize(prompts, list(labels))
        labels_syn = [i+num_classes for i in labels]

        prompts_tensor = tf.convert_to_tensor(np.array(prompts))
        prompts_syn_tensor = tf.convert_to_tensor(np.array(prompts_syn))

        labels_tensor = tf.convert_to_tensor(np.array(labels), dtype=tf.float32)
        labels_syn_tensor = tf.convert_to_tensor(np.array(labels_syn), dtype=tf.float32) 
        d_loss, g_loss, gr_loss = train_step(prompts_tensor, prompts_syn_tensor, labels_tensor, labels_syn_tensor)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
    logits = discriminator(generator(text_input))
    model_ = keras.Model(inputs=text_input, outputs=logits)

    preds_ = model_.predict(ds.df_test['content'].values, batch_size=256, verbose=0)

    preds_uni_ = preds_[:,:num_classes] + preds_[:,num_classes:]

    m_.update_state(ds.df_test['label'].values, preds_uni_)
    print('generator test acc:', m_.result().numpy())
    accs.append(m_.result().numpy())
    print(d_loss.numpy(), g_loss.numpy(), gr_loss.numpy())
    m_.reset_states()
    if len(accs) >=7 and accs[-1] <= accs[-3] and accs[-2] <= accs[-3]:
        print('best test acc:', max(accs))
        break 






############## baseline


text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 


generator = get_generator_bert()
discriminator = get_discriminator(num_classes)
logits = discriminator(generator(text_input))
model_base = keras.Model(inputs=text_input, outputs=logits)


@tf.function
def train_step_base(prompts_tensor, labels_tensor):

    # generator_ral update
    with tf.GradientTape() as tape:
        predictions = model_base(prompts_tensor)
        loss = keras.losses.SparseCategoricalCrossentropy()(labels_tensor, predictions)
    grads = tape.gradient(loss, generator_real.trainable_weights)
    gr_optimizer.apply_gradients(zip(grads, generator_real.trainable_weights))
    return loss

















