import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math,random
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
if tf.__version__.startswith("2.3"):
    assert gpus
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
parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str)
parser.add_argument("--samplecnt", default=100, type=int)
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--model", default='bert', type=str)
args = parser.parse_args()
print('args==>', args)



####### prepare data
ds = load_data(dataset=args.dsn, samplecnt=args.samplecnt, seed=random.sample(range(10000),1)[0])
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


def check_weights_no_identical(w1, w2):
    assert len(w2.trainable_weights) == len(w1.trainable_weights)
    for i in range(len(w2.trainable_weights)):
        if tf.reduce_sum(w1.trainable_weights[0]).numpy()==0 and tf.reduce_sum(w2.trainable_weights[0]).numpy()==0:
            continue 
        assert not np.array_equal(w1.trainable_weights[i], w2.trainable_weights[i])

if args.model == 'bert':
    generator = get_generator_bert()
elif args.model == 'former':
    generator = get_generator_former_()

generator_real = tf.keras.models.clone_model(generator)
generator_base = tf.keras.models.clone_model(generator)

discriminator = get_discriminator(num_classes*2)
discriminator_base = get_discriminator(num_classes)

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
model_base = keras.Model(inputs=text_input, outputs=discriminator_base(generator_base(text_input)))

if args.model == 'bert':
    lr = 1e-5
    d_optimizer = keras.optimizers.Adam(learning_rate=lr)
    g_optimizer = keras.optimizers.Adam(learning_rate=lr)
    gr_optimizer = keras.optimizers.Adam(learning_rate=lr)
    base_optimizer = keras.optimizers.Adam(learning_rate=lr)
elif args.model == 'former':
    d_optimizer = keras.optimizers.Adam()
    g_optimizer = keras.optimizers.Adam()
    gr_optimizer = keras.optimizers.Adam()
    base_optimizer = keras.optimizers.Adam()

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

    # generator_real update
    with tf.GradientTape() as tape:
        predictions = discriminator(generator_real(prompts_tensor))
        gr_loss = keras.losses.SparseCategoricalCrossentropy()(labels_tensor, predictions)
    grads = tape.gradient(gr_loss, generator_real.trainable_weights)
    gr_optimizer.apply_gradients(zip(grads, generator_real.trainable_weights))
    return d_loss, g_loss, gr_loss


@tf.function
def train_step_base(prompts, labels):
    # generator_ral update
    with tf.GradientTape() as tape:
        predictions = model_base(prompts)
        loss = keras.losses.SparseCategoricalCrossentropy()(labels, predictions)
    grads = tape.gradient(loss, model_base.trainable_weights)
    base_optimizer.apply_gradients(zip(grads, model_base.trainable_weights))
    return loss

# def loss_fn(output_sequences, labels):

#     preds = model(np.array(syn_sents_pure))

#     assert preds.shape[0] == len(prompts) and preds.shape[1] == num_classes

#     label_oht = tf.keras.utils.to_categorical( np.array([label_idx[l] for l in labels]), num_classes = num_classes, dtype='int' ) 
#     label_oht_tf = tf.convert_to_tensor(label_oht)
#     assert label_oht.shape == preds.shape

#     loss_value = cce(label_oht_tf, preds)#.numpy()
#     return loss_value
baseline_accs = []
gan_accs = []
monitoracc = []
for epoch in range(args.epoch):
    print("\nStart epoch", epoch)
    for step, trunk in enumerate(ds_train):
        prompts = trunk[0]
        labels = trunk[1]

        #print('begin to generate')
        prompts_syn = synthesize([s.decode() for s in prompts.numpy()], list(labels.numpy()), max_len, 64)
        #print('generated')
        labels_syn = labels + num_classes 

        d_loss, g_loss, gr_loss = train_step(prompts, prompts_syn,\
                            tf.cast(labels, tf.float32), tf.cast(labels_syn, tf.float32) )
        
        # baseline
        loss = train_step_base(prompts, labels)

        check_weights_no_identical(generator_base, generator_real)
        check_weights_no_identical(generator_base, generator)
        check_weights_no_identical(generator_real, generator)

        check_weights_no_identical(discriminator, discriminator_base)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
    logits = discriminator(generator(text_input))
    model_gan = keras.Model(inputs=text_input, outputs=logits)
    for x_batch_val, y_batch_val in ds_test:
        preds = model_gan(x_batch_val, training=False)  
        preds_accum =  preds[:,:num_classes] + preds[:,num_classes:]
        val_acc_metric.update_state(y_batch_val, preds_accum)
    print("gan Validation acc: %.4f" % (float(val_acc_metric.result()),))
    gan_accs.append(float(val_acc_metric.result()))
    val_acc_metric.reset_states()
    #print(d_loss.numpy(), g_loss.numpy(), gr_loss.numpy())

    for x_batch_val, y_batch_val in ds_test:
        preds = model_base(x_batch_val, training=False)
        val_acc_metric.update_state(y_batch_val, preds)
    print("baseline Validation acc: %.4f" % (float(val_acc_metric.result()),))
    #print('loss:', loss.numpy())
    baseline_accs.append(float(val_acc_metric.result()))
    val_acc_metric.reset_states()

    base_cur_best = round(max(baseline_accs),4)
    gan_cur_best = round(max(gan_accs),4)
    gain = round( (gan_cur_best-base_cur_best) / base_cur_best, 4) 
    print("summary==>", "dsn:", args.dsn, "samplecnt:", args.samplecnt, 'epoch:',epoch,\
      'base:', base_cur_best, 'gan:', gan_cur_best, 'gain:',  gain  )
    monitoracc.append( gain )

    if len(monitoracc) >= 20 and len(set(monitoracc[-10:])) ==1:
        print('summary==> terminated')
        break 








