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
parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str)
parser.add_argument("--samplecnt", default=128, type=int)
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--batch_size", default=32, type=int)
#parser.add_argument("--model", default='bert', type=str)
#parser.add_argument("--syn", default='gpt', type=str, choices=['gpt', 'raw'])
#parser.add_argument("--unify", default=0, type=int, choices=[0,1,2])
args = parser.parse_args()
print('args==>', args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu )
gpus = tf.config.experimental.list_physical_devices('GPU')
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
from utils.load_data import * 
from utils.transblock import * 
from utils.gan_config import * 

assert gpus


# if args.model=='bert':
#     assert gpus
# elif args.model == 'former':
#     assert tf.__version__.startswith('2.5')


# def parser(x):
#     inputs = tokenizer([ii.decode() for ii in xx], padding='max_length', add_prefix_space=True, truncation=True, max_length=max_len, return_tensors="tf")
#     return inputs

# for mm in dstf.map(lambda x, y: (x, y) ).take(5):
#     print(mm)
#     print(sent)
#     print(label)
#     break 

val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(prompts_tensor, prompts_syn_tensor, labels_tensor, labels_syn_tensor):

    generated_images = generator_fake(prompts_syn_tensor )
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

    # generator_fake update
    with tf.GradientTape() as tape:
        predictions = discriminator(generator_fake(prompts_syn_tensor))
        g_loss = keras.losses.SparseCategoricalCrossentropy()(labels_tensor, predictions)
    grads = tape.gradient(g_loss, generator_fake.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, generator_fake.trainable_weights))

    # generator_real update
    with tf.GradientTape() as tape:
        predictions = discriminator(generator_real(prompts_tensor))
        gr_loss = keras.losses.SparseCategoricalCrossentropy()(labels_tensor, predictions)
    grads = tape.gradient(gr_loss, generator_real.trainable_weights)
    gr_optimizer.apply_gradients(zip(grads, generator_real.trainable_weights))
    
    return d_loss, g_loss, gr_loss

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



@tf.function
def train_step_base(prompts, labels):
    # generator_ral update
    with tf.GradientTape() as tape:
        predictions = model_base(prompts)
        loss = keras.losses.SparseCategoricalCrossentropy()(labels, predictions)
    grads = tape.gradient(loss, model_base.trainable_weights)
    base_optimizer.apply_gradients(zip(grads, model_base.trainable_weights))
    return loss

####### prepare data
ds = load_data(dataset=args.dsn, samplecnt=args.samplecnt)
ds, num_classes, max_len = process_ds(ds)

if args.samplecnt <= 32:
    args.batch_size = 8

ds_train = tf.data.Dataset.from_tensor_slices((ds.df_train['content'].values, ds.df_train['label'].values))
ds_train = ds_train.shuffle(buffer_size=12800).batch(args.batch_size)

ds_test = tf.data.Dataset.from_tensor_slices((ds.df_test['content'].values, ds.df_test['label'].values))
ds_test = ds_test.batch(args.batch_size)

generator_fake = get_generator_bert()
generator_real = tf.keras.models.clone_model(generator_fake)
generator_base = tf.keras.models.clone_model(generator_fake)

discriminator = get_discriminator(num_classes*2)
discriminator_base = get_discriminator(num_classes)

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
model_base = keras.Model(inputs=text_input, outputs=discriminator_base(generator_base(text_input)))

model_gan = keras.Model(inputs=text_input, outputs=discriminator(generator_real(text_input)))

#if args.model == 'bert':
lr = 4e-5
d_optimizer = keras.optimizers.Adam(learning_rate=lr)
g_optimizer = keras.optimizers.Adam(learning_rate=lr)
gr_optimizer = keras.optimizers.Adam(learning_rate=lr)
gan_optimizer = keras.optimizers.Adam(learning_rate=lr)
base_optimizer = keras.optimizers.Adam(learning_rate=lr)
# else:
#     d_optimizer = keras.optimizers.Adam()
#     g_optimizer = keras.optimizers.Adam()
#     gr_optimizer = keras.optimizers.Adam()
#     uni_optimizer = keras.optimizers.Adam()
#     base_optimizer = keras.optimizers.Adam()
# from aug_fillinmask import *
# augmentor = fillInmask()

# def get_sents_fake(ds_):
#     df_batch = ds_.df_train.sample(32)
#     df_batch['content_fake'] = df_batch['content'].map(lambda x: augmentor.augment(x))
#     sents_syn = tf.convert_to_tensor(df_batch['content_fake'].values)
#     sents_syn_label = tf.convert_to_tensor([0.0]*32)

#     sents_real = tf.convert_to_tensor(df_batch['content'].values)
#     sents_real_label = tf.convert_to_tensor([1.0]*32)
#     return sents_syn, sents_real, sents_syn_label, sents_real_label

baseline_accs = []
gan_accs = []
monitoracc = []
for epoch in range(args.epoch):
    print("\nStart epoch", epoch)
    for step, trunk in enumerate(ds_train):
        prompts = trunk[0]
        labels = trunk[1] 

        #print('begin to generate')
        
        prompts_syn = gpt2_nlp([s.decode() for s in prompts.numpy()], max_length=max_len, do_sample=True, top_p=0.9, top_k=0, \
                    repetition_penalty=1.0, num_return_sequences=1, clean_up_tokenization_spaces=True)

        prompts_syn = tf.convert_to_tensor(np.array([ii[0]['generated_text'].replace('\n',' ').strip() for ii in prompts_syn]))
        #prompts_syn = synthesize_for_gan(, max_len, gpt2)

        labels_syn = labels + num_classes 

        d_loss, g_loss, gr_loss = train_step(prompts, prompts_syn,  tf.cast(labels, tf.float32), tf.cast(labels_syn, tf.float32) )
             
        # if args.unify == 1:
        #     loss_gan = train_step_gan(prompts, prompts_syn,  \
        #                tf.cast(labels, tf.float32), tf.cast(labels_syn, tf.float32))
        # else:
        #     pass 

        # baseline
        loss = train_step_base(prompts, labels)

        check_weights_no_identical(generator_base, generator_real)
        check_weights_no_identical(generator_base, generator_fake)
        check_weights_no_identical(generator_real, generator_fake)
        check_weights_no_identical(discriminator, discriminator_base)
        #print("loss:", loss)
    if epoch < 10 or epoch % 10 != 0:
        continue
    # gan validate
    for x_batch_val, y_batch_val in ds_test:
        preds = model_gan(x_batch_val, training=False)  
        preds_accum =  preds[:,:num_classes] + preds[:,num_classes:]
        val_acc_metric.update_state(y_batch_val, preds_accum)
    print("gan Validation acc: %.4f" % (float(val_acc_metric.result()),))
    gan_accs.append(float(val_acc_metric.result()))
    val_acc_metric.reset_states()
    #print(d_loss.numpy(), g_loss.numpy(), gr_loss.numpy())

    # baseline validate
    for x_batch_val, y_batch_val in ds_test:
        preds = model_base(x_batch_val, training=False)
        val_acc_metric.update_state(y_batch_val, preds)
    print("baseline Validation acc: %.4f" % (float(val_acc_metric.result()),))
    #print('loss:', loss.numpy())
    baseline_accs.append(float(val_acc_metric.result()))
    val_acc_metric.reset_states()

    # summary
    base_cur_best = round(max(baseline_accs),4)
    gan_cur_best = round(max(gan_accs),4)
    #gain = round( (gan_cur_best-base_cur_best) / base_cur_best, 4) 
    cur = datetime.datetime.strftime(datetime.datetime.now() + datetime.timedelta(hours=8), '%Y-%m-%d %H:%M:%S')
    print(cur, "epoch==>", "dsn:", args.dsn, "samplecnt:", args.samplecnt, 'epoch:',epoch,\
      'base:', base_cur_best, 'gan:', gan_cur_best )
    #monitoracc.append( gain )
    # if (len(monitoracc) >= 20 and len(set(monitoracc[-5:])) ==1) or \
    #    (len(monitoracc) >= 50 and monitoracc[-1]<=0 )  :
    #     print('epochs terminated ', monitoracc[-1] )
    #     break
         
record_log('loggan', \
                 ['summary==>'] + ['{}:{}'.format(k, v) for k, v in vars(args).items()] \
                 + ['baseline_acc:{}'.format(base_cur_best), 'gan_acc:{}'.format(gan_cur_best) ]
           )





