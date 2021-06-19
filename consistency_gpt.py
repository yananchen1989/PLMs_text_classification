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
parser.add_argument("--model", default='former', type=str)
parser.add_argument("--beams", default=8, type=int)
args = parser.parse_args()
print('args==>', args)

if args.model=='bert':
    assert gpus
elif args.model == 'former':
    assert tf.__version__.startswith('2.5')

####### prepare data
ds = load_data(dataset=args.dsn, samplecnt=args.samplecnt, seed=random.sample(range(10000),1)[0])
label_unique = ds.df_test.label.unique()
label_ix = {label_unique[i]:i for i in range(label_unique.shape[0])}
ix_label = {i:label_unique[i] for i in range(label_unique.shape[0])}
ds.df_train['label'] = ds.df_train['label'].map(lambda x: label_ix[x])
ds.df_test['label'] = ds.df_test['label'].map(lambda x: label_ix[x])

batch_size=64
max_len = get_tokens_len(ds, 0.99) 
num_classes = label_unique.shape[0]

ds_train = tf.data.Dataset.from_tensor_slices((ds.df_train['content'].values, ds.df_train['label'].values))
ds_train = ds_train.shuffle(buffer_size=12800).batch(batch_size)

ds_test = tf.data.Dataset.from_tensor_slices((ds.df_test['content'].values, ds.df_test['label'].values))
ds_test = ds_test.batch(batch_size)

# def parser(x):
#     inputs = tokenizer([ii.decode() for ii in xx], padding='max_length', add_prefix_space=True, truncation=True, max_length=max_len, return_tensors="tf")
#     return inputs

# for mm in dstf.map(lambda x, y: (x, y) ).take(5):
#     print(mm)
#     print(sent)
#     print(label)
#     break 

if args.model == 'bert':
    lr = 1e-5
    cs_optimizer = keras.optimizers.Adam(learning_rate=lr)
    base_optimizer = keras.optimizers.Adam(learning_rate=lr)
else:
    cs_optimizer = keras.optimizers.Adam()
    base_optimizer = keras.optimizers.Adam()

if args.model == 'bert':
    model_base = get_model_bert_(num_classes, 'albert')
    model_cs = get_model_bert_(num_classes*2, 'albert')
elif args.model == 'former':
    model_base = get_model_transormer_(num_classes)
    model_cs = get_model_transormer_(num_classes*2)


@tf.function
def train_step(prompts_tensor, prompts_syn_tensor, labels_tensor, labels_syn_tensor):

    prompts_comb = tf.concat([prompts_tensor, prompts_syn_tensor], axis=0)
    labels_comb = tf.concat([labels_tensor, labels_syn_tensor], axis=0)

    with tf.GradientTape() as tape:
        predictions = model_cs(prompts_comb)
        loss = keras.losses.SparseCategoricalCrossentropy()(labels_comb, predictions)
    grads = tape.gradient(loss, model_cs.trainable_weights)
    cs_optimizer.apply_gradients(zip(grads, model_cs.trainable_weights))
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

# nlp_nli = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli', device=-1)
# def synthesize_nli(prompts, labels, max_len):
#     inputs = tokenizer(prompts, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
#     inputs.to(device)
#     output_sequences = gpt2.generate(
#         input_ids = inputs['input_ids'],
#         attention_mask = inputs['attention_mask'] ,
#         max_length= max_len*2,
#         temperature=1,
#         top_k=0,
#         top_p=0.9,
#         repetition_penalty=1,
#         do_sample=True,
#         num_return_sequences=16
#     )
#     syn_sents = tokenizer.batch_decode(output_sequences, clean_up_tokenization_spaces=True, skip_special_tokens=True)
#     syn_sents_pure = []
#     for sent, label, sent_syn in zip(prompts, labels, syn_sents):
#         sent_syn_rm = sent_syn.replace(sent, '').replace('\n',' ').strip()
#         sent_syn_eq = sent_syn_rm[:len(sent)]

#         result_nli = nlp_nli(sent_syn_eq, list(label_ix.keys()), multi_label=False, hypothesis_template="This text is about {}.")
#         if result_nli['scores'][0] >= args.thres and result_nli['labels'][0] == ix_label[label]:
#         syn_sents_pure.append(sent_syn_eq)
#     return tf.convert_to_tensor(np.array(syn_sents_pure))
def synthesize_beams(prompts, labels):
    prompts_syn_ll = []
    labels_syn_ll = []

    for _ in range(args.beams):
        prompts_syn = synthesize([s.decode() for s in prompts.numpy()], list(labels.numpy()), max_len)
        labels_syn = labels + num_classes 
        prompts_syn_ll.append(prompts_syn)
        labels_syn_ll.append(labels_syn)

    prompts_syn_beams = tf.concat(prompts_syn_ll, axis=0)
    labels_syn_beams = tf.concat(labels_syn_ll, axis=0)

    assert prompts_syn_beams.shape[0] == args.beams*prompts.shape[0] and prompts_syn_beams.shape[0]==labels_syn_beams.shape[0]
    return prompts_syn_beams, labels_syn_beams



baseline_accs = []
gan_accs = []
monitoracc = []
for epoch in range(args.epoch):
    print("\nStart epoch", epoch)
    for step, trunk in enumerate(ds_train):
        prompts = trunk[0]
        labels = trunk[1] 

        prompts_syn_beams, labels_syn_beams = synthesize_beams(prompts, labels)

        loss_cs = train_step(prompts, prompts_syn_beams, labels, labels_syn_beams)
        # baseline
        loss = train_step_base(prompts, labels)

        #print(loss.numpy(), loss_cs.numpy())

    for x_batch_val, y_batch_val in ds_test:
        preds = model_cs(x_batch_val, training=False)  
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
        print('summary==> terminated ', max(monitoracc), args)
        break 








