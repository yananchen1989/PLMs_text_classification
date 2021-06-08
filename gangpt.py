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
ds = load_data(dataset='ag', samplecnt=-1, seed=45)

(x_train, y_train),  (x_test, y_test), num_classes, label_idx = get_keras_data(ds.df_train, ds.df_test)

model = get_model_transormer(num_classes)
model.fit(
        x_train, y_train, batch_size=64, epochs=50, \
        validation_batch_size=64,
        validation_data=(x_test, y_test), verbose=1,
        callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
    )


# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_dataset = train_dataset.shuffle(buffer_size=10000).batch(64)

# for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#     break 






# generator
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token # to avoid an error

gpt2 = TFGPT2LMHeadModel.from_pretrained('gpt2')
gpt2.trainable = True

#model = TFAutoModelForCausalLM.from_pretrained("gpt2")
#model = TFGPT2LMHeadModel.from_pretrained('gpt2')

#model.train()
# when generating, we will use the logits of right-most token to predict the next token
# so the padding should be on the left

rows = ds.df_test.sample(5)
prompts = rows['content'].tolist()
labels = rows['label'].tolist()

num_return_sequences = 1
#prompts = list(x_batch_train.numpy().reshape(-1))

#token_lens = [len(tokenizer.tokenize(sent)) for sent in prompts]
#max_length = math.ceil(np.array(token_lens).max())*2
max_len = get_tokens_len(ds, 0.99) 



cce = tf.keras.losses.CategoricalCrossentropy()   
optimizer = keras.optimizers.Adam(learning_rate=0.0001)


def loss_fn(output_sequences, labels):
    syn_sents = tokenizer.batch_decode(output_sequences, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    syn_sents_pure = []
    for sent, sent_syn in zip(prompts, syn_sents):
        syn_sents_pure.append(sent_syn.replace(sent, '').replace('\n',' ').strip())

    preds = model.predict(np.array(syn_sents_pure), steps=1)

    assert preds.shape[0] == len(prompts) and preds.shape[1] == num_classes

    label_oht = tf.keras.utils.to_categorical( np.array([label_idx[l] for l in labels]), num_classes = num_classes, dtype='int' ) 

    assert label_oht.shape == preds.shape

    loss_value = cce(label_oht, preds)#.numpy()
    return loss_value


with tf.GradientTape() as tape:
    # Run the forward pass of the layer.
    # The operations that the layer applies
    # to its inputs are going to be recorded
    # on the GradientTape.
    #logits = model(x_batch_train, training=True)  # Logits for this minibatch

    inputs = tokenizer(prompts, padding='max_length', truncation=True, max_length=max_len, return_tensors="tf")
    output_sequences = gpt2.generate(
        input_ids = inputs['input_ids'],
        attention_mask = inputs['attention_mask'],
        max_length= max_len*2,
        temperature=1,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1,
        do_sample=True,
        num_return_sequences=num_return_sequences
    )

    # Compute the loss value for this minibatch.
    #loss_value = loss_fn(y_batch_train, logits)
    loss_value = loss_fn(output_sequences, labels)


# Use the gradient tape to automatically retrieve
# the gradients of the trainable variables with respect to the loss.
grads = tape.gradient(loss_value, gpt2.trainable_weights)

# Run one step of gradient descent by updating
# the value of the variables to minimize the loss.
optimizer.apply_gradients(zip(grads, model.trainable_weights))


y_true = np.array([[0, 1, 0], [0, 0, 1]])
y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
# Using 'auto'/'sum_over_batch_size' reduction type.
cce(y_true, y_pred).numpy()














