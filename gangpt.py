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
ds.df_train['label'] = pd.Categorical(ds.df_train['label'])
ds.df_train['label'] = ds.df_train['label'].cat.codes
ds.df_test['label'] = pd.Categorical(ds.df_test['label'])
ds.df_test['label'] = ds.df_test['label'].cat.codes
max_len = get_tokens_len(ds, 0.99) 
num_classes = ds.df_test.label.unique().shape[0]

dstf = tf.data.Dataset.from_tensor_slices((ds.df_train['content'].values, ds.df_train['label'].values))
dstf = dstf.shuffle(buffer_size=10000).batch(32)

def parser(x):
    inputs = tokenizer([ii.decode() for ii in xx], padding='max_length', add_prefix_space=True, truncation=True, max_length=max_len, return_tensors="tf")
    return inputs

# for mm in dstf.map(lambda x, y: (x, y) ).take(5):
#     print(mm)
#     print(sent)
#     print(label)
#     break 
# get2
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token # to avoid an error "<|endoftext|>": 50256
gpt2 = TFGPT2LMHeadModel.from_pretrained('gpt2')
gpt2.trainable = True
gpt2.config.pad_token_id=50256

preprocessor_file = "./albert_en_preprocess_3" # https://tfhub.dev/tensorflow/albert_en_preprocess/3
preprocessor_layer = hub.KerasLayer(preprocessor_file)
encoder = hub.KerasLayer('albert_en_base_2', trainable=True)
preprocessor = hub.load(preprocessor_file)
vocab_size = preprocessor.tokenize.get_special_tokens_dict()['vocab_size'].numpy()

def get_generator_bert():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
    encoder_inputs = preprocessor_layer(text_input)
    outputs = encoder(encoder_inputs)
    embed = outputs["pooled_output"] # (None, 768)
    model = keras.Model(inputs=text_input, outputs=embed)
    return model

def get_generator_former():
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
    encoder_inputs = preprocessor_layer(text_input)['input_word_ids']
    embedding_layer = TokenAndPositionEmbedding(encoder_inputs.shape[1], vocab_size, embed_dim)
    x = embedding_layer(encoder_inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    #embed = layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Flatten()(x)
    embed = layers.Dense(768, activation="relu")(x)
    model = keras.Model(inputs=text_input, outputs=embed)
    return model

def get_discriminator():
    input_embed = keras.Input(shape=(768, ))
    x = layers.Dense(256, activation="relu")(input_embed)
    outputs = layers.Dense(num_classes*2, activation="softmax")(x)
    model = keras.Model(inputs=input_embed, outputs=outputs)
    return model


def synthesize(prompts):
    inputs = tokenizer(prompts, padding='max_length', truncation=True, max_length=max_len, return_tensors="tf")
    output_sequences = gpt2.generate(
        input_ids = inputs['input_ids'],
        attention_mask = inputs['attention_mask'] ,
        max_length= max_len*2,
        temperature=1,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1,
        do_sample=True,
        num_return_sequences=1
    )
    syn_sents = tokenizer.batch_decode(output_sequences, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    syn_sents_pure = []
    for sent, sent_syn in zip(prompts, syn_sents):
        sent_syn_rm = sent_syn.replace(sent, '').replace('\n',' ').strip()
        sent_syn_eq = sent_syn_rm[:len(sent)]
        syn_sents_pure.append(sent_syn_eq)
    return syn_sents_pure

generator = get_generator_bert()
generator_real = tf.keras.models.clone_model(generator)

discriminator = get_discriminator()

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

m = tf.keras.metrics.SparseCategoricalAccuracy()
m_ = tf.keras.metrics.SparseCategoricalAccuracy()

batch_size = 64
ite = 0
stepp = 100
# def loss_fn(output_sequences, labels):

#     preds = model(np.array(syn_sents_pure))

#     assert preds.shape[0] == len(prompts) and preds.shape[1] == num_classes

#     label_oht = tf.keras.utils.to_categorical( np.array([label_idx[l] for l in labels]), num_classes = num_classes, dtype='int' ) 
#     label_oht_tf = tf.convert_to_tensor(label_oht)
#     assert label_oht.shape == preds.shape

#     loss_value = cce(label_oht_tf, preds)#.numpy()
#     return loss_value
while 1:
    rows = ds.df_train.sample(batch_size)
    prompts = rows['content'].tolist()
    labels = rows['label'].tolist()
    
    prompts_syn = synthesize(prompts)
    labels_syn = [i+num_classes for i in labels]

    prompts_tensor = tf.convert_to_tensor(np.array(prompts))
    prompts_syn_tensor = tf.convert_to_tensor(np.array(prompts_syn))

    labels_tensor = tf.convert_to_tensor(np.array(labels), dtype=tf.float32)
    labels_syn_tensor = tf.convert_to_tensor(np.array(labels_syn), dtype=tf.float32) 
    d_loss, g_loss, gr_loss = train_step(prompts_tensor, prompts_syn_tensor, labels_tensor, labels_syn_tensor)
    ite += 1
    print(d_loss.numpy(), g_loss.numpy(), gr_loss.numpy())

    if ite % stepp == 0:
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
        logits = discriminator(generator_real(text_input))
        model = keras.Model(inputs=text_input, outputs=logits)

        preds = model.predict(ds.df_test['content'].values, batch_size=256, verbose=1)

        preds_uni = preds[:,:num_classes] + preds[:,num_classes:]

        m.update_state(ds.df_test['label'].values, preds_uni)
        print('generator_real test acc:', m.result().numpy())
        m.reset_states()

    if ite % stepp == 0:
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
        logits = discriminator(generator(text_input))
        model_ = keras.Model(inputs=text_input, outputs=logits)

        preds_ = model_.predict(ds.df_test['content'].values, batch_size=256, verbose=1)

        preds_uni_ = preds_[:,:num_classes] + preds_[:,num_classes:]

        m_.update_state(ds.df_test['label'].values, preds_uni_)
        print('generator test acc:', m_.result().numpy())
        m_.reset_states()




















