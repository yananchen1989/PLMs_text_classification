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
from transblock import * 

from transformers import GPT2Tokenizer, GPT2LMHeadModel#TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token # to avoid an error "<|endoftext|>": 50256
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2.trainable = True
gpt2.config.pad_token_id=50256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2.to(device)

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

def get_discriminator(num_classes):
    input_embed = keras.Input(shape=(768, ))
    x = layers.Dense(256, activation="relu")(input_embed)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=input_embed, outputs=outputs)
    return model


def synthesize(prompts, labels, max_len):
    inputs = tokenizer(prompts, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
    inputs.to(device)
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
    return tf.convert_to_tensor(np.array(syn_sents_pure))

val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()


# @tf.function
# def test_step(model, x, y):
#     val_logits = model(x, training=False)
#     val_acc_metric.update_state(y, val_logits)


d_optimizer = keras.optimizers.Adam(learning_rate=1e-5)
g_optimizer = keras.optimizers.Adam(learning_rate=1e-5)
gr_optimizer = keras.optimizers.Adam(learning_rate=1e-5)


base_optimizer = keras.optimizers.Adam(learning_rate=1e-5)



















