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

#nlp_nli = pipeline("zero-shot-classification", model='joeddav/xlm-roberta-large-xnli', device=-1)

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

# def encode_rcnn(x, rnn=False):
#     # Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(title_embed)
#     #title_gru = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)#(?, ?, 256)
#     title_conv4 = layers.Conv1D(128, kernel_size = 4, padding = "valid", kernel_initializer = "glorot_uniform")(x) 
#     title_conv3 = layers.Conv1D(128, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x) # (?, 28, 128)
#     title_conv2 = layers.Conv1D(128, kernel_size = 2, padding = "valid", kernel_initializer = "glorot_uniform")(x) # (?, 29, 128)
#     title_conv1 = layers.Conv1D(128, kernel_size = 1, padding = "valid", kernel_initializer = "glorot_uniform")(x) # (?, 30, 128)
#     avg_pool_4 = layers.GlobalAveragePooling1D()(title_conv4)# (?, 128)
#     max_pool_4 = layers.GlobalMaxPooling1D()(title_conv4) # (?, 128)   
#     avg_pool_3 = layers.GlobalAveragePooling1D()(title_conv3)# (?, 128)
#     max_pool_3 = layers.GlobalMaxPooling1D()(title_conv3) # (?, 128)
#     avg_pool_2 = layers.GlobalAveragePooling1D()(title_conv2)# (?, 128)
#     max_pool_2 = layers.GlobalMaxPooling1D()(title_conv2) # (?, 128)
#     avg_pool_1 = layers.GlobalAveragePooling1D()(title_conv1)# (?, 128)
#     max_pool_1 = layers.GlobalMaxPooling1D()(title_conv1) # (?, 128)   
#     if rnn:
#         title_encode = layers.concatenate([title_gru, avg_pool_4, max_pool_4, avg_pool_3, max_pool_3, \
#                                        avg_pool_2, max_pool_2, avg_pool_1, max_pool_1]) 
#     else:
#         title_encode = layers.concatenate([avg_pool_4, max_pool_4, avg_pool_3, max_pool_3, \
#                                        avg_pool_2, max_pool_2, avg_pool_1, max_pool_1]) 
#     return title_encode

# def get_generator_textcnn():
#     text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
#     encoder_inputs = preprocessor_layer(text_input)    
#     embedding = layers.Embedding(vocab_size, 128,  trainable=True)
#     text_embed = embedding(encoder_inputs['input_word_ids'])
#     text_cnn = encode_rcnn(text_embed)
#     mlp1 = layers.Dense(768,activation='relu',name='mlp1')(text_cnn)
#     model = keras.Model(inputs=text_input, outputs=mlp1)
#     return model
    
# def get_generator_former():
#     embed_dim = 32  # Embedding size for each token
#     num_heads = 2  # Number of attention heads
#     ff_dim = 32  # Hidden layer size in feed forward network inside transformer
#     text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
#     encoder_inputs = preprocessor_layer(text_input)['input_word_ids']
#     embedding_layer = TokenAndPositionEmbedding(encoder_inputs.shape[1], vocab_size, embed_dim)
#     x = embedding_layer(encoder_inputs)
#     transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
#     x = transformer_block(x)
#     #embed = layers.GlobalAveragePooling1D()(x)
#     x = tf.keras.layers.Flatten()(x)
#     embed = layers.Dense(768, activation="relu")(x)
#     model = keras.Model(inputs=text_input, outputs=embed)
#     return model

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
    for sent, label, sent_syn in zip(prompts, labels, syn_sents):
        sent_syn_rm = sent_syn.replace(sent, '').replace('\n',' ').strip()
        sent_syn_eq = sent_syn_rm[:len(sent)]

        # result_nli = nlp_nli(sent_syn_eq, list(label_ix.keys()), multi_label=False, hypothesis_template="This text is about {}.")
        # if result_nli['scores'][0] >= args.thres and result_nli['labels'][0] == ix_label[label]:
        syn_sents_pure.append(sent_syn_eq)
    return tf.convert_to_tensor(np.array(syn_sents_pure))


# def falsify(contents):
    




    

val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()


# @tf.function
# def test_step(model, x, y):
#     val_logits = model(x, training=False)
#     val_acc_metric.update_state(y, val_logits)





# def get_generator_former_():
#     embed_dim = 32  # Embedding size for each token
#     num_heads = 2  # Number of attention heads
#     ff_dim = 32  # Hidden layer size in feed forward network inside transformer
#     text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
#     encoder_inputs = preprocessor_layer(text_input)['input_word_ids']
#     #embedding_layer = TokenAndPositionEmbedding(encoder_inputs.shape[1], vocab_size, embed_dim)

#     token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
#     pos_emb = layers.Embedding(input_dim=encoder_inputs.shape[1], output_dim=embed_dim)    

#     positions = tf.range(start=0, limit=tf.shape(encoder_inputs)[-1], delta=1)
#     x = token_emb(encoder_inputs) + pos_emb(positions)

#     # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
#     # x = transformer_block(x)

#     att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

#     ffn = keras.Sequential(
#             [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
#         )
#     layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#     layernorm2 = layers.LayerNormalization(epsilon=1e-6)
#     dropout1 = layers.Dropout(0.1)
#     dropout2 = layers.Dropout(0.1)
#     attn_output = att(x, x)
#     attn_output = dropout1(attn_output, training=True)
#     out1 = layernorm1(x + attn_output)
#     ffn_output = ffn(out1)
#     ffn_output = dropout2(ffn_output, training=True)
#     x = layernorm2(out1 + ffn_output)

#     #embed = layers.GlobalAveragePooling1D()(x)
#     x = tf.keras.layers.Flatten()(x)
#     embed = layers.Dense(768, activation="relu")(x)
#     model = keras.Model(inputs=text_input, outputs=embed)
#     return model

















