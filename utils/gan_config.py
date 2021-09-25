import sys,os,logging,glob,pickle,csv,datetime,gc,argparse,math
import numpy as np
import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.optimizers import Adam
from tensorflow import keras





'''
gpt2_nlp([sent], max_length=64, do_sample=True, top_p=0.9, top_k=0, \
                        repetition_penalty=1.0, num_return_sequences=4, clean_up_tokenization_spaces=True)

'''












def get_generator_bert():
    preprocessor_layer = hub.KerasLayer("./resource/albert_en_preprocess_3")
    encoder = hub.KerasLayer('./resource/albert_en_base_2', trainable=True)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
    encoder_inputs = preprocessor_layer(text_input)
    outputs = encoder(encoder_inputs)
    embed = outputs["pooled_output"] # (None, 768)
    model = keras.Model(inputs=text_input, outputs=embed)
    return model

def get_generator_mlp():
    input_embed = keras.Input(shape=(512, ))
    x = layers.Dense(256, activation="relu")(input_embed)
    embed = layers.Dense(768)(x)# (None, 768)
    model = keras.Model(inputs=input_embed, outputs=embed)
    return model


    
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

# def get_discriminator_exter():
#     input_embed = keras.Input(shape=(768, ))
#     x = layers.Dense(256, activation="relu")(input_embed)
#     outputs = layers.Dense(1, activation="sigmoid")(x)
#     model = keras.Model(inputs=input_embed, outputs=outputs)
#     return model

def get_discriminator(num_classes):
    input_embed = keras.Input(shape=(768, ))
    x = layers.Dense(256, activation="relu")(input_embed)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=input_embed, outputs=outputs)
    return model



# def synthesize_for_gan(prompts,  max_len, gpt2):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     gpt2.to(device)
#     inputs = tokenizer(prompts, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
#     inputs.to(device)
#     output_sequences = gpt2.generate(
#         input_ids = inputs['input_ids'],
#         attention_mask = inputs['attention_mask'] ,
#         max_length= min(tokenizer.model_max_length, max_len*2),
#         temperature=1,
#         top_k=0,
#         top_p=0.9,
#         repetition_penalty=1,
#         do_sample=True,
#         num_return_sequences=1
#     )
#     syn_sents = tokenizer.batch_decode(output_sequences, clean_up_tokenization_spaces=True, skip_special_tokens=True)

#     syn_sents_pure = []
#     for sent, sent_syn in zip(prompts, syn_sents):
#         sent_syn_rm = sent_syn.replace(sent, '').replace('\n',' ').strip()
#         sent_syn_eq = sent_syn_rm[:len(sent)]

#         # result_nli = nlp_nli(sent_syn_eq, list(label_ix.keys()), multi_label=False, hypothesis_template="This text is about {}.")
#         # if result_nli['scores'][0] >= args.thres and result_nli['labels'][0] == ix_label[label]:
#         syn_sents_pure.append(sent_syn_eq)
#     return tf.convert_to_tensor(np.array(syn_sents_pure))



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

















