import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np 
def check_weights_no_identical(w1, w2):
    assert len(w2.trainable_weights) == len(w1.trainable_weights)
    for i in range(len(w2.trainable_weights)):
        if tf.reduce_sum(w1.trainable_weights[0]).numpy()==0 and tf.reduce_sum(w2.trainable_weights[0]).numpy()==0:
            continue 
        assert not np.array_equal(w1.trainable_weights[i], w2.trainable_weights[i])


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()  
        #if tf.__version__.startswith('2.4') or tf.__version__.startswith('2.5'):        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        #else:
        #    self.att = MultiHeadSelfAttention(num_heads=num_heads, embed_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        #if tf.__version__.startswith('2.4') or tf.__version__.startswith('2.5'): 
        attn_output = self.att(inputs, inputs)
        #else:
        #    attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    # def get_config(self):
    #     cfg = super(TokenAndPositionEmbedding, self).get_config().copy()
    #     return cfg    
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

preprocessor_file = "./albert_en_preprocess_3" # https://tfhub.dev/tensorflow/albert_en_preprocess/3
preprocessor_layer = hub.KerasLayer(preprocessor_file)


def get_model_transormer(num_classes):
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    
    preprocessor = hub.load(preprocessor_file)
    vocab_size = preprocessor.tokenize.get_special_tokens_dict()['vocab_size'].numpy()

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 

    encoder_inputs = preprocessor_layer(text_input)['input_word_ids']

    embedding_layer = TokenAndPositionEmbedding(encoder_inputs.shape[1], vocab_size, embed_dim)
    x = embedding_layer(encoder_inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)

    # if num_classes == 2:
    #     outputs = layers.Dense(1, activation="sigmoid")(x)
    #     model = keras.Model(inputs=text_input, outputs=outputs)
    #     model.compile("adam", "binary_crossentropy", metrics=["binary_accuracy"])
    # else:
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=text_input, outputs=outputs)
    #model.compile("adam", "categorical_crossentropy", metrics=["acc"])

    return model

def get_model_bert(num_classes, m='albert'):

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string
    m_file = {'albert':"./albert_en_base_2", 'electra':'./electra_base_2', 'dan':"./universal-sentence-encoder_4"}

    encoder = hub.KerasLayer(m_file[m], trainable=True)

    if m in ['albert', 'electra']:
        encoder_inputs = preprocessor_layer(text_input)
        outputs = encoder(encoder_inputs)
        embed = outputs["pooled_output"]  
    elif m in ['dan']:
        embed = encoder(text_input)
    else:
        raise KeyError("model illegal!")

    # if num_classes == 2:
    #     out = layers.Dense(1, activation='sigmoid')(embed)
    #     model = tf.keras.Model(inputs=text_input, outputs=out)
    #     model.compile(Adam(lr=1e-5), "binary_crossentropy", metrics=["binary_accuracy"])
    # else:
    out = layers.Dense(num_classes, activation="softmax")(embed)
    model = tf.keras.Model(inputs=text_input, outputs=out)
    #model.compile(Adam(lr=1e-5), "categorical_crossentropy", metrics=["acc"])
    return model




def get_model_bert_pair(trainable):

    text_input1 = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string
    text_input2 = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string

    encoder = hub.KerasLayer('albert_en_base_2', trainable=trainable)

    embed = []
    for textin in [text_input1, text_input2]:
        encoder_inputs = preprocessor_layer(textin)
        outputs = encoder(encoder_inputs)
        embed.append(outputs["pooled_output"])
    embed_all = tf.concat(embed, axis=1)
    x = layers.Dense(512, activation="relu")(embed_all)
    # if num_classes == 2:
    #     out = layers.Dense(1, activation='sigmoid')(embed)
    #     model = tf.keras.Model(inputs=text_input, outputs=out)
    #     model.compile(Adam(lr=1e-5), "binary_crossentropy", metrics=["binary_accuracy"])
    # else:
    out = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=[text_input1, text_input2], outputs=out)
    #model.compile(Adam(lr=1e-5), "categorical_crossentropy", metrics=["acc"])
    return model








