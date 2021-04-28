import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()          
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
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

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

preprocessor_file = "./albert_en_preprocess_3"
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
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    #outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=text_input, outputs=outputs)

    model.compile("adam", "categorical_crossentropy", metrics=["acc"])
    #model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return model

def get_model_albert(num_classes):
    # https://tfhub.dev/tensorflow/albert_en_base/2
    encoder = hub.KerasLayer("./albert_en_base_2", trainable=True)
    # https://tfhub.dev/tensorflow/albert_en_preprocess/3
    
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string

    encoder_inputs = preprocessor_layer(text_input)

    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]   # (None, 768)
    sequence_output = outputs["sequence_output"] # (None, 128, 768)
    #pooled_output_ = tf.keras.layers.Dense(256, activation="relu")(pooled_output)

    if num_classes == 2:
        out = layers.Dense(1, activation='sigmoid')(pooled_output)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(Adam(lr=1e-5), "binary_crossentropy", metrics=["binary_accuracy"])
    else:
        out = layers.Dense(num_classes, activation="softmax")(pooled_output)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(Adam(lr=1e-5), "categorical_crossentropy", metrics=["acc"])
    return model

def get_model_electra(num_classes):

    encoder = hub.KerasLayer('./electra_base_2', trainable=True)
    
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string

    encoder_inputs = preprocessor_layer(text_input)

    pooled_output = encoder(encoder_inputs)["pooled_output"] 

    if num_classes == 2:
        out = layers.Dense(1, activation='sigmoid')(pooled_output)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(Adam(lr=1e-5), "binary_crossentropy", metrics=["binary_accuracy"])
    else:
        out = layers.Dense(num_classes, activation="softmax")(pooled_output)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(Adam(lr=1e-5), "categorical_crossentropy", metrics=["acc"])
    return model

def get_model_dan(num_classes):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string
    encoder = hub.KerasLayer("./universal-sentence-encoder_4", trainable=True)
    embed = encoder(text_input)

    if num_classes == 2:
        out = layers.Dense(1, activation='sigmoid')(embed)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(Adam(lr=1e-5), "binary_crossentropy", metrics=["binary_accuracy"])
    else:
        out = layers.Dense(num_classes, activation="softmax")(embed)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(Adam(lr=1e-5), "categorical_crossentropy", metrics=["acc"])
    return model



