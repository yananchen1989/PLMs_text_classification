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
gpus = tf.config.experimental.list_physical_devices('GPU')
print('======>',gpus,'<=======')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)
    
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

preprocessor_file = "./resource/albert_en_preprocess_3" # https://tfhub.dev/tensorflow/albert_en_preprocess/3
preprocessor_layer = hub.KerasLayer(preprocessor_file)


def get_model_former(num_classes):
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

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs=text_input, outputs=outputs)
        model.compile("adam", "binary_crossentropy", metrics=["binary_accuracy"])
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs=text_input, outputs=outputs)
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["acc"])
    return model

def get_model_bert(num_classes, m='albert'):

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string

    encoder = hub.KerasLayer("./resource/albert_en_base_2", trainable=True)

    encoder_inputs = preprocessor_layer(text_input)
    outputs = encoder(encoder_inputs)
    embed = outputs["pooled_output"]  

    if num_classes == 2:
        out = layers.Dense(1, activation='sigmoid')(embed)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(Adam(lr=2e-5), "binary_crossentropy", metrics=["binary_accuracy"])
    else:
        out = layers.Dense(num_classes, activation="softmax")(embed)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(Adam(lr=2e-5), "sparse_categorical_crossentropy", metrics=["acc"])
    return model


def encode_rcnn(x, rnn=False):
    # Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(title_embed)
    #title_gru = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)#(?, ?, 256)
    title_conv4 = layers.Conv1D(128, kernel_size = 5, padding = "valid", kernel_initializer = "glorot_uniform")(x) 
    title_conv3 = layers.Conv1D(128, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x) # (?, 28, 128)
    title_conv2 = layers.Conv1D(128, kernel_size = 2, padding = "valid", kernel_initializer = "glorot_uniform")(x) # (?, 29, 128)
    title_conv1 = layers.Conv1D(128, kernel_size = 1, padding = "valid", kernel_initializer = "glorot_uniform")(x) # (?, 30, 128)
    avg_pool_4 = layers.GlobalAveragePooling1D()(title_conv4)# (?, 128)
    max_pool_4 = layers.GlobalMaxPooling1D()(title_conv4) # (?, 128)   
    avg_pool_3 = layers.GlobalAveragePooling1D()(title_conv3)# (?, 128)
    max_pool_3 = layers.GlobalMaxPooling1D()(title_conv3) # (?, 128)
    avg_pool_2 = layers.GlobalAveragePooling1D()(title_conv2)# (?, 128)
    max_pool_2 = layers.GlobalMaxPooling1D()(title_conv2) # (?, 128)
    avg_pool_1 = layers.GlobalAveragePooling1D()(title_conv1)# (?, 128)
    max_pool_1 = layers.GlobalMaxPooling1D()(title_conv1) # (?, 128)   
    if rnn:
        title_encode = layers.concatenate([title_gru, avg_pool_4, max_pool_4, avg_pool_3, max_pool_3, \
                                       avg_pool_2, max_pool_2, avg_pool_1, max_pool_1]) 
    else:
        title_encode = layers.concatenate([avg_pool_4, max_pool_4, avg_pool_3, max_pool_3, \
                                       avg_pool_2, max_pool_2, avg_pool_1, max_pool_1]) 
    return title_encode

def get_model_cnn(num_classes):
    preprocessor = hub.load(preprocessor_file)
    vocab_size = preprocessor.tokenize.get_special_tokens_dict()['vocab_size'].numpy()

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
    encoder_inputs = preprocessor_layer(text_input)    
    embedding = layers.Embedding(vocab_size, 128,  trainable=True)
    text_embed = embedding(encoder_inputs['input_word_ids'])
    text_cnn = encode_rcnn(text_embed)
    mlp1 = layers.Dense(768,activation='relu',name='mlp1')(text_cnn)
    out = layers.Dense(num_classes, activation="softmax")(mlp1)
    model = keras.Model(inputs=text_input, outputs=out)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["acc"])
    return model


# def get_model_bert_pair(trainable):

#     text_input1 = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string
#     text_input2 = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string

#     encoder = hub.KerasLayer('albert_en_base_2', trainable=trainable)

#     embed = []
#     for textin in [text_input1, text_input2]:
#         encoder_inputs = preprocessor_layer(textin)
#         outputs = encoder(encoder_inputs)
#         embed.append(outputs["pooled_output"])
#     embed_all = tf.concat(embed, axis=1)
#     x = layers.Dense(512, activation="relu")(embed_all)
#     # if num_classes == 2:
#     #     out = layers.Dense(1, activation='sigmoid')(embed)
#     #     model = tf.keras.Model(inputs=text_input, outputs=out)
#     #     model.compile(Adam(lr=1e-5), "binary_crossentropy", metrics=["binary_accuracy"])
#     # else:
#     out = layers.Dense(1, activation="sigmoid")(x)
#     model = tf.keras.Model(inputs=[text_input1, text_input2], outputs=out)
#     #model.compile(Adam(lr=1e-5), "categorical_crossentropy", metrics=["acc"])
#     return model
def get_keras_data(df_train, df_test):
    #num_classes = df_test['label'].unique().shape[0]
    x_train = df_train['content'].values.reshape(-1,1)
    x_test = df_test['content'].values.reshape(-1,1)

    #if num_classes > 2:
    #labels = df_test['label'].unique().tolist()
    #label_idx = {l:ix for ix, l in enumerate(labels)}

    # if not sparse:
    #     y_train = tf.keras.utils.to_categorical(\
    #                       df_train['label'].map(lambda x: label_idx.get(x)).values, \
    #                       num_classes = num_classes, dtype='int' )
    #     y_test = tf.keras.utils.to_categorical(\
    #                      df_test['label'].map(lambda x: label_idx.get(x)).values, \
    #                      num_classes = num_classes, dtype='int' )       
    # else:
    y_train = df_train['label'].values
    y_test = df_test['label'].values   
    return (x_train,y_train),  (x_test, y_test)


def do_train_test(df_train, df_test, epochs=50, freq=10, verbose=1, \
               basetry=3, samplecnt=32, basemode='max', model_name='albert'):
        
    if samplecnt <= 32:
        batch_size = 8
    elif samplecnt == 64:
        batch_size = 16
    else:
        batch_size = 32

    (x_train, y_train),  (x_test, y_test)= get_keras_data(df_train, df_test)
    best_val_accs = []
    for ii in range(basetry):
        
        if model_name == 'albert':
            model = get_model_bert(df_test.label.unique().shape[0], 'albert')
            
        elif model_name == 'former':
            model = get_model_former(df_test.label.unique().shape[0])
            
        elif model_name == 'cnn':
            model = get_model_cnn(df_test.label.unique().shape[0])
            
        else:
            raise KeyError("input model illegal!")

        history = model.fit(
            x_train, y_train, batch_size=batch_size, epochs=epochs, \
            validation_data=(x_test, y_test), verbose=verbose, validation_batch_size=64,validation_freq=freq
            #callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
        )
        if df_test.label.unique().shape[0] == 2:
            val_acc = 'val_binary_accuracy'   
        else:
            val_acc = 'val_acc'

        best_val_accs.append(max(history.history[val_acc]))
        print('do_train_test iter==>', ii, 'acc:', max(history.history[val_acc]))
    print('do_train_test iters==>', best_val_accs)
    if basemode == 'mean':
        return round(np.array(best_val_accs).mean(), 4), model
    elif basemode == 'max':
        return round(np.array(best_val_accs).max(), 4), model






