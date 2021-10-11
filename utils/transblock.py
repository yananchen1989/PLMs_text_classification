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
import random
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

def get_model_bert(num_classes):

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string

    encoder = hub.KerasLayer("./resource/albert_en_base_2", trainable=True, name=str(random.sample(list(range(10000)), 1)[0]))

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


# def encode_rcnn(x, rnn=False):
#     # Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(title_embed)
#     #title_gru = layers.Bidirectional(layers.GRU(128, return_sequences=False))(x)#(?, ?, 256)
#     title_conv4 = layers.Conv1D(128, kernel_size = 5, padding = "valid", kernel_initializer = "glorot_uniform")(x) 
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

# def get_model_cnn(num_classes):
#     preprocessor = hub.load(preprocessor_file)
#     vocab_size = preprocessor.tokenize.get_special_tokens_dict()['vocab_size'].numpy()

#     text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
#     encoder_inputs = preprocessor_layer(text_input)    
#     embedding = layers.Embedding(vocab_size, 128,  trainable=True)
#     text_embed = embedding(encoder_inputs['input_word_ids'])
#     text_cnn = encode_rcnn(text_embed)
#     mlp1 = layers.Dense(768,activation='relu',name='mlp1')(text_cnn)
#     out = layers.Dense(num_classes, activation="softmax")(mlp1)
#     model = keras.Model(inputs=text_input, outputs=out)
#     model.compile("adam", "sparse_categorical_crossentropy", metrics=["acc"])
#     return model


def get_model_mlp(x_train, num_classes):
    text_input = tf.keras.layers.Input(shape=(x_train.shape[1]), dtype=tf.float32) 
    if num_classes == 2:
        out = tf.keras.layers.Dense(1, activation='sigmoid')(text_input)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(tf.keras.optimizers.Adam(lr=0.01), "binary_crossentropy", metrics=["binary_accuracy"])
    else:
        out = tf.keras.layers.Dense(num_classes, activation="softmax")(text_input)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(tf.keras.optimizers.Adam(lr=0.01), "sparse_categorical_crossentropy", metrics=["acc"])
    return model

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
    models = []
    for ii in range(basetry):
        with tf.distribute.MirroredStrategy().scope():
        #with tf.device('/GPU:{}'.format(gpu)):
            if model_name == 'albert':
                model = get_model_bert(df_test.label.unique().shape[0])
                
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
        models.append(model)
        print('do_train_test iter==>', ii, 'acc:', max(history.history[val_acc]))
    print('do_train_test iters==>', best_val_accs)

    best_model = models[np.array(best_val_accs).argmax()]
    if basemode == 'mean':
        return round(np.array(best_val_accs).mean(), 4), best_model
    elif basemode == 'max':
        return round(np.array(best_val_accs).max(), 4), best_model


def fit_within_thread(model, x_train, y_train, batch_size, epochs, x_test, y_test, df_test, best_val_accs, models):
    history = model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, \
        validation_data=(x_test, y_test), verbose=0, validation_batch_size=64, validation_freq=10
    )
    if df_test.label.unique().shape[0] == 2:
        val_acc = 'val_binary_accuracy'   
    else:
        val_acc = 'val_acc'

    best_val_accs.append(max(history.history[val_acc]))
    models.append(model)

def do_train_test_thread(df_train, df_test, epochs=50, freq=10, verbose=1, \
               basetry=3, samplecnt=32, basemode='max', model_name='albert'):
        
    if samplecnt <= 32:
        batch_size = 8
    elif samplecnt == 64:
        batch_size = 16
    else:
        batch_size = 32

    (x_train, y_train),  (x_test, y_test)= get_keras_data(df_train, df_test)

    with tf.distribute.MirroredStrategy().scope():
    #with tf.device('/GPU:{}'.format(gpu)):
        if model_name == 'albert':
            model = get_model_bert(df_test.label.unique().shape[0])
            
        elif model_name == 'former':
            model = get_model_former(df_test.label.unique().shape[0])
            
        elif model_name == 'cnn':
            model = get_model_cnn(df_test.label.unique().shape[0])
            
        else:
            raise KeyError("input model illegal!")
    best_val_accs = []
    models = []

    from threading import Thread
    threads = []
    for ii in range(basetry):
        t = Thread(target=fit_within_thread, args=(model, x_train, y_train, batch_size, epochs, x_test, y_test, df_test,\
                                                best_val_accs, models))
        t.start()
        threads.append(t)

    # join all threads
    for t in threads:
        t.join()
    print("do_train_test joined")
    print('do_train_test iters==>', best_val_accs)
    assert len(best_val_accs) == basetry and len(models) == basetry

    best_model = models[np.array(best_val_accs).argmax()]
    if basemode == 'mean':
        return round(np.array(best_val_accs).mean(), 4), best_model
    elif basemode == 'max':
        return round(np.array(best_val_accs).max(), 4), best_model    



# from utils.load_data import * 

# enc = encoder('cmlm-large','cpu')
# for _ in range(5):
#     ds = load_data(dataset='ag', samplecnt= 1024)
#     acc_aug, _ = do_train_test_cmlm(ds.df_train, ds.df_test, enc)
#     print('iter', acc_aug)


# from utils.encoders import *
def do_train_test_cmlm(df_train, df_test, enc, basemode='max',  gpu=0):
    
    x_train = enc.infer(df_train['content'].values)
    x_test = enc.infer(df_test['content'].values)

    y_train, y_test = df_train['label'].values, df_test['label'].values

    best_val_accs = []
    models = []
    for ii in range(3):
        #with tf.distribute.MirroredStrategy().scope():
        with tf.device('/GPU:{}'.format(gpu)):
            model = get_model_mlp(x_train, df_test.label.unique().shape[0])
                
        if df_test.label.unique().shape[0] == 2:
            val_acc = 'val_binary_accuracy'   
        else:
            val_acc = 'val_acc'

        history = model.fit(
            x_train, y_train, batch_size=32, epochs=400, \
            validation_data=(x_test, y_test), verbose=0, validation_batch_size=64,validation_freq=1
            #callbacks = [tf.keras.callbacks.EarlyStopping(monitor=val_acc, patience=10, mode='max')
        )
        best_val_accs.append(max(history.history[val_acc]))
        models.append(model)
        print('do_train_test iter==>', ii, 'acc:', max(history.history[val_acc]))
    print('do_train_test iters==>', best_val_accs)

    best_model = models[np.array(best_val_accs).argmax()]
    if basemode == 'mean':
        return round(np.array(best_val_accs).mean(), 4), best_model
    elif basemode == 'max':
        return round(np.array(best_val_accs).max(), 4), best_model



from transformers import BertTokenizer, TFBertForNextSentencePrediction
def get_model_nsp(max_length):

    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )
    bert_layer = TFBertForNextSentencePrediction.from_pretrained('bert-base-uncased', cache_dir='./cache', local_files_only=True)

    logits = bert_layer(input_ids, token_type_ids=token_type_ids)[0]
    out = tf.keras.activations.softmax(logits)
    #out = tf.keras.layers.Dense(2, activation="softmax")(logits)
    #out = tf.keras.layers.Dense(1, activation="sigmoid")(logits)
    
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=out
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return model


def get_ids(sentence_pairs,  max_length, tokenizer_bert):
    encoded = tokenizer_bert.batch_encode_plus(
        sentence_pairs,
        add_special_tokens=True,
        max_length= max_length,
        return_attention_mask=True,
        return_token_type_ids=True,
        padding='max_length',
        return_tensors="tf",
        truncation=True,  # Truncate to max_length
    )
    return (encoded["input_ids"] , encoded["attention_mask"], encoded["token_type_ids"])
