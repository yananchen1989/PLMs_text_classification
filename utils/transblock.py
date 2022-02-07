import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import confusion_matrix
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

    # if num_classes == 2:
    #     out = layers.Dense(1, activation='sigmoid')(embed)
    #     model = tf.keras.Model(inputs=text_input, outputs=out)
    #     model.compile(Adam(learning_rate=2e-5), "binary_crossentropy", metrics=["binary_accuracy"])
    # else:
    out = layers.Dense(num_classes, activation="softmax")(embed)
    model = tf.keras.Model(inputs=text_input, outputs=out)
    model.compile(Adam(learning_rate=2e-5), "sparse_categorical_crossentropy", metrics=["acc"])
    return model


def get_model_bert_ac(num_actions):

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string

    encoder = hub.KerasLayer("./resource/albert_en_base_2", trainable=True, name=str(random.sample(list(range(10000)), 1)[0]))

    encoder_inputs = preprocessor_layer(text_input)
    outputs = encoder(encoder_inputs)
    embed = outputs["pooled_output"]  

    action = tf.keras.layers.Dense(num_actions, activation="softmax")(embed)

    critic = tf.keras.layers.Dense(1)(embed)
    model = tf.keras.Model(inputs=text_input, outputs=[action, critic])

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
        model.compile(tf.keras.optimizers.Adam(learning_rate=0.01), "binary_crossentropy", metrics=["binary_accuracy"])
    else:
        out = tf.keras.layers.Dense(num_classes, activation="softmax")(text_input)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(tf.keras.optimizers.Adam(learning_rate=0.01), "sparse_categorical_crossentropy", metrics=["acc"])
    return model

def get_keras_data(df):
    #num_classes = df_test['label'].unique().shape[0]
    x = df['content'].values.reshape(-1,1)
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
    y = df['label'].values
    return x, y


def get_class_acc(model, x_test, y_test, ixl):
    preds = model.predict(x_test, batch_size=64)
    preds_label = preds.argmax(axis=1)

    #Get the confusion matrix
    cm = confusion_matrix(y_test, preds_label)
    acc = cm.diagonal().sum() / cm.sum()
    for i in np.unique(y_test):
        acc_class = cm.diagonal()[i] / cm[:,i].sum()
        print("acc_class==>", ixl[i], acc_class)

# def do_train_test_valid(df_train_valid, df_test, ixl, epochs=50, freq=10, verbose=1, \
#                basetry=3, basemode='max', model_name='albert'):
    
#     # df_train_valid = ds.df_train
#     # df_test = ds.df_test
#     # model_name = 'albert'
#     # verbose = 1
#     # epochs = 100

#     best_val_accs = []
#     best_test_accs = []
#     models = []
#     for ii in range(basetry):
#         print("basetry==>", ii)
#         df_train, df_valid = train_test_split(df_train_valid, test_size=0.2)

#         x_train, y_train = get_keras_data(df_train)
#         x_valid, y_valid = get_keras_data(df_valid)
#         x_test, y_test = get_keras_data(df_test)

#         #with tf.distribute.MirroredStrategy().scope():
#         with tf.device('/GPU:0'):
#             if model_name == 'albert':
#                 model = get_model_bert(df_test.label.unique().shape[0])
                
#             elif model_name == 'former':
#                 model = get_model_former(df_test.label.unique().shape[0])
                
#             elif model_name == 'cnn':
#                 model = get_model_cnn(df_test.label.unique().shape[0])
                
#             else:
#                 raise KeyError("input model illegal!")

#         model.fit(
#             x_train, y_train, batch_size=16, epochs=epochs, \
#             validation_data=(x_valid, y_valid), verbose=verbose, validation_batch_size=64, 
#             callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=7, mode='max',restore_best_weights=True)]
#         )

#         result_valid = model.evaluate(x_valid, y_valid, batch_size=64)
#         result_test = model.evaluate(x_test, y_test, batch_size=64)

#         best_val_accs.append(result_valid[1])
#         best_test_accs.append(result_test[1])
#         models.append(model)
#         #tf.keras.backend.clear_session()

#     print('do_train_test iters valid==>', best_val_accs)
#     print('do_train_test iters test==>', best_test_accs)
#     #get_class_acc(model, x_test, y_test, ixl)

#     best_model = models[np.array(best_test_accs).argmax()]
#     if basemode == 'mean':
#         return round(np.array(best_test_accs).mean(), 4), best_model
#     elif basemode == 'max':
#         return round(np.array(best_test_accs).max(), 4), best_model

def do_train_test_thread(df_train, df_test, model_name='albert', bs=8, epochs=72):

    # if df_test.label.unique().shape[0] == 2:
    #     val_acc = 'val_binary_accuracy'   
    # else:
    #     val_acc = 'val_acc'

    x_train, y_train = get_keras_data(df_train)
    x_test, y_test = get_keras_data(df_test)

    #with tf.distribute.MirroredStrategy().scope():
    with tf.device('/GPU:0'):
        if model_name == 'albert':
            model = get_model_bert(df_test.label.unique().shape[0])
        elif model_name == 'former':
            model = get_model_former(df_test.label.unique().shape[0])
            
        elif model_name == 'cnn':
            model = get_model_cnn(df_test.label.unique().shape[0])
            
        else:
            raise KeyError("input model illegal!")

    history = model.fit(
        x_train, y_train, batch_size=bs, epochs=epochs, \
        validation_data=(x_test, y_test), verbose=1, validation_batch_size=bs, validation_freq=5 #,
        #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=4, mode='max',restore_best_weights=True)]
    )
    return max(history.history['val_acc']), model
    #best_test_accs.append(max(history.history['val_acc']))
    #models.append(model)

    #print('do_train_test iters test==>', best_test_accs)




def do_train_test_valid_thread(df_train_, df_test, model_name='albert',bs=8):

    df_train, df_valid = train_test_split(df_train_, test_size=0.2)

    x_train, y_train = get_keras_data(df_train)
    x_valid, y_valid = get_keras_data(df_valid)
    x_test, y_test = get_keras_data(df_test)

    #with tf.distribute.MirroredStrategy().scope():
    with tf.device('/GPU:0'):
        if model_name == 'albert':
            model = get_model_bert(df_test.label.unique().shape[0])
            
        elif model_name == 'former':
            model = get_model_former(df_test.label.unique().shape[0])
            
        elif model_name == 'cnn':
            model = get_model_cnn(df_test.label.unique().shape[0])
            
        else:
            raise KeyError("input model illegal!")

    model.fit(
        x_train, y_train, batch_size=bs, epochs=100, \
        validation_data=(x_valid, y_valid), verbose=0, validation_batch_size=bs, 
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, mode='max',restore_best_weights=True)]
    )

    result_test = model.evaluate(x_test, y_test, batch_size=32)

    return result_test[1], model
    #best_test_accs.append(result_test[1])
    #models.append(model)
    #tf.keras.backend.clear_session()

    #print('do_train_test_valid iters test==>', best_test_accs)
    #get_class_acc(model, x_test, y_test, ixl)







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
    try:
        bert_layer = TFBertForNextSentencePrediction.from_pretrained('bert-base-uncased', cache_dir='./cache', local_files_only=True)
    except:
        bert_layer = TFBertForNextSentencePrediction.from_pretrained('bert-base-uncased', cache_dir='./cache')


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

import torch
def nsp_infer(sent1, sent2, bert_nsp, bert_tokenizer):
    device0 = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    scores = []
    for s1, s2 in [(sent1, sent2), (sent2, sent1)]:
        encoding = bert_tokenizer(s1, s2, return_tensors='pt', max_length=256, truncation=True).to(device0)
        outputs = bert_nsp(**encoding, labels=torch.LongTensor([1]).cpu().to(device0) )
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        scores.append(probs.cpu().detach().numpy()[0][0])
    return sum(scores) / 2

def nsp_infer_pairs(pairs, bert_nsp, bert_tokenizer, device0):
    #device0 = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
    pairs_ids = bert_tokenizer.batch_encode_plus(
            pairs,
            add_special_tokens=True,
            max_length= 256,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding='max_length',
            return_tensors="pt",
            truncation=True,  # Truncate to max_length
        ).to(device0)

    outputs = bert_nsp(**pairs_ids, labels=torch.LongTensor([1]*pairs_ids['input_ids'].shape[0]).cpu().to(device0) )

    pairs_ = [[ii[1], ii[0]] for ii in pairs]
    pairs_ids_ = bert_tokenizer.batch_encode_plus(
            pairs_,
            add_special_tokens=True,
            max_length= 256,
            return_attention_mask=True,
            return_token_type_ids=True,
            padding='max_length',
            return_tensors="pt",
            truncation=True,  # Truncate to max_length
        ).to(device0)

    outputs_ = bert_nsp(**pairs_ids_, labels=torch.LongTensor([1]*pairs_ids_['input_ids'].shape[0]).cpu().to(device0) )

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs_ = torch.nn.functional.softmax(outputs_.logits, dim=-1)
    return probs.cpu().detach().numpy() + probs_.cpu().detach().numpy()



def nli_infer(premise, hypothesis, model_nli, tokenizer_nli):
    device0 = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    # run through model pre-trained on MNLI
    x = tokenizer_nli.encode(premise, hypothesis, return_tensors='pt', truncation_strategy='only_first')
    logits = model_nli(x.to(device0))[0]
    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true 
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:,1]
    return prob_label_is_true.cpu().detach().numpy()[0]

    