

for samplecnt in [100, 1000]:
    for dsn in ['ag','yahoo']:
        for ite  in range(3):  
            ds = load_data(dataset=dsn, samplecnt=samplecnt)
            (x_train, y_train),  (x_test, y_test), num_classes, label_idx = get_keras_data(ds.df_train, ds.df_test)
            model = get_model_bert(num_classes)

            history = model.fit(
                                x_train, y_train, batch_size=64, epochs=100, \
                                validation_batch_size=64,
                                validation_data=(x_test, y_test), verbose=1,
                                callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
                            )
            best_val_acc = max(history.history['val_acc'])
            print(samplecnt, dsn, ite, best_val_acc) 








from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
import tensorflow as tf
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2')
inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1)) # Batch size 1
outputs = model(inputs)
loss = outputs.loss
logits = outputs.logits






nlp = pipeline("ner", model="flair/ner-english-fast")
nlp = pipeline("ner", model="dslim/bert-base-NER")
content = random.sample(contents,1)[0]
print(content)
nlp(content)



content = "Edelman Partners. New York NY\n\nJ.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc\n\nFrequent speaker, former U.S. Ambassador"

content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''




# https://github.com/GT-SALT/MixText/blob/master/data/yahoo_answers_csv/back_translate.ipynb
import torch
while 1:
    en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
    ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')


en2ru.cuda()
ru2en.cuda()

en2de.cuda()
de2en.cuda()

ru2en.translate(en2ru.translate(content,  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)







preprocessor_file = "./albert_en_preprocess_3" # https://tfhub.dev/tensorflow/albert_en_preprocess/3
preprocessor = hub.load(preprocessor_file)
vocab_size = preprocessor.tokenize.get_special_tokens_dict()['vocab_size'].numpy()

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    # def get_config(self):
    #     cfg = super().get_config().copy()
    #     return cfg    
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def get_generator_former_():
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 
    encoder_inputs = preprocessor_layer(text_input)['input_word_ids']
    #embedding_layer = TokenAndPositionEmbedding(encoder_inputs.shape[1], vocab_size, embed_dim)

    token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
    pos_emb = layers.Embedding(input_dim=encoder_inputs.shape[1], output_dim=embed_dim)    

    positions = tf.range(start=0, limit=tf.shape(encoder_inputs)[-1], delta=1)
    x = token_emb(encoder_inputs) + pos_emb(positions)

    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # x = transformer_block(x)

    att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
    layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    dropout1 = layers.Dropout(0.1)
    dropout2 = layers.Dropout(0.1)
    attn_output = att(x, x)
    attn_output = dropout1(attn_output, training=True)
    out1 = layernorm1(x + attn_output)
    ffn_output = ffn(out1)
    ffn_output = dropout2(ffn_output, training=True)
    x = layernorm2(out1 + ffn_output)

    #embed = layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Flatten()(x)
    embed = layers.Dense(768, activation="relu")(x)
    model = keras.Model(inputs=text_input, outputs=embed)
    return model

generator = get_generator_former_()
#new_model = generator.__class__.from_config(generator.get_config())

generator_real = tf.keras.models.clone_model(generator)


generator_real = tf.keras.Model().from_config(generator.get_config())


model_ = tf.keras.models.clone_model(model)

