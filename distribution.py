
from sklearn.metrics.pairwise import cosine_distances
from utils.encoders import *
import operator,argparse,torch,sys
from utils.load_data import * 
from sentence_transformers import SentenceTransformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type=='cuda'
from tensorflow.keras.callbacks import *
from utils.transblock import * 
# embedding_df = pd.DataFrame(embeds)
# embedding_df.to_csv('output.tsv', sep='\t', index=None, header=None)
#dfs.to_csv('metadata.tsv', index=False, sep='\t')
parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default='ag', type=str, choices=['yahoo','ag','dbpedia', 'imdb','yelp2','amazon2','stsa','dbpedia'])
parser.add_argument("--samplecnt", default=128, type=int)
parser.add_argument("--setmaxlen", default=0, type=int)
parser.add_argument("--trainable", default=0, type=int)
parser.add_argument("--encm", default='dan', type=str, \
     choices=['dan', 'cmlm', \
     'paraphrase-distilroberta-base-v2','paraphrase-mpnet-base-v2','paraphrase-TinyBERT-L6-v2',\
     'paraphrase-MiniLM-L3-v2', 'average_word_embeddings_glove.6B.300d','nli-distilroberta-base-v2'])
parser.add_argument("--batch_size", default=32, type=int)
args = parser.parse_args()


#enc = encoder(args.encm)

ds = load_data(dataset = args.dsn, samplecnt = args.samplecnt)
print("ds loaded")
(x_train, y_train),  (x_test, y_test), num_classes, label_idx = get_keras_data(ds.df_train, ds.df_test)

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
enc = hub.KerasLayer("./resource/universal-sentence-encoder_4")
embed = enc(text_input)
model_enc = tf.keras.Model(inputs=text_input, outputs=embed)
print('enc loaded')

if args.trainable == 1:
    model_enc.trainable = True
    out = layers.Dense(num_classes, activation="softmax")(model_enc(text_input))
    model = tf.keras.Model(inputs=text_input, outputs=out)
    model.compile(Adam(lr=4e-5), "categorical_crossentropy", metrics=["acc"])

#model_enc.trainable_weights[-1].numpy().mean()
#-0.0010945562
    history = model.fit(
            x_train, y_train, batch_size=32, epochs=50, \
            validation_data=(x_test, y_test), verbose=1,
            callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
        )
elif args.trainable == 0:
    model_enc.trainable = False

elif args.trainable == -1:
    model = get_model_bert(num_classes, 'albert')
    model.compile(Adam(lr=1e-5), "categorical_crossentropy", metrics=["acc"])
    history = model.fit(
            x_train, y_train, batch_size=32, epochs=50, \
            validation_data=(x_test, y_test), verbose=1,
            callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
        )
    best_val_acc = max(history.history['val_acc'])
    print('best acc==>', best_val_acc)
    os._exit(0)

if args.encm not in ['dan','cmlm'] and args.setmaxlen:
    max_len = get_tokens_len(ds, 0.95)
    enc.model.max_seq_length = max(enc.model.max_seq_length, max_len)
    print('enc max len:', enc.model.max_seq_length)


label_embed = {}
for l in ds.df_test['label'].unique():
    print('label==>', l)
    embeds_ = enc.infer(ds.df_train.loc[ds.df_train['label']==l]['content'].tolist(), batch_size=args.batch_size )
    #embeds_ = model_enc.predict(ds.df_train.loc[ds.df_train['label']==l]['content'].tolist(), batch_size=64, verbose=1)
    label_embed[l] = embeds_

infos = []
embeds_test =  enc.infer(ds.df_test['content'].tolist(), batch_size=args.batch_size)
#embeds_test =  model_enc.predict(ds.df_test['content'].tolist(), batch_size=64, verbose=1)

test_labels = ds.df_test['label'].tolist()
for i in range(ds.df_test.shape[0]):
    embed = embeds_test[i].reshape(1,-1)
    label = test_labels[i]

    label_simi = {}
    for l, embed_anchor in label_embed.items():
        simi = cosine_distances(embed, embed_anchor) 
        label_simi[l] = simi.mean()
    #print(label_simi, label)
    if min(label_simi.items(), key=operator.itemgetter(1))[0] == label:
        infos.append(1)
    else:
        infos.append(0)
    if i % 2000 == 0:
        print('{} of {} dsn:{}'.format(i, ds.df_test.shape[0], args.dsn ))
print('summary==>', ' '.join(['{}:{}'.format(k, v) for k, v in vars(args).items()]), sum(infos)/len(infos))




'''
summary==> samplecnt: 128 enc: dan dsn: ag 0.8322368421052632
summary==> samplecnt: 128 enc: dan dsn: ag 0.8355263157894737

ag: 0.82671 ==> 0.83355
stsa: 0.743732 ==> 0.762037

0: 0.7091 1:0.7210 
'''








