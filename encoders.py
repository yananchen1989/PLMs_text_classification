import pandas as pd 
import tensorflow as tf
from load_data import * 
import tensorflow_hub as hub
import tensorflow_text as text
import random,gc,csv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F

class encoder():
    def __init__(self, m):
        self.m = m
        print('loading m:', self.m)
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        if self.m == 'dan':
            # https://tfhub.dev/google/universal-sentence-encoder/4
            #self.model = hub.load("./universal-sentence-encoder_4")
            encoder = hub.KerasLayer("./universal-sentence-encoder_4")
            embed = encoder(text_input)
        elif self.m == 'distil':
            self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')
            #self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')
        elif self.m == 'cmlm':    
            # https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1 
            encoder = hub.KerasLayer("./universal-sentence-encoder-cmlm_en-base_1")
            preprocessor = hub.KerasLayer("./bert_en_uncased_preprocess_3")
            embed = encoder(preprocessor(text_input))["default"]
        else:
            raise KeyError("model illegal!")
        if self.m in ['dan','cmlm']:
            self.model = tf.keras.Model(inputs=text_input, outputs=embed)

    def infer(self, sents, batch_size=32):
        if self.m in ['dan', 'cmlm']:
            embeds = self.model.predict(sents, batch_size=batch_size, verbose=1)
        elif self.m == 'distil':
            embeds = self.model.encode(sents, batch_size=batch_size,  show_progress_bar=True)
        else:
            raise KeyError("model illegal!")
        return embeds


# def insert_label(sent, label, rep=0.1):
#     tokens = sent.split(' ')
#     for i in range(int(len(tokens)*rep )):
#         ix = random.randint(0, len(tokens) - 1)
#         #tokens.insert(ix, label)
#         tokens[ix] = label
#     return ' '.join(tokens)

# m = 'dan'
# enc = encoder(m)

# dsn = '20news'
# ds = load_data(dataset=dsn)
# labels = ds.df_test['label'].unique()
# print(dsn, labels)
# embeds = enc.infer(ds.df_test['content'].tolist(), batch_size = 512) 

# embeds_label = enc.infer(list(labels), batch_size=16) 

# label_simis = {}
# for ix, label in enumerate(list(labels)):
#     simi = F.cosine_similarity(torch.tensor(embeds), torch.tensor(embeds_label[ix].reshape(1,-1))).numpy()
#     label_simis[label] = simi 

# df_simis = pd.DataFrame(label_simis)

# df_simis['pred'] = df_simis.idxmax(axis=1)
# df_simis['label'] = ds.df_test['label']
# acc = df_simis.loc[df_simis['pred']==df_simis['label']].shape[0] / df_simis.shape[0]
# print('enc:', m,  'dsn:', dsn, '  acc==>', acc)

# base = []
# with open('../datasets_aug/cnn_dailymail_stories.csv', 'r') as f:
#     for line in f:
#         if not line.strip() or len(line.strip().split(' ')) <= 20:
#             continue
#         base.append(line.strip().replace('"',''))

# df_base = pd.DataFrame(random.sample(base, 1000000), columns=['content'])
# del base 
# gc.collect()
# print('df base==>', df_base.shape[0])
# #for dsn in ['yahoo','dbpedia','nyt','pop','20news','uci']:

# dsn = 'uci'  
# ds = load_data(dataset= dsn)

# embeds_df_cnndm = enc.infer(df_base['content'].tolist(), batch_size = 512) 

# for l in ds.df['label'].unique():
#     embed_label = enc.infer([l], batch_size=16)
#     simis = cosine_similarity(embed_label, embeds_df_cnndm)
#     df_base[l] = simis[0]
#     print(l)
    
# ds.df_test = ds.df_test.sample(1000)
# labels = ds.df_test['label'].tolist()
# contents = ds.df_test['content'].tolist()
# contents_test_embed = enc.infer(contents, batch_size=512)


# simiall = cosine_similarity(contents_test_embed, embeds_df_cnndm)

# acc = []
# for ii in range(ds.df_test.shape[0]):
#     label = labels[ii]
#     content = contents[ii]
#     test_embed = contents_test_embed[ii]
#     scores = simiall[ii]#cosine_similarity(test_embed.reshape(1,-1), embeds_df_cnndm)
#     #assert scores.shape[0] == samplecnt
#     label_score = {}
#     for l in ds.df['label'].unique():
#         score_l = np.dot(scores, df_base[l].values)
#         label_score[l] = score_l

#     label_score_sort = sorted(label_score.items(), key=lambda item: item[1], reverse=True)
#     pred = label_score_sort[0][0]
#     if pred == label:
#         acc.append(1)
#     else:
#         acc.append(0)
# print('acc==>', sum(acc) / len(acc))











