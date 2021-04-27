import pandas as pd 
import tensorflow as tf
from load_data import * 
import tensorflow_hub as hub
import tensorflow_text as text
import random 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class encoder():
    def __init__(self, m):
        self.m = m
        print('loading m:', self.m)
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        if self.m == 'universal':
            # https://tfhub.dev/google/universal-sentence-encoder/4
            #self.model = hub.load("./universal-sentence-encoder_4")
            encoder = hub.KerasLayer("./universal-sentence-encoder_4")
            embed = encoder(text_input)
            self.model = tf.keras.Model(inputs=text_input, outputs=embed)
        elif self.m == 'distil':
            self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')
        elif self.m == 'cmlm':    
            # https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1 
            encoder = hub.KerasLayer("./universal-sentence-encoder-cmlm_en-base_1")
            preprocessor = hub.KerasLayer("./bert_en_uncased_preprocess_3")
            embed = encoder(preprocessor(text_input))["default"]
            self.model = tf.keras.Model(inputs=text_input, outputs=embed)
        else:
            raise KeyError("model illegal!")

    def infer(self, sents, batch_size=32):
        if m in ['universal', 'cmlm']:
            embeds = self.model.predict(sents, batch_size=batch_size, verbose=1)
        elif m == 'distil':
            embeds = self.model.encode(sents, batch_size=batch_size,  show_progress_bar=True)
        else:
            raise KeyError("model illegal!")
        return embeds


def insert_label(sent, label, rep=0.1):
    tokens = sent.split(' ')
    for i in range(int(len(tokens)*rep )):
        ix = random.randint(0, len(tokens) - 1)
        #tokens.insert(ix, label)
        tokens[ix] = label
    return ' '.join(tokens)



for m in ['cmlm', 'universal','distil']:
    print('enc==>', m)
    enc = encoder(m)

    for dsn in ['yahoo','tweet','bbcsport','pop','uci']:
        ds = load_data(dataset=dsn)
        labels = ds.df['label'].unique()
        print(dsn)
        correct_sum = 0
        for l in labels:
            dfl = ds.df.loc[ds.df['label']==l]
            embeds = enc.infer(dfl['content'].tolist(), batch_size = 256) 
            label_embeds = []
            for ll in labels:
                sentsi = [insert_label(sent, ll, rep=0.1) for sent in dfl['content'].tolist()]
                embeds_ll = enc.infer(sentsi, batch_size = 256) 
                simis_matrix = cosine_similarity(embeds, embeds_ll) # (1900, 1900) 
                simis = [simis_matrix[i][i] for i in range(simis_matrix.shape[0])]
                label_embeds.append(simis)
            scores = np.array(label_embeds).T
            preds = [labels[j] for j in scores.argmax(axis=1)]
            correct = sum([1 if p==l else 0 for p in preds ])
            print(l, '==>', correct/dfl.shape[0])
            correct_sum += correct

        print('overall acc==>', correct_sum / ds.df.shape[0])











