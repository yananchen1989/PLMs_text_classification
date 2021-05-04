import pandas as pd 
import tensorflow as tf
from load_data import * 
import tensorflow_hub as hub
import tensorflow_text as text
import random 
from sklearn.metrics.pairwise import cosine_similarity
#from sentence_transformers import SentenceTransformer
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
        #elif self.m == 'distil':
        #    self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')
        elif self.m == 'cmlm':    
            # https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1 
            encoder = hub.KerasLayer("./universal-sentence-encoder-cmlm_en-base_1")
            preprocessor = hub.KerasLayer("./bert_en_uncased_preprocess_3")
            embed = encoder(preprocessor(text_input))["default"]
        else:
            raise KeyError("model illegal!")
        self.model = tf.keras.Model(inputs=text_input, outputs=embed)

    def infer(self, sents, batch_size=32):
        if m in ['dan', 'cmlm']:
            embeds = self.model.predict(sents, batch_size=batch_size, verbose=1)
        #elif m == 'distil':
        #    embeds = self.model.encode(sents, batch_size=batch_size,  show_progress_bar=True)
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

# import GPUtil,time
# while True:
#     memoryUtil = min([gpu.memoryUtil for gpu in GPUtil.getGPUs()])
#     if memoryUtil < 0.1:
#         print('memoryUtil:', memoryUtil)
#         break
#     else:
#         time.sleep(60*5)

batch_size = 32
for rep in [0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.6, 0.7]:
    print("rep==>", rep)
    for m in ['cmlm', 'dan']:
        print('enc==>', m)
        enc = encoder(m)

        for dsn in ['ag','pop', 'yahoo','uci','dbpedia']:
            ds = load_data(dataset=dsn)
            labels = ds.df['label'].unique()
            print(dsn, labels)
            embeds = enc.infer(ds.df['content'].tolist(), batch_size = batch_size) 

            label_simis = {}
            for ll in labels:
                sents = [insert_label(sent, ll, rep=rep) for sent in ds.df['content'].tolist()]
                embeds_ll = enc.infer(sents, batch_size = batch_size) 
                simis = F.cosine_similarity(torch.tensor(embeds), torch.tensor(embeds_ll)).numpy()
                label_simis[ll] = simis
                #simis_ll.append(simis.reshape(-1,1))
            df_simis = pd.DataFrame(label_simis)

            df_simis['pred'] = df_simis.idxmax(axis=1)
            df_simis['label'] = ds.df['label']
            acc = df_simis.loc[df_simis['pred']==df_simis['label']].shape[0] / df_simis.shape[0]
            print('dsn:', dsn, '  acc==>', acc)









