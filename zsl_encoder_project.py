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
        if self.m in ['dan', 'cmlm']:
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



m = 'dan'
enc = encoder(m)

dsn = '20news'
ds = load_data(dataset=dsn)
labels = ds.df_test['label'].unique()
print(dsn, labels)
embeds = enc.infer(ds.df_test['content'].tolist(), batch_size = 512) 

embeds_label = enc.infer(list(labels), batch_size=16) 

label_simis = {}
for ix, label in enumerate(list(labels)):
    simi = F.cosine_similarity(torch.tensor(embeds), torch.tensor(embeds_label[ix].reshape(1,-1))).numpy()
    label_simis[label] = simi 

df_simis = pd.DataFrame(label_simis)

df_simis['pred'] = df_simis.idxmax(axis=1)
df_simis['label'] = ds.df_test['label']
acc = df_simis.loc[df_simis['pred']==df_simis['label']].shape[0] / df_simis.shape[0]
print('enc:', m,  'dsn:', dsn, '  acc==>', acc)









