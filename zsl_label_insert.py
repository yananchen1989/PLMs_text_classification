import pandas as pd 
import tensorflow as tf
from load_data import * 
import tensorflow_hub as hub
import tensorflow_text as text
import random 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


preprocessor = hub.KerasLayer("./bert_en_uncased_preprocess_3")

def load_model(m):
    if m == 'universal':
        # https://tfhub.dev/google/universal-sentence-encoder/4
        model = hub.load("./universal-sentence-encoder_4")
    elif m == 'distil':
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')
    elif m == 'cmlm':
        # https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1 
        model = hub.KerasLayer("./universal-sentence-encoder-cmlm_en-base_1")
    else:
        raise KeyError("model illegal!")
    return model 
#english_sentences = tf.constant(["Puppies are nice.", "I enjoy taking long walks along the beach with my dog."])


def encoding(model, sents, m):
    if m == 'universal':
        embeds = model(sents)
    elif m == 'distil':
        embeds = model.encode(sents, batch_size=32,  show_progress_bar=False)
    elif m == 'cmlm':
        embeds = model(preprocessor(sents))["default"].numpy()
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



m = 'distil'
model = load_model(m)

for dsn in ['yahoo','tweet','bbcsport','pop','uci']:
    ds = load_data(dataset=dsn)
    labels = ds.df['label'].unique()
    print(dsn, ' ==>', labels)
    if ds.df.shape[0] > 50000:
        ds.df = ds.df.sample(50000)

    correct_sum = 0
    for l in labels:
        dfl = ds.df.loc[ds.df['label']==l]
        embeds = encoding(model, dfl['content'].tolist(), m) 
        label_embeds = []
        for ll in labels:
            sentsi = [insert_label(sent, ll, rep=0.1) for sent in dfl['content'].tolist()]
            embeds_ll = encoding(model, sentsi, m) 
            simis_matrix = cosine_similarity(embeds, embeds_ll) # (1900, 1900) 
            simis = [simis_matrix[i][i] for i in range(simis_matrix.shape[0])]
            label_embeds.append(simis)
        scores = np.array(label_embeds).T
        preds = [labels[j] for j in scores.argmax(axis=1)]
        correct = sum([1 if p==l else 0 for p in preds ])
        print(l, '==>', correct/dfl.shape[0])
        correct_sum += correct

    print('overall acc==>', correct_sum / ds.df.shape[0])











