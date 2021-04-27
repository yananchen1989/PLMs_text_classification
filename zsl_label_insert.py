import pandas as pd 
import tensorflow as tf
from load_data import * 
import tensorflow_hub as hub
import tensorflow_text as text
import random 
from sklearn.metrics.pairwise import cosine_similarity

# https://tfhub.dev/google/universal-sentence-encoder/4
model = hub.load("./universal-sentence-encoder_4")
embeddings = model(ds.df_train['content'])



# https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1

#english_sentences = tf.constant(["Puppies are nice.", "I enjoy taking long walks along the beach with my dog."])
preprocessor = hub.KerasLayer("./bert_en_uncased_preprocess_3")
encoder = hub.KerasLayer("./universal-sentence-encoder-cmlm_en-base_1")

embeddings = encoder(preprocessor(ds.df['content'].tolist()))["default"].numpy()



ds = load_data(dataset='yahoo', samplecnt=-1)

ds.df_train['label'] = ds.df_train['label'].map(lambda x: yahoo_label_name[x])
embeddings = model(ds.df_train['content'].tolist())

# Convert NumPy array of embedding into data frame
embedding_df = pd.DataFrame(embeddings)
# Save dataframe as as TSV file without any index and header
embedding_df.to_csv('output.tsv', sep='\t', index=None, header=None)

# Save dataframe without any index
df.to_csv('metadata.tsv', index=False, sep='\t')


ds.df['label'] = ds.df['label'].map(lambda x: 'technology' if x=='tech' else x)




labels = ds.df['label'].unique()

embeddings = model(ds.df['content'].tolist())

def insert_label(sent, label, rep=0.1):
    tokens = sent.split(' ')
    for i in range(int(len(tokens)*rep )):
        ix = random.randint(0, len(tokens) - 1)
        tokens.insert(ix, label)
    return ' '.join(tokens)


infos = []
for rep in [0.01, 0.05, 0.1, 0.2, 0.4, 0.5]:
    acc = []
    for ix, row in ds.df.iterrows():
        content = row['content']
        embed = model([content]).numpy()
        #embed = encoder(preprocessor([content]))["default"].numpy()

        pred = ''
        score_init = -1
        for label in labels:
            content_i = insert_label(content, label)
            #embed_i = encoder(preprocessor([content_i]))["default"].numpy()
            embed_i = model([content_i]).numpy()
            score = cosine_similarity(embed, embed_i)[0][0]
            if score > score_init:
                score_init = score 
                pred = label 
        if pred == row['label']:
            acc.append(1)
        else:
            acc.append(0)
    infos.append((rep, sum(acc) / len(acc) ) )
    print('rep:', rep,  ' acc==>', sum(acc) / len(acc))












