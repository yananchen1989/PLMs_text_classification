import pandas as pd 
from transformers import pipeline
import os, argparse, random
from load_data import * 
from transblock import * 

# parser = argparse.ArgumentParser()
# parser.add_argument("--dsn", default="ag", type=str)
# args = parser.parse_args()

# ds = load_data(dataset=args.dsn, samplecnt=-1)
# generator  = pipeline("text-generation", model='gpt2', device=0)


# generator_col = []
# ix = 0
# while ix < ds.df_train.shape[0]:
#     dfi = ds.df_train[ix:(ix+64)]
#     results = generator(ds.df_train['content'].tolist(), max_length=200, do_sample=True, top_p=0.9, top_k=0, \
#                 repetition_penalty=1, num_return_sequences=64) 

#     generated_texts_filters = []
#     for ori, gens in zip(dfi['content'].tolist(), results):
#         generated_texts_filter = [ sent['generated_text'].replace(ori, '').replace('\t',' ').replace('\n',' ')\
#                 for sent in  gens ]
#         generated_texts_filters.append(('\t'.join(generated_texts_filter)))

#     generator_col.extend(list(zip(dfi['label'].tolist(), generated_texts_filters)))

#     if len(generator_col) % 000 == 0:
#         print(args.dataset, ' ==> ', len(generator_col), ' /', ds.df_train.shape[0])
#         df = pd.DataFrame(generator_col, columns=['label', 'content'])
#         df.to_csv("{}_df_train.csv".format(args.dsn), index=False)

#     ix += 64




from sklearn.model_selection import train_test_split
while 1:
    ds = load_data(dataset='ag', samplecnt=-1)
    df_t = pd.concat([ds.df_test, ds.df_train])
    df_t['label'] = 1
    del df_t['title']

    df_f = pd.read_csv("zsl_gpt2_contents.tsv")
    df_f['label'] = 0

    if df_f.shape[0] > df_t.shape[0]:
        df_f = df_f.sample(df_t.shape[0])

    if df_f.shape[0] < df_t.shape[0]:
        df_t = df_t.sample(df_f.shape[0])
    assert df_f.shape[0] == df_t.shape[0]
    df = pd.concat([df_f, df_t])
    
    df_train, df_test = train_test_split(df, test_size=0.2)
    (x_train, y_train),  (x_test, y_test), num_classes = get_keras_data(df_train, df_test)

    model = get_model_albert(num_classes)
    model.fit(
                x_train, y_train, batch_size=64, epochs=12, validation_data=(x_test, y_test), verbose=1,
                callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
            )


