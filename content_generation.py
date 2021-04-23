import pandas as pd 
from transformers import pipeline
import os, argparse, random
from load_data import * 
from transblock import * 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="ag", type=str)
args = parser.parse_args()

ds = load_data(dataset=args.dataset, samplecnt=-1)
generator  = pipeline("text-generation", model='gpt2', device=0)


generator_col = []
ix = 0
while ix < ds.df_train.shape[0]:
    dfi = ds.df_train[ix:(ix+64)]
    results = generator(dfi['content'].tolist(), max_length=200, do_sample=True, top_p=0.9, top_k=0, \
                repetition_penalty=1, num_return_sequences=8) 

    generated_texts_filters = []
    for ori, gens in zip(dfi['content'].tolist(), results):
        generated_texts_filter = [ sent['generated_text'].replace(ori, '').replace('\t',' ').replace('\n',' ')\
                for sent in  gens ]
        generated_texts_filters.append(('\t'.join(generated_texts_filter)))

    generator_col.extend(list(zip(dfi['label'].tolist(), generated_texts_filters)))

    if len(generator_col) % 2000 == 0:
        print(args.dataset, ' ==> ', len(generator_col), ' /', ds.df_train.shape[0])
        df = pd.DataFrame(generator_col, columns=['label', 'content'])
        df.to_csv("{}_df_train.csv".format(args.dataset), index=False)

    ix += 64






















