import pandas as pd 
from transformers import pipeline
import os, argparse, random
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="", type=str)
args = parser.parse_args()

ds = load_data(dataset=args.dataset, samplecnt=-1)
generator  = pipeline("text-generation", model='gpt2', device=0)


generator_col = []
for ix, row  in ds.df_train.iterrows():
    results = generator(row['content'], max_length=250, do_sample=True, top_p=0.9, top_k=0, \
                repetition_penalty=1, num_return_sequences=32) 
    generated_texts_filter = [ sent['generated_text'].replace(row['content'], '').replace('\t',' ').replace('\n',' ')\
            for sent in  results ]
    generator_col.append('\t'.join(generated_texts_filter))
    if len(generator_col) % 2000 == 0:
        print(args.dataset, ' ==> ', len(generator_col), ' /', ds.df_train.shape[0])
 
ds.df_train['content_g'] = generator_col

ds.df_train.to_csv("{}_df_train.csv".format(args.dataset), index=False)

