'''
['Pregnancy', 'Christianity', 'Explain', 'Fitness', 'Saving', 'Ask', 'Ass', 'Joke', 
'Questions', 'Thoughts', 'Retail', 'Feminism', 'Writing', 'Atheism', 'Netflix', 
'Computing', 'Opinion', 'Alone', 'Funny', 'Gaming', 'Human', 'India', 'Joker', 
'Diet', 'Legal', 'Norman', 'Tip', 'Weight', 'Movies', 'Running', 'Science', 'Horror', 
'Confession', 'Finance', 'Politics', 'Scary', 'Support', 'Technologies', 'Teenage', 
'Event', 'Learned', 'Notion', 'Wikipedia', 'Books', 'Extract', 'Confessions', 'Conspiracy', 
'Links', 'Narcissus', 'Relationship', 'Relationships', 'Reviews', 'News', 'Translation', 'multilingual']
'''
import pandas as pd 
from transformers import pipeline
import os, argparse, random
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="", type=str)
args = parser.parse_args()

import torch
#from transformers import CTRLTokenizer, CTRLLMHeadModel
#tokenizer_ctrl = CTRLTokenizer.from_pretrained('ctrl')
#model = CTRLLMHeadModel.from_pretrained('ctrl')
#control_codes = tokenizer_ctrl.control_codes.keys()

dfcodes = pd.read_csv("label_codes.tsv", sep='\t')

model  = pipeline("text-generation", model=args.model, device=0)

while 1:
    row = dfcodes.sample(1)
    dsn = row['dsn'].tolist()[0]
    label = row['label'].tolist()[0]
    codes = row['codes'].tolist()[0].split(',')
    code = random.sample(codes, 1)[0]
    if args.model == 'ctrl':
        repetition_penalty=1.2
    else:
        repetition_penalty = 1

    results = model(code, max_length=250, do_sample=True, top_p=0.9, top_k=0, \
            repetition_penalty=repetition_penalty, num_return_sequences=32)

    for row in results:
        content = row['generated_text']
        content = content.replace(code, '').replace('\t',' ').replace('\n',' ')
        if len(content.split(' ')) <= 30:
            continue 
        print(dsn, '\t', label, '\t',code, '\t', content)





