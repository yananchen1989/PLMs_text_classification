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
import os, argparse, random,gc,csv
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt2", type=str)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--rp", default=1.1, type=int)
args = parser.parse_args()



import torch
#from transformers import CTRLTokenizer, CTRLLMHeadModel
#tokenizer_ctrl = CTRLTokenizer.from_pretrained('ctrl')
#model = CTRLLMHeadModel.from_pretrained('ctrl')
#control_codes = tokenizer_ctrl.control_codes.keys()


from load_data import * 
labels_set = set()
for dsn in ['ag','yahoo','dbpedia','nyt','pop','20news','uci']:
    ds = load_data(dataset=dsn, samplecnt=-1)
    labels_set.update(list(ds.df.label.unique()))
print(labels_set)

del ds 
gc.collect()

labels = list(labels_set)
model  = pipeline("text-generation", model=args.model, device=args.gpu)


nlp = pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers", device=0) #  1.8.1+cu102
def check_premise_score(content, labels_candidate):
    result = nlp(content, labels_candidate, multi_label=False, hypothesis_template="This text is about {}.")
    return result['scores'][0]


while 1:
    label = random.sample(labels, 1)[0]
    if label == 'World':
        code = random.sample(['Politics','War','Military','Terrorism','Election','Finance',\
                   'Crime','Murder','Religion','jurisdiction', 'Democracy'], 1)[0]
    else:
        code = label

    results = model(code, max_length=250, do_sample=True, top_p=0.9, top_k=0, \
            repetition_penalty=args.rp, num_return_sequences=64)

    for row in results:
        content = row['generated_text'].replace(code, '').replace('\t',' ').replace('\n',' ')
        if len(content.split(' ')) <= 30:
            continue 
        premise_score = check_premise_score(content, [code])
        #print(label, '\t', content, '\t', premise_score)
        with open('pseudos_{}.tsv'.format(args.model), 'a') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([label, content, premise_score])

'''
nohup python -u zsl_generation.py --model ctrl --rp 1.2 &
'''

