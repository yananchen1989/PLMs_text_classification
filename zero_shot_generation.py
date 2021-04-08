'''
['Pregnancy', 'Christianity', 'Explain', 'Fitness', 'Saving', 'Ask', 'Ass', 'Joke', 
'Questions', 'Thoughts', 'Retail', 'Feminism', 'Writing', 'Atheism', 'Netflix', 
'Computing', 'Opinion', 'Alone', 'Funny', 'Gaming', 'Human', 'India', 'Joker', 
'Diet', 'Legal', 'Norman', 'Tip', 'Weight', 'Movies', 'Running', 'Science', 'Horror', 
'Confession', 'Finance', 'Politics', 'Scary', 'Support', 'Technologies', 'Teenage', 
'Event', 'Learned', 'Notion', 'Wikipedia', 'Books', 'Extract', 'Confessions', 'Conspiracy', 
'Links', 'Narcissus', 'Relationship', 'Relationships', 'Reviews', 'News', 'Translation', 'multilingual']
'''
from transformers import pipeline
import os, argparse, random
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="", type=str)
args = parser.parse_args()

import torch
from transformers import CTRLTokenizer, CTRLLMHeadModel
tokenizer_ctrl = CTRLTokenizer.from_pretrained('ctrl')
#model = CTRLLMHeadModel.from_pretrained('ctrl')
control_codes = tokenizer_ctrl.control_codes.keys()

labels = ["Society & Culture",
        "Science & Mathematics",
        "Health",
        "Education & Reference",
        "Computers & Internet",
        "Sports",
        "Business & Finance",
        "Entertainment & Music",
        "Family & Relationships",
        "Politics & Government",
        "palestine",
        "president obama",
        "microsoft",
        "Economy",
        #"world news",
        "Business",
        "science and technology"]



model  = pipeline("text-generation", model=args.model, device=0) #  

while 1:
    label = random.sample(labels, 1)[0]
    results = model(label, max_length=250, do_sample=True, top_p=0.9, top_k=0, num_return_sequences=5)
    for row in results:
        content = row['generated_text']
        content = content.replace(label, '').replace('\t',' ')
        if len(content.split(' ')) <= 30:
            continue 
        print(label, '\t', content.replace('\n',' '))








