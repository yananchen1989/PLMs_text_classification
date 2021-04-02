import sys,os,logging,glob,pickle,torch,random,gc
import numpy as np
import pandas as pd 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from transformers import AutoModelForSequenceClassification, AutoTokenizer
#https://huggingface.co/joeddav/bart-large-mnli-yahoo-answers
#MODEL = "joeddav/bart-large-mnli-yahoo-answers"
# 'facebook/bart-large-mnli'  'joeddav/xlm-roberta-large-xnli'
# from transformers import BartForSequenceClassification, BartTokenizer
# tokenizer = BartTokenizer.from_pretrained(MODEL)
# nli_model = BartForSequenceClassification.from_pretrained(MODEL)
# nli_model.to(device)

def get_sentences():
    df = pd.read_csv("/root/yanan/berts/datasets_aug/cnn_dailymail_stories.csv").sample(frac=1)
    sentences = []
    for content in df['content'].tolist():
        sentences.extend([s for s in content.split('\n') if len(s.strip().split(' ')) >= 50])
    return sentences

sentences = get_sentences()
random.shuffle(sentences) # 703351


#batch_size = 32
labels_candidate = ['Sports', 'Business', 'Science', 'Technology','Entertainment','Politics','Society',\
             'Culture', 'Mathematics','Health','Education','Reference','Computers','Internet',\
             'Finance','Music','Family','Relationships','Government']

'''
infos = []
ix = 0
while ix < len(sentences):
    premises = sentences[ix:ix+batch_size]

    for cate in labels:
        hypothesises = ['This text is about {}.'.format(cate)] * batch_size
        # run through model pre-trained on MNLI
        x = tokenizer(premises, hypothesises, return_tensors='pt', padding=True ,truncation=True,max_length=64)

        logits = nli_model(**x.to(device))[0] 
        entail_contradiction_logits = logits[:,[0,2]].softmax(dim=1)
        true_prob = entail_contradiction_logits[:,1].cpu().detach().numpy()
        
        if true_prob.max() >= 0.9: 
            pos = np.where(true_prob>=0.9)[0]
            for p in pos:
                #print(premises[p])
                infos.append((premises[p], cate))
    ix += batch_size
    logger.info('ix==>{}'.format(ix))

    gc.collect()


'''

from transformers import pipeline
nlp = pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers", device=0) #  

#nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# 

#content = "Johnson Helps D-Backs End Nine-Game Slide (AP) AP - Randy Johnson took a four-hitter into the ninth inning to help the Arizona Diamondbacks end a nine-game losing streak Sunday, beating Steve Trachsel and the New York Mets 2-0."
# generate selected samples which belong to the defined categories
import csv
save_train_file = open('cnn_dm_nli.csv', 'w')
writer = csv.writer(save_train_file, delimiter='\t')

for content in sentences:
    result = nlp(content, labels_candidate, multi_label=False, hypothesis_template="This text is about {}.")

    if result['scores'][0] < 0.4:
        continue
    writer.writerow([content.strip().replace('\t',' '), result['labels'][0]])



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
        "world news",
        "Business",
        "science and technology"]

import csv,random
model_name = 'gpt2'
save_train_file = open('{}.csv'.format(model_name), 'w')
writer = csv.writer(save_train_file, delimiter='\t')


from transformers import pipeline
model  = pipeline("text-generation", model=model_name, device=0) #  

while 1:
    label = random.sample(labels, 1)[0]
    results = model(label, max_length=250, do_sample=True, top_p=0.9, top_k=0, num_return_sequences=5)
    for content in results:
        content = content.replace(label, '').replace('\t',' ')
        if len(content.split(' ')) <= 30:
            continue 
        writer.writerow([label, content])





