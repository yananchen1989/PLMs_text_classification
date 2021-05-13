
from load_data import *
from transformers import pipeline
import random,torch
ds = load_data(dataset='ag', samplecnt=-1)
contents = ds.df.sample(10000)['content'].tolist()
content = contents[1]
print(content)




nlp = pipeline("fill-mask" , model = 'distilbert-base-cased', device=0)


text_mask = text.replace('Fassa Bortolo', nlp.tokenizer.mask_token, 1)

pred_tokens = nlp(text_mask)

text_mask.replace( nlp.tokenizer.mask_token, pred_tokens[0]['token_str'])


ds.df_train.sample(10000)['content'].map(lambda x: augmentor.augment(x)) 

augmentor = fillInmask(ner_set= 0 )






from transformers import pipeline

from transformers import AutoModelWithLMHead, AutoTokenizer
lang = 'zh'
model = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(lang))
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(lang))
nlp = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer, device=0)

results = nlp(contents[:10], max_length=300, do_sample=True)
print(results[-1]['translation_text'])


{'translation_text': "Les patriotes trompent Bills Troy Brown a dû faire beaucoup d'ajustements en jouant des deux côtés du football. quot;Vous voulez toujours marquer quand vous obtenez le ballon -- offense ou défense"},
 {'translation_text': "Boozer fait Jazz Debut dans Win Over Knicks (AP) AP - Carlos Boozer a eu huit points et neuf rebonds dans son premier match de présaison pour Utah, et l'agent gratuit Keith McLeod a terminé avec 11 points et six passes dans la victoire du Jazz 113-89 sur les Knicks de New York mardi soir."},
 {'translation_text': "Garcia prend la tête d'une course aux Masters d'Europe CRANS-SUR-SIERRE, Suisse (Reuters) - Miguel Angel Jimenez a offert une réprimande de bonne qualité à son jeune compatriote Sergio Garcia vendredi pour avoir tenté de le persuader de se reposer avant la Ryder Cup."}




nlp = pipeline("ner", model="dslim/bert-base-NER")
content = random.sample(contents,1)[0]
print(content)
nlp(content)

