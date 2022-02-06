import pandas as pd
import time
from flair.data import Sentence
from flair.models import SequenceTagger

import torch, flair
flair.device = torch.device('cpu')
#flair.device = torch.device('cuda:0')
#  note that get_ners can only run one single GPU !!! 
tagger = SequenceTagger.load("flair/ner-english-fast")

tagger = SequenceTagger.load("/Users/yanan/.flair/models/ner-english-fast")


text = '''
Reuters/Vincest West Lucasfilm is eyeing a date for an Obi-Wan Kenobi solo.\nLucasfilm is already eyeing a production date for an Obi-Wan Kenobi solo film.\nThere are plans to create a solo debut for Obi-Wan Kenobi from the "Star Wars" franchise, and according to Omega Underground, Lucasfilm Productions is setting a film date in January 2019. The set location is expected to be at Pinewood Studios in the United Kingdom.\nMeanwhile, the anticipated Obi-Wan debut also has a working title, which is "Joshua Tree." The working title offers hints that the film will feature scenes when Obi-Wan was watching over a young Luke Skywalker in Tatooine.\nAccording to Screenrant, the story of Obi-Wan in his solo film could take place between the saga\'s episode 3, "Star Wars: Revenge of the Sith," and episode 4, "A New Hope." However, there might be difficulty in finding the perfect timeline for the Obi-Wan solo.\nObi-Wan\'s adventures have been featured in many other platforms aside from the "Star Wars" saga. His character and adventures already appeared in the animated "Star Wars: The Clone Wars," as well as the "Star Wars Rebels" TV series. Hopefully the upcoming Obi-Wan spin-off won\'t collide with other stories from the "Star Wars" franchise.\nThere are speculations that Stephen Daldry, director of "The Hours" and "The Reader" is currently in talks to spearhead the upcoming Obi-Wan solo film.\nAlthough it is yet to be confirmed, fans are already expecting Ewan McGregor to fulfil the role of Obi-Wan in his solo debut. There are talks that McGregor was not given the opportunity to embody the character of Obi-Wan in the prequel trilogy of "Star Wars," and that reprising the role in a solo would give him that chance.\nAside from that, fans won\'t be happy if Lucasfilm hires another actor to play Obi-Wan, since he is the original portrayer.\nThe director and screenwriter for the Obi-Wan Kenobi solo film are yet to be confirmed.
'''

text = '''
' Photo by Jezael Melgoza on\xa0Unsplash Millions of Satisfied Customers You need reliability in just about anything, who can you go to? Don’t know? How about the Germans and the Japanese, masters of design, perfect engineering precision and trustworthiness. Nothing much has changed over the last fifty years. They’re still knocking out one good unit after another. Proof of it, the millions of satisfied customers who own products from Toshiba, Bosch, Hitachi, Volkswagen, and Sony. And these consumers can’t be wrong. In quantum computing (QC), too, both countries are in commanding positions on the market\u200a—\u200anot quite competing with the United States and China at the moment, disappointingly\u200a—\u200abut they’re riding on those nations’ coattails, for sure. So, like the German QC market, Japan is gaining traction. Currently, the space is dominated by research universities and corporations. The likes of the University of Tokyo, Keio University and its Advancing Quantum Architecture (Aqua) Group and Quantum Computing Center, Osaka University’s Quantum Computing Research Group, Tohoku University and its Quantum Annealing Research Development initiative (T-QARD), Tokyo Institute of Technology (Tokyo Tech) and the Kosaka Laboratory at Yokohama National University are all working hard with their scientists to make new, exciting breakthroughs in QC. On the flip side, the big players of business are playing their part, too: JSR Corporation, Mitsubishi Electric, NTT Basic Research Laboratories, and Quemix, a subsidiary of TerraSky Corporation, are putting their deep pockets where their mouths are by funding research. This leaves private companies and startups. A*Quantum, D Slit Technologies, MDR, Sigma-i, Tokyo Quantum Computing. And joining them is QunaSys Inc, a startup founded in 2018 and located in the Bunkyō ward of Tokyo. The startup believes that: ‘Even though quantum physics is a relatively new field built at the beginning of the 20th century, it plays an important role in the technology that supports us nowadays. Some of the examples are MRI (nuclear magnetic resonance apparatus) and laser therapy used for precise examinations at hospitals, and semiconductors mounted on computers’. QunaSys Inc Led by CEO Tennin Yan, who also co-founded QunaSys, his team is made up of engineers and tech quantum information evangelists, while being supported by a small group of advisors, all with PhDs from Osaka University in quantum information science and quantum chemistry. Yan himself is a graduate of the University of Tokyo. The startup’s ultimate goal is to develop quantum technology while working with credible researchers, highlight its potential and then deliver that knowledge to the masses for the betterment of mankind. QunaSys’ secret weapon is Qulacs, the startup’s ‘high-speed simulator of quantum circuits for quantum computational research’. The system was developed by the Fujii group at Kyoto University and is available in nine active repositories on GitHub. According to the startup’s website, ‘new function proposals and development at GitHub are always welcome’. Qulacs Good news on the funding front is that in November 2019 QunaSys managed to close a Series A round worth $2.5M. All the VCs involved in the round, ANRI, Global Brain and Shinsei Corporate Investment, are Japanese. On the funding round, CEO Tennin Yan commented: ‘This fundraising and the adoption of the SIP project make it possible to focus on the development of quantum computer’s application and software until the achievement of major milestones, ie. quantum advantage’. As always, there is no doubt ganbatte 頑張って(I hope the translation is befitting) of ‘doing your best a la Japanese’ will be thoroughly followed through. The world waits for QunaSys’ impact on the world of QC with bated breath.'
'''




with open('articles_sample.xml.json', 'r') as f:
    jxml = json.load(f)



content = df.loc[df['article_ID']==8036].post_content.tolist()[0]



for ii in jxml:
    if ii['article_id'] == '8036':
        print(ii)
        print(ii['Name'])

content = ds.df_train.loc[ds.df_train['label_name']=='Sports'].sample(1)['content'].tolist()[0]

print(content)
sentence = Sentence(content)
tagger.predict(sentence)

result = sentence.to_dict(tag_type='ner')['ner']

infos = []
for r in result:
    infos.append((r['span'].text, r['value'], r['confidence']))

df_ner = pd.DataFrame(infos, columns=['ner','type','score'])

df_ner.sort_values(['type','score'], ascending=False, inplace=True)

df_ner.drop_duplicates(['ner','type'], inplace=True)

for ix, row in df_ner.iterrows():
    print(row['ner'], '\t', row['type'], '\t', row['score'])





'''
from transformers import pipeline
# t5
from transformers import T5Tokenizer, AutoModelWithLMHead
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
print(tokenizer_t5)

t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)

# ft_model_path = 'ft_model_{}_{}'.format('t5', 'ep')
# checkpoint_files = glob.glob(ft_model_path+"/checkpoint_loss_*")
# list.sort(checkpoint_files)
# t5 = AutoModelWithLMHead.from_pretrained(checkpoint_files[0])  

gen_nlp  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=len(gpus)-1)
sep = tokenizer_t5.eos_token

# gpt2
from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#gpt2 = GPT2LMHeadModel.from_pretrained('ft_model_ft_ep')

gpt2.trainable = False
gpt2.config.pad_token_id = 50256
gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=0, return_full_text=False)

sep = tokenizer_gpt2.sep_token
'''



