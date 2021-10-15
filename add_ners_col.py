import random,os
os.environ['CUDA_VISIBLE_DEVICES'] = "6,7"

import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load("flair/ner-english-fast")


text = '''
Reuters/Vincest West Lucasfilm is eyeing a date for an Obi-Wan Kenobi solo.\nLucasfilm is already eyeing a production date for an Obi-Wan Kenobi solo film.\nThere are plans to create a solo debut for Obi-Wan Kenobi from the "Star Wars" franchise, and according to Omega Underground, Lucasfilm Productions is setting a film date in January 2019. The set location is expected to be at Pinewood Studios in the United Kingdom.\nMeanwhile, the anticipated Obi-Wan debut also has a working title, which is "Joshua Tree." The working title offers hints that the film will feature scenes when Obi-Wan was watching over a young Luke Skywalker in Tatooine.\nAccording to Screenrant, the story of Obi-Wan in his solo film could take place between the saga\'s episode 3, "Star Wars: Revenge of the Sith," and episode 4, "A New Hope." However, there might be difficulty in finding the perfect timeline for the Obi-Wan solo.\nObi-Wan\'s adventures have been featured in many other platforms aside from the "Star Wars" saga. His character and adventures already appeared in the animated "Star Wars: The Clone Wars," as well as the "Star Wars Rebels" TV series. Hopefully the upcoming Obi-Wan spin-off won\'t collide with other stories from the "Star Wars" franchise.\nThere are speculations that Stephen Daldry, director of "The Hours" and "The Reader" is currently in talks to spearhead the upcoming Obi-Wan solo film.\nAlthough it is yet to be confirmed, fans are already expecting Ewan McGregor to fulfil the role of Obi-Wan in his solo debut. There are talks that McGregor was not given the opportunity to embody the character of Obi-Wan in the prequel trilogy of "Star Wars," and that reprising the role in a solo would give him that chance.\nAside from that, fans won\'t be happy if Lucasfilm hires another actor to play Obi-Wan, since he is the original portrayer.\nThe director and screenwriter for the Obi-Wan Kenobi solo film are yet to be confirmed.
'''

def get_ners(text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    ners = list(set([ii['text'] for ii in sentence.to_dict(tag_type='ner')['entities']]))
    random.shuffle(ners)
    return '<=>'.join(ners)

ners = get_ners(text)

df = pd.read_csv("./torch_ds/df_cc_news.csv", lineterminator='\n')

df['ners'] = df['content'].map(lambda x: get_ners(x))
df.to_csv("./torch_ds/df_cc_news_ners.csv", index=False)