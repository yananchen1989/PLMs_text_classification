
from load_data import *
from transformers import pipeline
import random,torch
print(torch.__version__)
ds = load_data(dataset='ag', samplecnt=-1)


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
nlp = pipeline("translation_en_to_{}".format(lang), model=model, tokenizer=tokenizer, device=0)

results = nlp(ds.df.sample(1000)['content'].tolist(), max_length=128, do_sample=False)
print(results[0]['translation_text'])



nlp = pipeline("ner", model="dslim/bert-base-NER")
content = random.sample(contents,1)[0]
print(content)
nlp(content)




contents = ds.df_train['content'].tolist()[:50]

nlp  = pipeline("text-generation", model='gpt2', device=0, return_full_text=False)
results = nlp(ds.df_train['content'].tolist()[:50], max_length=250, do_sample=True, top_p=0.9, top_k=0, \
                    repetition_penalty=1, num_return_sequences=128)

contents_syn = [ii['generated_text'] for ii in results]

from encoders import *

enc = encoder('dan')


content_embed = enc.infer([content])

content_syn_embed = enc.infer(contents_syn)

simis = cosine_similarity(content_embed, content_syn_embed)


df_simi = pd.DataFrame(zip(contents_syn, simis[0]), columns=['content','simi'])


df_simi.sort_values(by=['simi'], ascending=False, inplace=True)

df_simi.loc[df_simi['simi']<=0.1]['content'].tolist()


model.encode(ds.df['content'].tolist()[:100], batch_size=32,  show_progress_bar=True)



for dsn in ['ag','yahoo','dbpedia','nyt','pop','20news','uci']:
    ds = load_data(dataset=dsn, samplecnt=-1)
    lens = []
    for content in ds.df['content'].tolist():
        tokens = tokenizer.tokenize(content)
        lens.append(len(tokens))
    print(dsn, np.array(lens).mean())




