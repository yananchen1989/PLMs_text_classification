import random,os,torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"



df = pd.read_csv("./torch_ds/df_cc_news.csv", lineterminator='\n')

df['content'] = df['content'].map(lambda x: truncate(x, 512))
ners_all = []

ix = 0
while ix < df.shape[0]:
    torch.cuda.empty_cache()    
    df_tmp = df[ix:ix+2048]

    sentences = [Sentence(sent) for sent in df_tmp['content'].tolist()]
    tagger.predict(sentences)
    ners_tmp = []
    for ii in sentences:
        ners = [j['text'] for j in ii.to_dict(tag_type='ner')['entities']]
        ners_tmp.append('<=>'.join(ners))
    print('{} done'.format(ix/df.shape[0]*100 ))
    ners_all.extend(ners_tmp)
    ix+= 2048

df['ners'] = ners_all
df.to_csv("./torch_ds/df_cc_news_ners.csv", index=False)










