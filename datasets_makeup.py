import pandas as pd
import glob
files = glob.glob("/root/yanan/berts/datasets/cnn_dailymail/*_stories_tokenized/*.story")
stories = []
for file in files:
    with open(file, 'r') as f:
        cl = f.read()
        cl = cl.replace('@highlight','')
        stories.append(cl)
# 312085


df= pd.DataFrame(stories, columns=['content'])

df.to_csv("cnn_dailymail_stories.csv", index=False)




# https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

# prepare fine-tune data
from load_data import * 

ds_ag = load_data(dataset='ag', samplecnt=-1)
ds_yahoo = load_data(dataset='yahoo', samplecnt=-1)


agnews_label = {1:"World", 2:"Sports", 3:"Business", 4:"science and technology"}

cates = []
with open('../datasets_aug/yahoo_news/classes.txt','r') as f:
    for line in f:
        cates.append(line.strip())
yahoo_label = {ix+1:cate for ix, cate in enumerate(cates)}


ds_ag.df_train['label'] = ds_ag.df_train['label'].map(lambda x: agnews_label[x])
ds_ag.df_test['label'] = ds_ag.df_test['label'].map(lambda x: agnews_label[x])


ds_yahoo.df_train['label'] = ds_yahoo.df_train['label'].map(lambda x: yahoo_label[x])
ds_yahoo.df_test['label'] = ds_yahoo.df_test['label'].map(lambda x: yahoo_label[x])







df_all = pd.concat([df_cnndm['content'], \
	 ds_bbc.df_train['content'], ds_bbc.df_test['content'], \
	 ds_ag.df_train['content'], ds_ag.df_test['content'], 
	 ds_yahoo.df_train['content'], ds_yahoo.df_test['content']  ])

df_all.columns=['content']

# 1901910

df_all.drop_duplicates(inplace=True)

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_all, test_size=0.05)

df_train.to_csv("df_finetune_train.csv", index=False)
df_test.to_csv("df_finetune_test.csv", index=False)



































