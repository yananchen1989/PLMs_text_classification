# import pandas as pd
# import glob
# files = glob.glob("/root/yanan/berts/datasets/cnn_dailymail/*_stories_tokenized/*.story")
# stories = []
# for file in files:
#     with open(file, 'r') as f:
#         cl = f.read()
#         cl = cl.replace('@highlight','')
#         stories.append(cl)
# # 312085


# df= pd.DataFrame(stories, columns=['content'])

# df.to_csv("cnn_dailymail_stories.csv", index=False)




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



tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

bos_token = tokenizer.bos_token
eos_token = tokenizer.eos_token
tokenizer.sep_token = '<|sep|>'

df_train = pd.concat([ds_ag.df_train, ds_yahoo.df_train]).sample(frac=1)
df_test = pd.concat([ds_ag.df_test, ds_yahoo.df_test])


def get_finetune_dataset(df, file):
    with open(file,'w') as f:
        for ix, row in df.iterrows():
            line = ' '.join([row['label'], tokenizer.sep_token, row['content'], tokenizer.eos_token])
            f.write(line+'\n')


'''
'science and technology <|sep|> IBM to hire even more new workers By the end of the year, the computing giant plans to have its biggest headcount since 1991. <|endoftext|>'
'World <|sep|> Explosions Echo Throughout Najaf NAJAF, Iraq - Explosions and gunfire rattled through the city of Najaf as U.S. troops in armored vehicles and tanks rolled back into the streets here Sunday, a day after the collapse of talks - and with them a temporary cease-fire - intended to end the fighting in this holy city... <|endoftext|>'
'Business <|sep|> Sneaky Credit Card Tactics Keep an eye on your credit card issuers -- they may be about to raise your rates. <|endoftext|>'
'Business <|sep|> Fund pessimism grows NEW YORK (CNN/Money) - Money managers are growing more pessimistic about the economy, corporate profits and US stock market returns, according to a monthly survey by Merrill Lynch released Tuesday.  <|endoftext|>'
'''

get_finetune_dataset(df_train, 'gpt2_train.txt')
get_finetune_dataset(df_test, 'gpt2_test.txt')






























