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


df_cnndm = pd.read_csv("./datasets/cnn_dailymail_stories.csv")

ds_bbc = load_data(dataset='bbc', samplecnt=-1)


ds_ag = load_data(dataset='ag', samplecnt=-1)

ds_yahoo = load_data(dataset='yahoo', samplecnt=-1)


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


# datasets
import datasets
import pandas as pd
ag_news_train = datasets.load_dataset('ag_news', split="train")
df_train = pd.DataFrame(zip(ag_news_train['text'], ag_news_train['label'] ), columns=['content','label'])

ag_news_test = datasets.load_dataset('ag_news', split="test")
df_test = pd.DataFrame(zip(ag_news_test['text'], ag_news_test['label'] ), columns=['content','label'])


# ['id', 'topic', 'question_title', 'question_content', 'best_answer']
yahoo_news_train = datasets.load_dataset('yahoo_answers_topics', split="train")
yahoo_news_test = datasets.load_dataset('yahoo_answers_topics', split="test")

