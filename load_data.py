import sys,os,logging,glob,pickle,torch
import numpy as np
import tensorflow as tf
import pandas as pd 
from sklearn.model_selection import train_test_split



def sample_stratify(df, samplecnt):
    if samplecnt < 0:
        return df 
    ll = []
    for cate in df['label'].unique():
        dfs = df.loc[df['label']==cate].sample(samplecnt)
        ll.append(dfs)
    return pd.concat(ll).sample(frac=1)

class load_data():
    def __init__(self, samplecnt = 100, dataset='ag'):
        self.samplecnt = samplecnt
        self.dataset = dataset
        if self.dataset == 'ag':
            self.df_train, self.df_test = self.get_ag_news()
        elif self.dataset == 'bbc':
            self.df_train, self.df_test = self.get_bbc_news()
        elif self.dataset == 'yahoo':
            self.df_train, self.df_test = self.get_yahoo_news()
        elif self.dataset == 'pop':
            self.df_train, self.df_test = self.get_pop_news()            
        else:
            raise KeyError("dataset illegal!")

    def get_yahoo_news(self):
        df_train = pd.read_csv("../datasets_aug/yahoo_news/train.csv", header=None)            
        df_train = df_train.fillna(' ')
        df_train['content'] = df_train[1] + ' ' + df_train[2] + ' ' + df_train[3]
        df_train['label'] = df_train[0]
        df_train = sample_stratify(df_train, self.samplecnt)

        df_test = pd.read_csv("../datasets_aug/yahoo_news/test.csv", header=None)
        df_test = df_test.fillna(' ')
        df_test['content'] = df_test[1] + ' ' + df_test[2] + ' ' + df_test[3]
        df_test['label'] = df_test[0]
        return df_train[['content','label']] , df_test[['content','label']] 

    # ag news
    def get_ag_news(self):
        df_train = pd.read_csv("../datasets_aug/ag_news/train.csv")
        df_test = pd.read_csv("../datasets_aug/ag_news/test.csv")
        df_train['content'] = df_train['title'] + ' ' + df_train['content']
        df_test['content'] = df_test['title'] + ' ' + df_test['content']
        agnews_label = {1:"World", 2:"Sports", 3:"Business", 4:"Sci/Tech"}
        df_train = sample_stratify(df_train, self.samplecnt)
        return df_train, df_test

    # bbc 
    def get_bbc_news(self):
        infos = []
        for cate in ['business', 'entertainment', 'politics', 'sport', 'tech']:
            files = glob.glob("../datasets_aug/bbc/{}/*.txt".format(cate))
            for ff in files:
                with open(ff, 'r', errors='ignore') as f :
                    content = f.read()
                    infos.append((content, cate))         
        df_bbc = pd.DataFrame(infos, columns=['content', 'label'])
        df_train, df_test = train_test_split(df_bbc, test_size=0.5)
        return df_train, df_test

    def get_pop_news(self):
        df_train = pd.read_csv("../datasets_aug/pop_news/train_file.csv")    
        df_test = pd.read_csv("../datasets_aug/pop_news/test_file.csv")
        df_train = df_train[['Headline','Title','Topic']]
        df_test = df_test[['Headline','Title','Topic']]
        df_train['content'] = df_train['Headline'] + ' ' + df_train['Title']
        df_test['content'] = df_test['Headline'] + ' ' + df_test['Title']       
        df_train.rename(
                columns={"Topic": "label"},
                inplace=True )
        df_test.rename(
                columns={"Topic": "label"},
                inplace=True )  
        df_train = sample_stratify(df_train, self.samplecnt)      
        return df_train, df_test

def get_keras_data(df_train, df_test):
    num_classes = df_test['label'].unique().shape[0]
    x_train = df_train['content'].values.reshape(-1,1)
    x_test = df_test['content'].values.reshape(-1,1)

    if num_classes > 2:
        labels = df_test['label'].unique().tolist()
        label_idx = {l:ix for ix, l in enumerate(labels)}


        y_train = tf.keras.utils.to_categorical(\
                          df_train['label'].map(lambda x: label_idx.get(x)).values, \
                          num_classes = num_classes, dtype='int' )
        y_test = tf.keras.utils.to_categorical(\
                         df_test['label'].map(lambda x: label_idx.get(x)).values, \
                         num_classes = num_classes, dtype='int' )       
    else:
        y_train = df_train['label'].values
        y_test = df_test['label'].values    

    return (x_train,y_train),  (x_test, y_test), num_classes

        
stopwords = ['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]