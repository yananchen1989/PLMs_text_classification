import sys,os,logging,glob,pickle,torch,joblib
import numpy as np
import tensorflow as tf
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups


cap = 600

def sample_stratify(df, samplecnt, seed):
    if samplecnt < 0:
        return df 
    ll = []
    for cate in df['label'].unique():
        dfs = df.loc[df['label']==cate].sample(samplecnt, random_state=seed)
        ll.append(dfs)
    return pd.concat(ll).sample(frac=1)

class load_data():
    def __init__(self, samplecnt = -1, dataset='yahoo', seed=1234):
        self.samplecnt = samplecnt
        self.dataset = dataset
        self.seed = seed 
        self.path = './torch_ds'
        if self.dataset == 'ag':
            self.df_train, self.df_test, self.df = self.get_ag_news()
        elif self.dataset == 'bbc':
            self.df_train, self.df_test, self.df = self.get_bbc_news()
        elif self.dataset ==  'bbcsport':
            self.df_train, self.df_test, self.df = self.get_bbcsports_news()
        elif self.dataset == 'tweet':
            self.df_train, self.df_test, self.df = self.get_tweet()
        elif self.dataset == 'yahoo':
            self.df_train, self.df_test, self.df = self.get_yahoo_news()
        elif self.dataset == 'pop':
            self.df_train, self.df_test, self.df = self.get_pop_news()    
        elif self.dataset == 'uci':
            self.df_train, self.df_test, self.df = self.get_uci_news()   
        elif self.dataset == 'dbpedia':
            self.df_train, self.df_test, self.df = self.get_dbpedia_news() 
        elif self.dataset == '20news':
            self.df_train, self.df_test, self.df = self.get_20_news() 
        elif self.dataset == 'nyt':
            self.df_train, self.df_test, self.df = self.get_nyt_news()
        elif self.dataset == 'snips':
            self.df_train, self.df_test, self.df = self.get_snips()
        elif self.dataset == 'stsa':
            self.df_train, self.df_test, self.df = self.get_stsa()
        else:
            raise KeyError("dataset illegal!")

    def get_yahoo_news(self):
        yahoo_label_name = {1: 'Society & Culture',
                             2: 'Science & Mathematics',
                             3: 'Health',
                             4: 'Education & Reference',
                             5: 'Computers & Internet',
                             6: 'Sports',
                             7: 'Business & Finance',
                             8: 'Entertainment & Music',
                             9: 'Family & Relationships',
                             10: 'Politics & Government'}
        ds_train, ds_test =  torchtext.datasets.YahooAnswers(root=self.path, split=('train', 'test'))
        df_train,df_test = pd.DataFrame(ds_train, columns=['label', 'content']), pd.DataFrame(ds_test, columns=['label','content'])

        df_train['label'] = df_train['label'].map(lambda x: yahoo_label_name[x])
        df_test['label'] = df_test['label'].map(lambda x: yahoo_label_name[x])

        df = pd.concat([df_train, df_test]).sample(frac=1)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df 
    
    def get_imdb(self):
        ds_train, ds_test =  torchtext.datasets.IMDB(root=self.path, split=('train', 'test'))      
        df_train,df_test = pd.DataFrame(ds_train, columns=['label', 'content']), pd.DataFrame(ds_test, columns=['label','content'])
        ixl = {'pos':'positive', 'neg':'negative'}
        df_train['label'] = df_train['label'].map(lambda x: ixl[x])
        df_test['label'] = df_test['label'].map(lambda x: ixl[x])
        df = pd.concat([df_train, df_test]).sample(frac=1)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df 

    def get_yelp5(self):
        ds_train, ds_test =  torchtext.datasets.YelpReviewFull(root=self.path, split=('train', 'test'))
        df_train,df_test = pd.DataFrame(ds_train, columns=['label', 'content']), pd.DataFrame(ds_test, columns=['label','content'])
        df = pd.concat([df_train, df_test]).sample(frac=1)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df       

    def get_yelp2(self):
        ds_train, ds_test =  torchtext.datasets.YelpReviewPolarity(root=self.path, split=('train', 'test'))
        df_train,df_test = pd.DataFrame(ds_train, columns=['label', 'content']), pd.DataFrame(ds_test, columns=['label','content'])
        df = pd.concat([df_train, df_test]).sample(frac=1)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df   


    def get_amazon5(self):
        ds_train, ds_test =  torchtext.datasets.AmazonReviewFull(root=self.path, split=('train', 'test'))
        df_train,df_test = pd.DataFrame(ds_train, columns=['label', 'content']), pd.DataFrame(ds_test, columns=['label','content'])
        df = pd.concat([df_train, df_test]).sample(frac=1)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df       

    def get_amazon2(self):
        ds_train, ds_test =  torchtext.datasets.AmazonReviewPolarity(root=self.path, split=('train', 'test'))
        df_train,df_test = pd.DataFrame(ds_train, columns=['label', 'content']), pd.DataFrame(ds_test, columns=['label','content'])
        df = pd.concat([df_train, df_test]).sample(frac=1)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df   

    def get_tweet(self):
        files = glob.glob("../datasets_aug/tweetraw/*.txt")
        infos = []
        for f in files:
            label = f.split('/')[-1].replace('.txt','')
            with open(f,'r') as ff:
                for line in ff:
                    infos.append((label, line.strip()))
        df = pd.DataFrame(infos, columns=['label','content'])
        df_train, df_test = train_test_split(df, test_size=0.2)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df.sample(frac=1)   

    def get_uci_news(self):
        df = pd.read_csv("../datasets_aug/uci-news-aggregator.csv")  
        df = df[['CATEGORY','TITLE']]
        df.rename(
                columns={"CATEGORY": "label", "TITLE":"content"},
                inplace=True )
        ld = {'e':'entertainment', 'b':'business', 't':"science technology", 'm':"health"}
        df['label'] = df['label'].map(lambda x: ld[x])
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=self.seed)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test , df.sample(frac=1)

    # ag news
    def get_dbpedia_news(self):
        ds_train, ds_test =  torchtext.datasets.DBpedia(root=path, split=('train', 'test'))
        df_train,df_test = pd.DataFrame(ds_train, columns=['label', 'content']), pd.DataFrame(ds_test, columns=['label','content'])

        ixl = {1:"Company",
                2:"Educational Institution",
                3:"Artist",
                4:"Athlete",
                5:"Office Holder",
                6:"Mean Of Transportation",
                7:"Building",
                8:"Natural Place",
                9:"Village",
                10:"Animal",
                11:"Plant",
                12:"Album",
                13:"Film",
                14:"Written Work"}

        df_train['label'] = df_train['label'].map(lambda x: ixl[x])
        df_test['label'] = df_test['label'].map(lambda x: ixl[x])
        df = pd.concat([df_train, df_test]).sample(frac=1)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df

    # ag news
    def get_ag_news(self):
        world_replace = ' '.join(['Politics','War','Military','Terrorism','Election','Finance',\
                   'Crime','Murder','Religion','Jurisdiction', 'Democracy'])
        ds_train, ds_test =  torchtext.datasets.AG_NEWS(root=path, split=('train', 'test'))
        df_train,df_test = pd.DataFrame(ds_train, columns=['label', 'content']), pd.DataFrame(ds_test, columns=['label','content'])

        agnews_label = {1:world_replace, 2:"Sports", 3:"Business", 4:"Science and technology"}
        df_train['label'] = df_train['label'].map(lambda x: agnews_label[x])
        df_test['label'] = df_test['label'].map(lambda x: agnews_label[x])
        df = pd.concat([df_train, df_test]).sample(frac=1)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df

    # bbc 
    def get_bbc_news(self):
        infos = []
        for cate in ['business', 'entertainment', 'politics', 'sport', 'tech']:
            files = glob.glob("../datasets_aug/bbc/{}/*.txt".format(cate))
            for ff in files:
                with open(ff, 'r', errors='ignore') as f :
                    content = f.read()
                    infos.append((content, cate))         
        df = pd.DataFrame(infos, columns=['content', 'label'])
        df['label'] = df['label'].map(lambda x: 'technology' if x=='tech' else x)

        df_train, df_test = train_test_split(df, test_size=0.5)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df.sample(frac=1)

    # bbc sports
    def get_bbcsports_news(self):
        infos = []
        for cate in ['athletics', 'cricket', 'football', 'rugby', 'tennis']:
            files = glob.glob("../datasets_aug/bbcsport/{}/*.txt".format(cate))
            for ff in files:
                with open(ff, 'r', errors='ignore') as f :
                    content = f.read()
                    infos.append((content, cate))         
        df = pd.DataFrame(infos, columns=['content', 'label'])
        df_train, df_test = train_test_split(df, test_size=0.5)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df.sample(frac=1)

    def get_pop_news(self):
        df_train = pd.read_csv("../datasets_aug/pop_news/train_file.csv")    
        df_test = pd.read_csv("../datasets_aug/pop_news/test_file.csv")
        df_train = df_train[['Headline','Title','Topic']]
        df_test = df_test[['Headline','Title','Topic']]
        df_train['content'] = df_train['Headline'] + ' ' + df_train['Title']
        df_test['content'] = df_test['Headline'] + ' ' + df_test['Title']    
        del df_train['Headline'], df_train['Title'],  df_test['Headline'],df_test['Title']   
        df_train.rename(
                columns={"Topic": "label"},
                inplace=True )
        df_test.rename(
                columns={"Topic": "label"},
                inplace=True )  
        df = pd.concat([df_train, df_test]).sample(frac=1)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)      
        return df_train, df_test, df

    def get_20_news(self):
        label_name_map = {
            'rec.autos':'autos automobile', 
            'comp.sys.mac.hardware':'computer system mac hardware', 
            'comp.graphics':'computer graphics', 
            'sci.space': 'science space',
            'talk.politics.guns':'politics guns', 
            'sci.med':'science medicine illness disease', 
            'comp.sys.ibm.pc.hardware': 'computer system ibm pc hardware',
            'comp.os.ms-windows.misc':'computer os microsoft windows', 
            'rec.motorcycles': 'motorcycles', 
            'talk.religion.misc':'religion',
            'misc.forsale':'forsale for sale', 
            'alt.atheism':'atheism', 
            'sci.electronics':'science electronics', 
            'comp.windows.x':'computer windows x',
            'rec.sport.hockey':'sport hockey', 
            'rec.sport.baseball':'sport baseball', 
            'soc.religion.christian':'religion christian',
            'talk.politics.mideast':'politics middle east', 
            'talk.politics.misc':'politics', 
            'sci.crypt':'science encryption'
            }
        #data_train = fetch_20newsgroups(subset='train',shuffle=True)
        #joblib.dump(data_train, '20news_data_train')
        data_train = joblib.load('../datasets_aug/20newsgroups/20news_data_train')
        df_train = pd.DataFrame(zip(data_train['data'], list(data_train['target'])), columns=['content','label'])
        ixl = {ix:n for ix, n in enumerate(data_train['target_names'])}
        df_train['label'] = df_train['label'].map(lambda x: label_name_map[ixl[x]])

        #data_test = fetch_20newsgroups(subset='test',shuffle=True)
        #joblib.dump(data_test, '20news_data_test')
        data_test = joblib.load('../datasets_aug/20newsgroups/20news_data_test')
        df_test = pd.DataFrame(zip(data_test['data'], list(data_test['target'])), columns=['content','label'])
        ixl = {ix:n for ix, n in enumerate(data_test['target_names'])}
        df_test['label'] = df_test['label'].map(lambda x: label_name_map[ixl[x]])
        df = pd.concat([df_train, df_test]).sample(frac=1)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)  
        return df_train, df_test, df 
    
    def get_nyt_news(self):      
        infos = []
        with open('../datasets_aug/NYT-Topics/dataset.txt','r') as f:
            for line in f:
                infos.append(line.strip())

        labels = []
        with open('../datasets_aug/NYT-Topics/labels.txt','r') as f:
            for line in f:
                labels.append(int(line.strip()))

        df = pd.DataFrame(zip(infos, labels), columns=['content','label'])

        names = []
        with open('../datasets_aug/NYT-Topics/classes.txt','r') as f:
            for line in f:
                names.append(line.strip())
        ixl = {ix:l for ix, l in enumerate(names)}
        df['label'] = df['label'].map(lambda x: ixl[x])
        df_train, df_test = train_test_split(df, test_size=0.2)
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df.sample(frac=1)  

    def get_stsa(self):
        df_train = pd.read_csv("../datasets_aug/stsa/train.tsv", sep='\t', header=None)
        df_test = pd.read_csv("../datasets_aug/stsa/test.tsv", sep='\t', header=None)
        df_dev = pd.read_csv("../datasets_aug/stsa/dev.tsv", sep='\t', header=None)
        df_train.columns = ['label', 'content']
        df_test.columns = ['label', 'content']
        df_dev.columns = ['label', 'content']
        df_test = pd.concat([df_dev, df_test])
        df = pd.concat([df_train, df_test]).sample(frac=1) 
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df

    def get_snips(self):
        df_train = pd.read_csv("../datasets_aug/snips/train.tsv", sep='\t', header=None)
        df_test = pd.read_csv("../datasets_aug/snips/devtest.tsv", sep='\t', header=None)
        df_train.columns = ['label', 'content']
        df_test.columns = ['label', 'content']
        df = pd.concat([df_train, df_test]).sample(frac=1) 
        df_train = sample_stratify(df_train, self.samplecnt, self.seed)
        return df_train, df_test, df        




def get_keras_data(df_train, df_test, sparse=False):
    num_classes = df_test['label'].unique().shape[0]
    x_train = df_train['content'].values.reshape(-1,1)
    x_test = df_test['content'].values.reshape(-1,1)

    #if num_classes > 2:
    labels = df_test['label'].unique().tolist()
    label_idx = {l:ix for ix, l in enumerate(labels)}

    if not sparse:
        y_train = tf.keras.utils.to_categorical(\
                          df_train['label'].map(lambda x: label_idx.get(x)).values, \
                          num_classes = num_classes, dtype='int' )
        y_test = tf.keras.utils.to_categorical(\
                         df_test['label'].map(lambda x: label_idx.get(x)).values, \
                         num_classes = num_classes, dtype='int' )       
    else:
        y_train = df_train['label'].map(lambda x: label_idx.get(x)).values
        y_test = df_test['label'].map(lambda x: label_idx.get(x)).values   

    return (x_train,y_train),  (x_test, y_test), num_classes, label_idx


   
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

import datetime,csv
def record_log(file, record):
    cur = datetime.datetime.strftime(datetime.datetime.now() + datetime.timedelta(hours=8), '%Y-%m-%d %H:%M:%S')
    with open(file, 'a') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow([cur] + record)

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def get_tokens_len(ds, cap3rd):
    lens = []
    for content in ds.df_test['content'].tolist():
        tokens = tokenizer.tokenize(content)
        lens.append(len(tokens))
    return int(np.quantile(np.array(lens), cap3rd, axis=0))











