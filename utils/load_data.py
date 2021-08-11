import sys,os,logging,glob,pickle,torch,joblib
import numpy as np
import tensorflow as tf
import pandas as pd 
import transformers
#from sklearn.model_selection import train_test_split
#from sklearn.datasets import fetch_20newsgroups
from transformers import AutoTokenizer


tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased',cache_dir="./cache")

def truncate(sent, max_length):
    return tokenizer_bert.batch_decode([tokenizer_bert.encode(sent, truncation=True, max_length=max_length)], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]


cap = 600

def sample_stratify(df, samplecnt):
    if samplecnt < 0:
        return df 
    ll = []
    for cate in df['label'].unique():
        dfs = df.loc[df['label']==cate].sample(samplecnt)
        ll.append(dfs)
    return pd.concat(ll).sample(frac=1)

class load_data():
    def __init__(self, samplecnt = -1, dataset='yahoo', samplecnt_test=10000):
        self.samplecnt = samplecnt
        self.dataset = dataset
        self.path = './torch_ds'
        self.samplecnt_test = samplecnt_test

        if self.dataset in ['ag','yahoo']:
            self.df_train = pd.read_csv('{}/{}_train.csv'.format(self.path, self.dataset))
            self.df_test = pd.read_csv('{}/{}_test.csv'.format(self.path, self.dataset))
            
            if self.dataset == 'ag':
                world_replace = ' '.join(['Politics','War','Military','Terrorism','Election','Finance',\
                                  'Crime','Murder','Religion','Jurisdiction', 'Democracy'])
                ixl = {1:'World', 2:"Sports", 3:"Business", 4:"Science and technology"} 
            if self.dataset == 'yahoo':
                ixl = {  1: 'Society & Culture',
                  2: 'Science & Mathematics',
                  3: 'Health',
                  4: 'Education & Reference',
                  5: 'Computers & Internet',
                  6: 'Sports',
                  7: 'Business & Finance',
                  8: 'Entertainment & Music',
                  9: 'Family & Relationships',
                 10: 'Politics & Government'}
            self.df_train['label_name'] = self.df_train['label'].map(lambda x: ixl.get(x))
            self.df_test['label_name'] = self.df_test['label'].map(lambda x: ixl.get(x))
            self.df_train['label'] = self.df_train['label'] - 1
            self.df_test['label'] = self.df_test['label'] - 1


        elif self.dataset == 'stsa':
            self.df_train = pd.read_csv("{}/stsa/train.tsv".format(self.path), sep='\t', header=None, names=['label', 'content'])
            self.df_test = pd.read_csv("{}/stsa/test.tsv".format(self.path), sep='\t', header=None, names=['label', 'content'])
            self.df_dev = pd.read_csv("{}/stsa/dev.tsv".format(self.path), sep='\t', header=None, names=['label', 'content'])
            self.df_test = pd.concat([self.df_dev, self.df_test])

            self.df_train['label_name'] = self.df_train['label'].map(lambda x: x.lower())
            self.df_test['label_name'] = self.df_test['label'].map(lambda x: x.lower())

            self.df_train['label'] = self.df_train['label'].map({'Negative':0, 'Positive':1})
            self.df_test['label'] = self.df_test['label'].map({'Negative':0, 'Positive':1})
        
        elif self.dataset in ['yelp2','amazon2']:
            self.df_train = pd.read_csv('{}/{}_train.csv'.format(self.path, self.dataset))
            self.df_test = pd.read_csv('{}/{}_test.csv'.format(self.path, self.dataset))
            ixl = {1:'negative', 2:'positive'}
            self.df_train['label_name'] = self.df_train['label'].map(lambda x: ixl.get(x))
            self.df_test['label_name'] = self.df_test['label'].map(lambda x: ixl.get(x))
            self.df_train['label'] = self.df_train['label'] - 1
            self.df_test['label'] = self.df_test['label'] - 1

        elif self.dataset == 'imdb':
            self.df_train = pd.read_csv('{}/{}_train.csv'.format(self.path, self.dataset))
            self.df_test = pd.read_csv('{}/{}_test.csv'.format(self.path, self.dataset))
            ixl = {'neg':'negative', 'pos':'positive'}                                
            self.df_train['label_name'] = self.df_train['label'].map(lambda x: ixl.get(x))
            self.df_test['label_name'] = self.df_test['label'].map(lambda x: ixl.get(x))
            self.df_train['label'] = self.df_train['label'].map({'neg':0, 'pos':1})
            self.df_test['label'] = self.df_test['label'].map({'neg':0, 'pos':1})



        else:
            raise KeyError("dsn illegal!")  

        self.df_train = sample_stratify(self.df_train, self.samplecnt)
        if self.samplecnt_test > 0:
            self.df_test = self.df_test.sample(min(self.df_test.shape[0], self.samplecnt_test))




        


'''
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

    def get_uci_news(self):
        df = pd.read_csv("../datasets_aug/uci-news-aggregator.csv")  
        df = df[['CATEGORY','TITLE']]
        df.rename(
                columns={"CATEGORY": "label", "TITLE":"content"},
                inplace=True )
        ld = {'e':'entertainment', 'b':'business', 't':"science technology", 'm':"health"}
        df['label'] = df['label'].map(lambda x: ld[x])
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=self.seed)


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
   

'''



import datetime,csv
def record_log(file, record):
    cur = datetime.datetime.strftime(datetime.datetime.now() + datetime.timedelta(hours=8), '%Y-%m-%d %H:%M:%S')
    with open(file, 'a') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow([cur] + record)

def get_tokens_len(ds, cap3rd):
    lens = []
    for content in ds.df_train['content'].tolist():
        tokens = tokenizer_bert.tokenize(content)
        lens.append(len(tokens))
    return int(np.quantile(np.array(lens), cap3rd, axis=0))



def process_ds(ds, maxlen=500):
    # ds.df_train['content'] = ds.df_train['content']\
    #       .map(lambda x: x.replace('<br />',' '))
    #if not transformers.__version__.startswith('2.'):
    ds.df_train['content'] = ds.df_train['content'].map(lambda x: truncate(x, maxlen))
    #label_unique = ds.df_test.label.unique()
    # num_classes = label_unique.shape[0]
    proper_len = get_tokens_len(ds, 0.9)
    return ds,  proper_len




expand_label_nli = {}
expand_label_nli['World'] = ['Politics','War','Military','Terrorism','Election','Finance','government', 'ideology',\
                            'legitimacy','socialism','totalitarian','constitution','court','fascism',
                                  'Crime','Murder','Religion','Jurisdiction', 'Democracy']
expand_label_nli['Science and technology'] = ['science','technology','IT','Computers','Internet',\
                                    'algorithm','Space technology','aerospace','boitech','physics','chemistry',\
                                    'biology','scientist','astronomy','universe']
expand_label_nli['Business'] = ['business','finance','oil price','supply','inflation','dollors','bank','Wall Street',\
                        'Federal Reserve','accrual','accountancy','sluggishness','consumers','trade','quarterly earnings',\
                         'deposit','revenue','stocks','recapitalization','marketing']
expand_label_nli['Sports'] = ['sports','athletics','Championships','Football','Olympic','tournament','Chelsea','league','Golf',\
                            'NFL','super bowl','World Cup']



