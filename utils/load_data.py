import sys,os,logging,glob,pickle,torch,joblib,random
import numpy as np
import tensorflow as tf
import pandas as pd 
import transformers
from sklearn.model_selection import train_test_split
#from sklearn.datasets import fetch_20newsgroups
from transformers import AutoTokenizer

tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased',cache_dir="./cache", local_files_only=True)

def truncate(sent, max_length):
    ids = tokenizer_bert.encode(sent, truncation=True, max_length=max_length)
    sent_ = tokenizer_bert.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return sent_

def noisy_label(l, unique_labels):
    unique_labels_ = [j for j in unique_labels if j!=l]
    assert l not in unique_labels_
    return random.sample(unique_labels_, 1)[0]
    

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
                ixl = {1:'World', 2:"Sports", 3:"Business", 4:"science and technology"} 
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

        elif self.dataset == 'uci':
            df = pd.read_csv("./torch_ds/uci-news-aggregator.csv")  
            df = df[['CATEGORY','TITLE']]
            df.rename(
                    columns={"CATEGORY": "label", "TITLE":"content"},
                    inplace=True )
            ld = {'e':'entertainment', 'b':'business', 't':"science and technology", 'm':"health"}
            ixl = {'e':0, 'b':1, 't':2, 'm':3}
            df['label_name'] = df['label'].map(lambda x: ld[x])
            df['label'] = df['label'].map(lambda x: ixl[x])
            self.df_train, self.df_test = train_test_split(df, test_size=0.3)

        elif self.dataset == 'nyt':
            infos = []
            with open('./torch_ds/nyt/dataset.txt','r') as f:
                for line in f:
                    infos.append(line.strip())

            labels = []
            with open('./torch_ds/nyt/labels.txt','r') as f:
                for line in f:
                    labels.append(int(line.strip()))

            df = pd.DataFrame(zip(infos, labels), columns=['content','label'])

            names = []
            with open('./torch_ds/nyt/classes.txt','r') as f:
                for line in f:
                    names.append(line.strip())
            ixl = {ix:l for ix, l in enumerate(names)}
            df['label_name'] = df['label'].map(lambda x: ixl[x])

            self.df_train, self.df_test = train_test_split(df, test_size=0.3)

        else:
            raise KeyError("dsn illegal!")  

        self.df_train = sample_stratify(self.df_train, self.samplecnt)
        if self.samplecnt_test > 0:
            self.df_test = self.df_test.sample(min(self.df_test.shape[0], self.samplecnt_test))



'''
nyt
business      7639
politics      7182
sports        2048
health        1656
education     1255
estate        1135
arts           840
science        349
technology     293
'''

 

from nltk.tokenize import sent_tokenize
def para_split2(para):
  sents = sent_tokenize(para)
  assert len(sents) > 0 and len(para.split(' ')) >= 4
  if len(sents)==1:
    tokens = para.split(' ')
    paras = [' '.join(tokens[:int(len(tokens)/2)]).strip(), ' '.join(tokens[int(len(tokens)/2):]).strip()]
  else:
    mid = int(len(sents) / 2)
    paras = [' '.join(sents[:mid]).strip(), ' '.join(sents[mid:]).strip()]
  return paras

import datasets
def get_cc_news(s=1):
    cc_news = datasets.load_dataset('cc_news', split="train", cache_dir='./torch_ds')
    '''
    Dataset({
        features: ['date', 'description', 'domain', 'image_url', 'text', 'title', 'url'],
        num_rows: 708241
    })
    '''
    df = pd.DataFrame(zip(cc_news['title'], cc_news['text'], cc_news['description'] ))
    df.columns = ['title','content','description']

    df.drop_duplicates(['title','content'], inplace=True) # 708241
    return df.sample(frac=s) #615019  

def get_cnndm_news(s=1):
    cnndm_news = datasets.load_dataset('cnn_dailymail', '3.0.0', cache_dir='./torch_ds')
    ll = []
    for col in ['train', 'validation', 'test']:
        df_tmp = pd.DataFrame(zip(cnndm_news[col]['article'], cnndm_news[col]['highlights']), \
                    columns=['content', 'title'])
        ll.append(df_tmp)
    df_cnndm = pd.concat(ll)
    df_cnndm.drop_duplicates(['title','content'], inplace=True) # 311971
    #df_cnndm.to_csv('df_cnndm.csv', index=False)
    return df_cnndm.sample(frac=s) #308870  


def get_cc_text_double(ft_pattern, dsn, s=1):
    if dsn == 'cc':
        df_cc = get_cc_news(s)
    elif dsn == 'cnndm':
        df_cc = get_cnndm_news(s)
    if ft_pattern == 'tc':
        df_cc = df_cc.loc[df_cc['title']!='']
        return df_cc.rename(columns={'title': 'text1'}).rename(columns={'content': 'text2'})[['text1','text2']]

    # elif ft_pattern == 'sc':
    #     df_cc = df_cc.loc[df_cc['description']!='']
    #     return df_cc.rename(columns={'description': 'text1'}).rename(columns={'content': 'text2'})[['text1','text2']]
  
    elif ft_pattern == 'pp':
        rr = df_cc['content'].map(lambda x: para_split2(x)).tolist()
        df_cc['text1'] = [c[0] for c in rr]
        df_cc['text2'] = [c[1] for c in rr]
        return df_cc[['text1','text2']]

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



def process_ds(ds, maxlen=256, truncate_testset=False):
    # ds.df_train['content'] = ds.df_train['content']\
    #       .map(lambda x: x.replace('<br />',' '))
    #if not transformers.__version__.startswith('2.'):
    ds.df_train['content'] = ds.df_train['content'].map(lambda x: truncate(x, maxlen))
    if truncate_testset:
        ds.df_test['content'] = ds.df_test['content'].map(lambda x: truncate(x, maxlen))
    proper_len = get_tokens_len(ds, 0.9)
    return ds,  proper_len



base_nli ={
    'politics':['Politics','War', 'Election','Constitution','Democracy','Conflict','Military',\
                'Terrorism', 'Government', 'Ideology', 'fascism', 'Socialism', 'Totalitarian', 'Religion'],
    'law':      ['Law', 'Legitimacy','Court','Crime','Murder','Jurisdiction'],
    'science':  ['Science','Aerospace','Physics','Chemistry','Biology','Scientist','Astronomy','Universe','Big Bang'],
    'technology':['Technology','Biotech', 'IT','Computers','Internet','Algorithm','Space','Bitcoin','artificial Intelligence','Robot'],
    'health': ['Health','Healthcare','Medicine','Clinics','Vaccine','Wellness','Nutrition','Dental','HIV','Disease'],
    'business': ['Business','Finance','Oil price','Supply','Inflation','Dollars','Bank','Wall Street','Bitcoin',
                        'Federal Reserve','Accrual','Accountancy','Sluggishness','Consumerism','Trade','Quarterly earnings',\
                         'Deposit','Revenue','Stocks','Recapitalization','Marketing','Futures'],
    'sports': ['Sports','Athletics','Championships','Football','Olympic','Tournament','Chelsea','League','Golf',
                            'NFL','Super bowl','World Cup'],
    'entertainment':['Entertainment','Pop music','Film','Music','Reality show','Drama','Concert','Rock music','Opera'],
    'education': ['Education', 'Tertiary education', 'University','Curriculum','Lecture'],
    'arts': ['Arts','Music','Painting','Art galleries','Classical music','Art Works','Stitchery'],
    'estate': ['Estate','Estate tax','Real estate']
}
'''
for l, ll in base_nli.items():
    print("{}:{}".format(l, ','.join(ll)))

'''


expand_label_nli = {}

# ag
expand_label_nli['World'] = base_nli['politics'] + base_nli['law']
expand_label_nli['Business'] = base_nli['business']
expand_label_nli['Sports'] = base_nli['sports']

# uci
expand_label_nli['entertainment'] = base_nli['entertainment']
expand_label_nli['business'] = base_nli['business']
expand_label_nli['science and technology'] = base_nli['science'] + base_nli['technology']
expand_label_nli['health'] = base_nli['health']
# nyt
expand_label_nli['education'] = base_nli['education']
expand_label_nli['arts'] = base_nli['arts']
expand_label_nli['politics'] = base_nli['politics']
expand_label_nli['sports'] = base_nli['sports']
expand_label_nli['estate'] = base_nli['estate']
expand_label_nli['science'] = base_nli['science'] 
expand_label_nli['technology'] = base_nli['technology']







