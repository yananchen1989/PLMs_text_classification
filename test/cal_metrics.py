sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''





import glob
error_files = []
folder = "log_baselines"
files = glob.glob("./{}/*.log".format(folder))
for file in files:
    with open(file,'r') as f: 
        for line in f:
            if "SyntaxError: invalid syntax" in line:
                error_files.append(file)
                break 

import os 
for file in error_files:
    os.system("rm {}".format(file))

import pandas as pd
import glob 
infos = []
#folder = "log_arxiv_nlinsp"
for folder in ["log_baselines", "log_arxiv_nlinsp"]:
    files = glob.glob("./{}/*.log".format(folder))
    for file in files:
        with open(file,'r') as f: 
            for line in f:
                if 'success summary===>' in line:
                    print(line) 
                    line = line.strip().split('summary===>')[-1] 
                    # if 'testvalid:valid' in line:
                    #     continue 
                    tokens = line.strip().replace('"','').split(' ') 
                    dic = {ii.split(':')[0]:ii.split(':')[1] for ii in tokens if ':' in ii}
                    infos.append(dic)
df = pd.DataFrame(infos)


for col in ['samplecnt','candidates','max_aug_times','candidates']:
    if col in df.columns:
        print(col)
        df[col] = df[col].astype('int')

for col in ['acc_base','acc_aug','gain']:
    if col in df.columns:
        df[col] = df[col].astype('float')


samplecnt = 128
for dsn in ['uci','ag','nyt']:
    # baselines : eda uci cbert
    for aug in ['eda', 'bt', 'cbert']:
        dfi = df.loc[(df['dsn']==dsn) & (df['samplecnt']==samplecnt) & (df['aug']==aug)][['acc_base','acc_aug','gain']] # & (df['candidates']==candidates)
        print(dsn, aug, round(dfi['acc_base'].mean(),4), round(dfi['acc_aug'].mean(),4), \
            round(dfi['gain'].mean(),4), round(dfi['gain'].std(),4) , dfi.shape[0])

    # baselines: lambda embed
    for fmark in ['cls', 'embed']:
        dfi = df.loc[(df['dsn']==dsn) & (df['samplecnt']==samplecnt) & (df['aug']=='generate') & (df['genm']=='gpt') \
                 & (df['fmark']==fmark) ][['acc_base','acc_aug','gain']] # & (df['candidates']==candidates)
        print(dsn, 'gpt-{}'.format(fmark), round(dfi['acc_base'].mean(),4), round(dfi['acc_aug'].mean(),4), \
            round(dfi['gain'].mean(),4), round(dfi['gain'].std(),4), dfi.shape[0])

    #nli nsp
    for genm in ['gpt', 't5']:
        #for candidates in df['candidates'].unique():
        for fmark in ['11', '10', '01', '00']:
            dfi = df.loc[(df['dsn']==dsn) & (df['samplecnt']==samplecnt) & (df['aug']=='generate') & (df['genm']==genm) \
                     & (df['fmark']==fmark) ][['acc_base','acc_aug','gain']] # & (df['candidates']==candidates)
            print(dsn, "{}-{}".format(genm, fmark), round(dfi['acc_base'].mean(),4), round(dfi['acc_aug'].mean(),4), \
                round(dfi['gain'].mean(),4), round(dfi['gain'].std(),4),  dfi.shape[0])
    print()




'''
uci gpt 11 0.7674 0.7908 3.0917 18
uci gpt 10 0.7679 0.7894 2.8359 17
uci gpt 01 0.7679 0.7751 0.9641 17
uci gpt 00 0.7679 0.772 0.5635 17

uci t5 11 0.7717 0.7942 2.9487 15
uci t5 10 0.7717 0.7947 3.0067 15
uci t5 01 0.7717 0.7873 2.0473 15
uci t5 00 0.7717 0.7871 2.0173 15


ag gpt 11 0.8526 0.8629 1.224 20
ag gpt 10 0.8526 0.8625 1.1685 20
ag gpt 01 0.8526 0.8572 0.5435 20
ag gpt 00 0.8526 0.8581 0.6495 20

ag t5 11 0.846 0.8573 1.35 13
ag t5 10 0.846 0.8557 1.1477 13
ag t5 01 0.846 0.8573 1.3408 13
ag t5 00 0.846 0.8569 1.2885 13


nyt gpt 11 0.817 0.8183 0.18 4
nyt gpt 10 0.817 0.8192 0.29 4
nyt gpt 01 0.817 0.803 -1.705 4
nyt gpt 00 0.817 0.8174 0.0425 4

nyt t5 11 0.8275 0.841 1.6575 4
nyt t5 10 0.8275 0.8533 3.13 4
nyt t5 01 0.8275 0.8256 -0.205 4
nyt t5 00 0.8275 0.8343 0.85 4
'''



