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
#folder = "log_arxiv_nlinsp" "log_baselines", log_arxiv_nlinsp_noner
for folder in [ "log_arxiv_nlinsp"]:
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


for samplecnt in [32, 64, 128]:
    print("samplecnt:", samplecnt)
    for dsn in ['uci','ag','nyt']:
        # # baselines : eda uci cbert
        # for aug in ['eda', 'bt', 'cbert']:
        #     dfi = df.loc[(df['dsn']==dsn) & (df['samplecnt']==samplecnt) & (df['aug']==aug)][['acc_base','acc_aug','gain']] # & (df['candidates']==candidates)
        #     print(dsn, aug, round(dfi['acc_base'].mean(),4), round(dfi['acc_aug'].mean(),4), \
        #         round(dfi['gain'].mean(),4), round(dfi['gain'].std(),4) , dfi.shape[0])

        # # baselines: lambda embed
        # for fmark in ['cls', 'embed']:
        #     dfi = df.loc[(df['dsn']==dsn) & (df['samplecnt']==samplecnt) & (df['aug']=='generate') & (df['genm']=='gpt') \
        #              & (df['fmark']==fmark) ][['acc_base','acc_aug','gain']] # & (df['candidates']==candidates)
        #     print(dsn, 'gpt-{}'.format(fmark), round(dfi['acc_base'].mean(),4), round(dfi['acc_aug'].mean(),4), \
        #         round(dfi['gain'].mean(),4), round(dfi['gain'].std(),4), dfi.shape[0])

        #nli nsp
        for genm in ['gpt', 't5']:
            for candidates in [   256]:
                for fmark in ['11', '10', '01', '00']:
                    dfi = df.loc[(df['dsn']==dsn) & (df['samplecnt']==samplecnt) & (df['aug']=='generate') & (df['genm']==genm) \
                             & (df['fmark']==fmark)  & (df['candidates']==candidates) ][['acc_base','acc_aug','gain']] 
                    if dfi.shape[0] == 0:
                        continue
                    print(dsn, candidates, "{}-{}".format(genm, fmark), round(dfi['acc_base'].mean(),4), round(dfi['acc_aug'].mean(),4), \
                          dfi.shape[0]) # round(dfi['gain'].mean(),4), round(dfi['gain'].std(),4),
        print()



import glob
import pandas as pd 
files = glob.glob("validate_repeat.ag.*.log")

infos = []
for file in files:
    with open(file,'r') as f: 
        for line in f:
            if 'summary==' in line:
                #print(line)  
                if 'keras.engine.functional.Functional' in line:
                    tokens = line.strip().replace('keras.engine.functional.Functional','')\
                            .replace('(','').replace(')','').replace('<','').replace('>','').replace(',','').split(' ')
                    samplecnt = int(tokens[2])
                    candidates = int(tokens[3])
                    fmark = tokens[4]
                    acc_base = float(tokens[5])
                    acc_aug = float(tokens[10])
                else:

                    tokens = line.strip().split('summary==')[-1].split(' ')
                    samplecnt = int(tokens[2])
                    candidates = int(tokens[3])
                    fmark = tokens[4]
                    acc_base = float(tokens[5])
                    acc_aug = float(tokens[6])
                infos.append((samplecnt, candidates, fmark, acc_base, acc_aug))

df = pd.DataFrame(infos, columns=['samplecnt','candidates','fmark','acc_base','acc_aug'])




for samplecnt in df['samplecnt'].unique():
    for candidates in df['candidates'].unique():
        for fmark in df['fmark'].unique():
            dfi = df.loc[(df['fmark']==fmark) & (df['samplecnt']==samplecnt) & (df['candidates']==candidates)]
            print(samplecnt, candidates, fmark, dfi.shape[0], dfi['acc_base'].mean(), dfi['acc_aug'].mean()) 
























