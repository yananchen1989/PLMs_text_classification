sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''



import pandas as pd
import glob 
infos = []
#folder = "log_arxiv_nlinsp"
folder = "log_baselines"
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


for col in ['samplecnt','candidates','max_aug_times','candidates',\
            'threads','valid_files_cnt']:
    if col in df.columns:
        print(col)
        df[col] = df[col].astype('int')

for col in ['acc_base','acc_aug','gain']:
    if col in df.columns:
        df[col] = df[col].astype('float')


for dsn in ['uci','ag']:
    for genm in ['gpt', 't5']:
        #for candidates in df['candidates'].unique():

        for fmark in ['11', '10', '01', '00']:
            dfi = df.loc[(df['dsn']==dsn) & (df['samplecnt']==32) & (df['aug']=='generate') & (df['genm']==genm) \
                     & (df['fmark']==fmark) ][['acc_base','acc_aug','gain']] # & (df['candidates']==candidates)
            print(dsn, genm, fmark,  round(dfi['acc_base'].mean(),4), round(dfi['acc_aug'].mean(),4), \
                round(dfi['gain'].mean(),4) , dfi.shape[0])
    print('\n')






