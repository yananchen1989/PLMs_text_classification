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
files = glob.glob("nyt.*.*.*.log")

infos = []
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


for samplecnt in [32, 64]:
    for candidates in [64, 256]:
        
        dfi = df.loc[(df['samplecnt']==samplecnt) & (df['candidates']==candidates)]
        if dfi.shape[0] == 0:
            continue
        print(samplecnt, candidates)    
        print("acc_base:", dfi['acc_base'].mean())
        for fmark in dfi['fmark'].unique():
            dfii = dfi.loc[dfi['fmark']==fmark]
            print(fmark, dfii.shape[0], dfii['acc_aug'].mean())
        print()





args.samplecnt=256
args.dsn = 'ag'
ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))
ds, proper_len = process_ds(ds, 128)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))

ixl = {ii[0]:ii[1] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
ixl_rev = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}

with tf.distribute.MirroredStrategy().scope():
    model_cls = get_model_bert(ds.df_test.label.unique().shape[0])
model_cls.load_weights("./model_cls/model_full_{}.h5".format(args.dsn))  


x_train, y_train = get_keras_data(ds.df_test)

eval_result = model_cls.evaluate(x_train, y_train, batch_size=128)
print(eval_result)





for samplecnt in [32, 64, 128]:
    for candidates in [64, 256]:
        files = glob.glob("./log_arxiv_clsembednlinsp/{}.{}.{}.*.log".format(args.dsn, samplecnt, candidates))
        if not files:
            continue
        infos = []
        for file in files:

            lines = []
            with open(file,'r') as f: 
                for line in f:
                    if not line.strip():
                        continue
                    if 'ori====>' in line or 'ft-nlisp ==>' in line or 'ft-nli ==>' in line or 'ft-nsp ==>' in line or \
                        'ft-cls ==>' in line or 'ft-embed ==>' in line or 'nofil ==>' in line:
                        print(line)
                        lines.append(line.strip())

            cutix = []
            for ix, line in enumerate(lines):
                if 'ori====>' in line:
                    cutix.append(ix)

            
            for i in range(len(cutix)-1):
                assert cutix[i+1] - cutix[i] == 13
                lines_ = lines[cutix[i]:cutix[i+1]]
                 
                ori_content = lines_[0].split('ori====>')[-1].strip().split('<===')[0].strip()
                label = lines_[0].split('ori====>')[-1].strip().split('<===')[1].strip()
                infos.append((ori_content, 'ori',label))
                for line in lines_[1:]:
                    fmark = line.split('==>')[0].strip()
                    content = line.split('==>')[1].strip()
                    infos.append((content, fmark, label))

            df_tmp = pd.DataFrame(infos, columns=['content','fmark','label_name'])

            df_tmp['label'] = df_tmp['label_name'].map(lambda x: ixl_rev[x])


        for fmark in  df_tmp['fmark'].unique():       
            x_train, y_train = get_keras_data(df_tmp.loc[df_tmp['fmark']==fmark])
            eval_result = model_cls.evaluate(x_train, y_train, batch_size=512, verbose=0)
            print(samplecnt, candidates, fmark, x_train.shape[0], eval_result[0], eval_result[1])



# ag ori: [0.13802948594093323, 0.9619140625]
# uci ori: [0.09009121358394623, 0.96875]











