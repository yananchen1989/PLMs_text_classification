sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''



import pandas as pd 
infos = []
for file in ['log__mc']:
    with open(file,'r') as f: 
        for line in f:
            if 'summary===>' not in line:
                continue
            line = line.strip().split('summary===>')[-1] 
            # if 'testvalid:valid' in line:
            #     continue 
            tokens = line.strip().replace('"','').split(' ') 
            dic = {ii.split(':')[0]:ii.split(':')[1] for ii in tokens if ':' in ii}
            infos.append(dic)


df = pd.DataFrame(infos)


for col in ['samplecnt','dpfuture_switch','dpfuture_cls_switch','candidates','test_beams',\
            'future_steps','threads','valid_files_cnt','nli_switch', 'nsp_switch']:
    if col in df.columns:
        df[col] = df[col].astype('int')

for col in ['acc_base','acc_aug','gain']:
    if col in df.columns:
        df[col] = df[col].astype('float')


df.loc[(df['dsn']=='uci') & (df['samplecnt']==32) & (df['aug']=='generate') \
        & (df['dpfuture_switch']==0) & (df['dpfuture_cls_switch']==0)][['acc_base','acc_aug','gain']]['gain'].mean()




print(df.max_aug_times.value_counts())
max_aug_times_sel = [1]

for dsn in ['ag', 'uci', 'yelp2', 'amazon2']:

    for aug in ['eda','bt','cbert']:
        for max_aug_times in max_aug_times_sel:
            dfi = df.loc[(df['dsn']==dsn) & (df['aug']==aug) &(df['max_aug_times']==max_aug_times)]
            if dfi.shape[0] < 1:
                continue
            print("{:>5} {:>8} {:>1} {:>5}({:>2})".format(dsn, aug, max_aug_times, round(dfi['gain'].mean(),2), dfi.shape[0])   )
    
    for aug in ['generate']:
        for max_aug_times in max_aug_times_sel:
            for genm in ['gpt', 't5']:
                for genft in ['no']: #  'pp','tc', 'ep'
                    for fil in ['no', 'dvrl', 'nli', 'cls', 'enc', 'nsp']:
                        dfi = df.loc[(df['dsn']==dsn) & (df['aug']==aug) & (df['max_aug_times']==max_aug_times) \
                          & (df['genm']==genm) & (df['genft']==genft) & (df['filter']==fil) ]
                        if dfi.shape[0] == 0:
                            continue 
                        print("{:>5} {:>8} {:>1} {:>3} {:>2} {:>4} {:>5}({:>2})"\
                            .format(dsn, aug, max_aug_times, genm, genft, fil, round(dfi['gain'].mean(),2), dfi.shape[0])   )
    print('\n\n')






import glob, os 

files = sorted(glob.glob('*.log'), key=os.path.getmtime)

for file in files:
    with open(file, 'r') as f:
        fail = 0 
        for line in f:
            if 'RuntimeError: CUDA out of memory' in line or 'ERROR' in line :
                print(line)
                #print("CUDA out of memory==>")
                fail += 1
    if fail > 0:
        print(file,'\n')


#################### test_valid ############################
   ag generate 1 gpt no   no -0.73( 4)
   ag generate 1 gpt no dvrl  0.34( 4)
   ag generate 1  t5 no   no -0.52( 4)
   ag generate 1  t5 no dvrl  0.69( 4)



  uci generate 1 gpt no   no  0.89( 4)
  uci generate 1 gpt no dvrl  0.99( 4)
  uci generate 1  t5 no   no  0.94( 4)
  uci generate 1  t5 no dvrl  1.14( 4)



yelp2 generate 1 gpt no   no -2.27( 4)
yelp2 generate 1 gpt no dvrl -0.17( 4)
yelp2 generate 1  t5 no   no  0.57( 4)
yelp2 generate 1  t5 no dvrl -0.08( 4)



amazon2 generate 1 gpt no   no -0.93( 3)
amazon2 generate 1 gpt no dvrl  0.15( 3)
amazon2 generate 1  t5 no dvrl  1.91( 1)

