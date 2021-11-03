sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''



import pandas as pd 
infos = []
for file in ['logb_valid']:
    with open(file,'r') as f: 
        for line in f:
            if 'summary===>' not in line:
                continue
            line = line.strip().split('summary===>')[-1] 
            # if 'testvalid:valid' in line:
            #     continue 
            tokens = line.strip().split(' ') 
            dic = {ii.split(':')[0]:ii.split(':')[1] for ii in tokens if ':' in ii}
            if dic['model']!='albert'  or int(dic['testbed'])!=1 or int(dic['samplecnt'])!=128 or int(dic['epochs'])<=10 :
                continue
            infos.append((dic['dsn'], dic['aug'], int(dic['max_aug_times']), dic.get('genm','*'), dic.get('genft', '*'), \
                dic.get('filter', '*'), int(dic.get('valid_files_cnt', -1)), float(dic['acc_aug']), float(dic['gain'].replace('"','')) ))
                # infos.append((dic['dsn'], dic.get('aug','*'), \
                # float(dic['acc_base']), float(dic['acc_aug'])))
df = pd.DataFrame(infos, columns=['dsn','aug', 'max_aug_times', 'genm','genft', 'filter', 
                    'valid_files_cnt', 'acc_aug', 'gain'])

print(df.max_aug_times.value_counts())
max_aug_times_sel = [1]

for dsn in ['ag', 'uci', 'yelp2', 'amazon2']:

    for aug in ['eda','bt','cbert']:
        for max_aug_times in max_aug_times_sel:
            dfi = df.loc[(df['dsn']==dsn) & (df['aug']==aug) &(df['max_aug_times']==max_aug_times)]
            if dfi.shape[0] < 1:
                continue
            print("{:>5} {:>8} {:>1} {:>5}({:>2})".format(dsn, aug, max_aug_times, round(dfi['gain'].mean()*100,2), dfi.shape[0])   )
    
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


###################### test_earlystop ######################


   ag generate 1 gpt no dvrl  0.18( 7)
   ag generate 1  t5 no dvrl -0.05( 7)

  uci generate 1 gpt no dvrl -0.04( 7)
  uci generate 1  t5 no dvrl  0.17( 7)

yelp2 generate 1 gpt no dvrl  0.39( 5)
yelp2 generate 1  t5 no dvrl -0.54( 5)

amazon2 generate 1 gpt no dvrl  0.63( 4)
amazon2 generate 1  t5 no dvrl  0.02( 4)

################ aug_times : 1 ###########
ag      eda 1             -0.01(12)
ag       bt 1             -0.15( 9)
ag generate 1 gpt no   no -0.37( 3)
ag generate 1 gpt no dvrl  0.06(15)  *
ag generate 1  t5 no   no -0.37( 2)
ag generate 1  t5 no dvrl  0.06(14)  *



uci      eda 1              0.03( 6)
uci       bt 1              0.31( 6)
uci generate 1 gpt no   no  0.29( 2)
uci generate 1 gpt no dvrl  0.81(14)  *
uci generate 1  t5 no   no  0.92( 2)
uci generate 1  t5 no dvrl  0.56(14)  *


nyt      eda 1              0.01( 6)
nyt       bt 1             -0.38( 4)
nyt generate 1 gpt no   no  0.33( 2)
nyt generate 1 gpt no dvrl  0.38(10)  *

nyt generate 1  t5 no dvrl  0.21(10)  *







################ aug_times : 4 ###########

 ag      eda 4            -0.45( 4) 
 ag       bt 4            -0.57( 5)
 ag generate 4 gpt no   no -0.71( 3)
ag generate 4 gpt no dvrl  0.07( 8)
ag generate 4  t5 no   no -0.11( 2)
ag generate 4  t5 no dvrl   0.1( 7)   *


uci      eda 4              0.26( 5)
uci       bt 4              0.29( 4)
uci generate 4 gpt no dvrl  0.07( 4)
uci generate 4  t5 no dvrl  0.39( 4)   *


nyt      eda 4              0.36( 3)
nyt       bt 4             -0.47( 3)
nyt generate 4 gpt no dvrl -0.03( 2)
nyt generate 4  t5 no dvrl  0.89( 2)   *
