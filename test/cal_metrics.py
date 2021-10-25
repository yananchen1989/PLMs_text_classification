sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''



import pandas as pd 
infos = []
for file in ['logb']:
    with open(file,'r') as f: 
        for line in f:
            line = line.strip().split('summary===>')[-1] 
            if 'testvalid:valid' in line:
                continue 
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
max_aug_times_sel = [4]

for dsn in ['ag', 'uci', 'nyt']:
    for aug in ['eda','bt','cbert']:
        for max_aug_times in max_aug_times_sel:
            dfi = df.loc[(df['dsn']==dsn) & (df['aug']==aug) &(df['max_aug_times']==max_aug_times)]
            if dfi.shape[0] < 1:
                continue
            print("{:>5} {:>8} {:>1} {:>5}({:>2})".format(dsn, aug, max_aug_times, round(dfi['gain'].mean()*100,2), dfi.shape[0])   )
    
    for aug in ['generate']:
        for max_aug_times in max_aug_times_sel:
            for genm in ['gpt', 't5']:
                for genft in ['no', 'pp','tc', 'ep']:
                    for fil in ['no', 'dvrl', 'nli', 'cls', 'enc', 'nsp']:
                        dfi = df.loc[(df['dsn']==dsn) & (df['aug']==aug) & (df['max_aug_times']==max_aug_times) \
                          & (df['genm']==genm) & (df['genft']==genft) & (df['filter']==fil) ]
                        if dfi.shape[0] == 0:
                            continue 
                        print("{:>5} {:>8} {:>1} {:>3} {:>2} {:>4} {:>5}({:>2})"\
                            .format(dsn, aug, max_aug_times, genm, genft, fil, round(dfi['gain'].mean()*100,2), dfi.shape[0])   )
    print('\n\n')






import glob, os 

files = sorted(glob.glob('*.log'), key=os.path.getmtime)

for file in files:
    with open(file, 'r') as f:
        fail = 0 
        for line in f:
            if 'RuntimeError: CUDA out of memory' in line:
                print("CUDA out of memory==>")
                fail += 1
    if fail > 0:
        print(file,'\n')


RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.1.genm.t5.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.26257.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.1.genm.t5.genft.pp.filter.dvrl.abundance.3.num_return_sequences.8.26257.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.1.genm.t5.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.2728.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.1.genm.t5.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.18371.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.1.genm.t5.genft.pp.filter.dvrl.abundance.3.num_return_sequences.8.18371.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.1.genm.t5.genft.pp.filter.dvrl.abundance.3.num_return_sequences.8.2728.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.4.genm.t5.genft.no.filter.dvrl.abundance.3.num_return_sequences.8.18034.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.4.genm.t5.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.18034.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.4.genm.t5.genft.pp.filter.dvrl.abundance.3.num_return_sequences.8.18034.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.4.genm.t5.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.3970.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.4.genm.t5.genft.pp.filter.dvrl.abundance.3.num_return_sequences.8.3970.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.4.genm.t5.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.11640.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.4.genm.t5.genft.ep.filter.dvrl.abundance.3.num_return_sequences.8.3970.log

RuntimeError==>
CUDA out of memory==>
ag.generate.128.max_aug_times.4.genm.t5.genft.pp.filter.dvrl.abundance.3.num_return_sequences.8.11640.log


#################### test_valid ############################
    ag      eda 1              0.01(15)
    ag       bt 1              0.70(11)

    ag generate 1 gpt no dvrl  0.71(10)
    ag generate 1 gpt pp dvrl  0.28( 5)
    ag generate 1 gpt tc dvrl  0.62( 5)
    ag generate 1 gpt ep dvrl -0.08( 5)
    ag generate 1  t5 no dvrl   0.6(10)
    ag generate 1  t5 tc dvrl  1.03( 1)
    ag generate 1  t5 ep dvrl  0.22( 3)


    uci      eda 1              1.46(15)
    uci       bt 1              0.88(15)

    uci generate 1 gpt no dvrl  2.44( 8)
    uci generate 1 gpt pp dvrl  -0.3( 4)
    uci generate 1 gpt tc dvrl  1.05( 4)
    uci generate 1 gpt ep dvrl  0.89( 4)
    uci generate 1  t5 no dvrl  2.13( 9)
    uci generate 1  t5 pp dvrl  0.96( 1)
    uci generate 1  t5 tc dvrl  2.14( 2)


    nyt      eda 1              0.89(14)
    nyt       bt 1              0.69(12)

    nyt generate 1 gpt no dvrl -0.23( 3)
    nyt generate 1 gpt pp dvrl  0.46( 1)
    nyt generate 1  t5 no dvrl  0.24( 4)


    ##########################


    ag      eda 3              0.35(10)
    ag       bt 3              0.12( 8)
    ag generate 3 gpt no dvrl  1.75( 2)
    ag generate 3 gpt pp dvrl -0.05( 1)
    ag generate 3 gpt tc dvrl  1.01( 1)
    ag generate 3 gpt ep dvrl   0.9( 1)
    ag generate 3  t5 no dvrl  1.04( 3)


    uci      eda 3              1.86(10)
    uci       bt 3               1.1( 7)
    uci generate 3 gpt no dvrl   3.0( 1)
    uci generate 3  t5 no dvrl  1.13( 3)


    nyt      eda 3              0.17( 8)
    nyt       bt 3             -0.18( 6)
    nyt generate 3 gpt no dvrl  0.27( 1)


    ###########################

    ag generate 4 gpt no dvrl  1.52( 4)
    ag generate 4 gpt pp dvrl   1.4( 4)
    ag generate 4 gpt tc dvrl  0.49( 4)
    ag generate 4 gpt ep dvrl  0.67( 4)
    ag generate 4  t5 no dvrl  0.05( 1)


    #########################
    ag generate 5 gpt no dvrl  2.18( 1)
    ag generate 5  t5 no dvrl -0.79( 1)

    uci generate 5 gpt no dvrl  2.18( 1)
    uci generate 5  t5 no dvrl  1.18( 1)




###################### test_earlystop ######################


################ aug_times : 1 ###########
 ag      eda 1 -0.01(12)
 ag       bt 1 -0.15( 9)
 ag generate 1 gpt no   no -0.31( 3)
 ag generate 1 gpt no  nli  0.12( 3)
 ag generate 1 gpt no  cls -0.01( 3)
 ag generate 1 gpt no  enc  0.03( 3)
 ag generate 1  t5 no   no  0.12( 3)
 ag generate 1  t5 no  nli  0.24( 3)
 ag generate 1  t5 no  cls  0.12( 3)
 ag generate 1  t5 no  enc -0.04( 3)

 ag generate 1 gpt no dvrl  0.19( 4)
 ag generate 1 gpt pp dvrl  0.47( 4) *
 ag generate 1 gpt tc dvrl  0.14( 4)
 ag generate 1 gpt ep dvrl   0.1( 4)
 ag generate 1  t5 no dvrl -0.11( 4)
 ag generate 1  t5 tc dvrl  0.46( 1) * 
 ag generate 1  t5 ep dvrl  0.21( 4)


uci      eda 1  0.03( 6)
uci       bt 1  0.31( 6)
uci generate 1 gpt no   no -0.12( 2)
uci generate 1 gpt no  nli  2.65( 2)
uci generate 1 gpt no  cls  1.26( 2)
uci generate 1 gpt no  enc   0.9( 2)
uci generate 1  t5 no   no -0.32( 2)
uci generate 1  t5 no  nli  1.04( 2)
uci generate 1  t5 no  cls -0.43( 2)
uci generate 1  t5 no  enc  1.25( 2)

uci generate 1 gpt no dvrl  0.83( 4)
uci generate 1 gpt pp dvrl  0.94( 4)
uci generate 1 gpt tc dvrl -0.16( 4)
uci generate 1 gpt ep dvrl  0.96( 4)
uci generate 1  t5 no dvrl  0.85( 4)
uci generate 1  t5 pp dvrl   0.7( 2)
uci generate 1  t5 tc dvrl -0.35( 3)

nyt      eda 1  0.01( 6)
nyt       bt 1 -0.38( 4)
nyt generate 1 gpt no   no  0.14( 2)
nyt generate 1 gpt no  nli  1.18( 2)
nyt generate 1 gpt no  cls  0.37( 2)
nyt generate 1 gpt no  enc  0.46( 2)
nyt generate 1  t5 no   no -0.75( 1)
nyt generate 1  t5 no  nli   1.6( 1)
nyt generate 1  t5 no  cls  1.84( 1)
nyt generate 1  t5 no  enc -2.98( 1)

################ aug_times : 4 ###########

     ag      eda 4 -0.45( 4)
     ag       bt 4 -0.57( 5)
     ag generate 4 gpt no   no -0.05( 2)
     ag generate 4 gpt no  nli -0.98( 1)
     ag generate 4 gpt no  cls -0.09( 2)
     ag generate 4 gpt no  enc  0.05( 2)
     ag generate 4  t5 no   no -0.16( 2)
     ag generate 4  t5 no  nli  0.32( 2)
     ag generate 4  t5 no  cls -0.19( 2)
     ag generate 4  t5 no  enc  0.06( 2)
     ag generate 4 gpt no dvrl  0.24( 4)
     ag generate 4 gpt pp dvrl -0.09( 4)
     ag generate 4 gpt tc dvrl -0.21( 4)
     ag generate 4 gpt ep dvrl  0.32( 4)
     ag generate 4  t5 no dvrl  0.19( 3)
     ag generate 4  t5 ep dvrl  0.36( 1)


uci      eda 4  0.26( 5)
uci       bt 4  0.29( 4)
uci generate 4 gpt no   no  0.46( 2)
uci generate 4 gpt no  nli  0.01( 2)
uci generate 4 gpt no  cls  1.31( 1)
uci generate 4 gpt no  enc  2.37( 1)
uci generate 4  t5 no   no -0.85( 1)
uci generate 4  t5 no  nli  1.06( 1)
uci generate 4  t5 no  cls  1.58( 1)
uci generate 4  t5 no  enc -0.06( 1)



nyt      eda 4  0.36( 3)
nyt       bt 4 -0.47( 3)
nyt generate 4 gpt no   no  0.95( 1)
