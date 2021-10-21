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
            tokens = line.strip().split(' ') 
            dic = {ii.split(':')[0]:ii.split(':')[1] for ii in tokens}
            if dic['model']!='albert' or dic['testvalid']!='valid' or int(dic['testbed'])!=1 \
                or int(dic['samplecnt'])!=128:
                continue
            infos.append((dic['dsn'], dic['aug'], int(dic['max_aug_times']), dic.get('genm','*'), dic.get('genft', '*'), \
                dic.get('filter', '*'), int(dic.get('valid_files_cnt', -1)), float(dic['acc_aug']), float(dic['gain'].replace('"','')) ))
                # infos.append((dic['dsn'], dic.get('aug','*'), \
                # float(dic['acc_base']), float(dic['acc_aug'])))
df = pd.DataFrame(infos, columns=['dsn','aug', 'max_aug_times', 'genm','genft', 'filter', 
                    'valid_files_cnt', 'acc_aug', 'gain'])


max_aug_times_sel = [3]

for dsn in ['ag', 'uci', 'nyt']:
    for aug in ['eda','bt','cbert']:
        for max_aug_times in max_aug_times_sel:
            dfi = df.loc[(df['dsn']==dsn) & (df['aug']==aug) &(df['max_aug_times']==max_aug_times)]
            if dfi.shape[0] < 3:
                continue
            print("{:>3} {:>8} {:>1} {:>4}({:>2})".format(dsn, aug, max_aug_times, round(dfi['gain'].mean()*100,2), dfi.shape[0])   )
    
    for aug in ['generate']:
        for max_aug_times in max_aug_times_sel:
            for genm in ['gpt', 't5']:
                for genft in ['no', 'pp','tc', 'ep']:
                    for fil in ['dvrl', 'nli,cls,enc,nsp']:
                        dfi = df.loc[(df['dsn']==dsn) & (df['aug']==aug) & (df['max_aug_times']==max_aug_times) \
                          & (df['genm']==genm) & (df['genft']==genft) & (df['filter']==fil) ]
                        if dfi.shape[0] < 1:
                            continue 
                        print("{:>3} {:>8} {:>1} {:>3} {:>2} {:>4} {:>4}({:>2})"\
                            .format(dsn, aug, max_aug_times, genm, genft, fil, round(dfi['gain'].mean()*100,2), dfi.shape[0])   )
    print('\n\n')


'''
ag eda 1 -0.03(7)
ag eda 3 0.4(5)
ag bt 1 1.31(3)

uci eda 1 1.35(6)
uci eda 3 1.4(4)
uci bt 1 1.16(6)
uci bt 3 0.83(3)

nyt eda 1 0.7(5)
nyt eda 3 0.03(4)
nyt bt 1 0.79(3)
'''





import glob 
files = glob.glob("*.log")
for file in files:
    with open(file, 'r') as f:
        success = 0 
        for line in f:
            if 'success summary===>' in line:
                break
            if 'RuntimeError' in line:
                print("RuntimeError==>", file)
                break 
            if 'error' in line.lower() and 'filtering' not in line.lower() \
                and 'CUDA_ERROR_NO_DEVICE' not in line and 'final_sample' not in line \
                and 'Running this sequence through the model will result in indexing errors' not in line :
                print(line)
                print("error==>", file)
                break 




ag.generate.128.max_aug_times.1.genm.t5.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.31781.log
ag.generate.128.max_aug_times.1.genm.t5.genft.pp.filter.dvrl.abundance.3.num_return_sequences.8.17132.log

ag.generate.128.max_aug_times.1.genm.t5.genft.ep.filter.dvrl.abundance.3.num_return_sequences.8.17132.log
ag.generate.128.max_aug_times.4.genm.gpt.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.8811.log
ag.generate.128.max_aug_times.1.genm.t5.genft.pp.filter.dvrl.abundance.3.num_return_sequences.8.31781.log
ag.generate.128.max_aug_times.4.genm.gpt.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.797.log
ag.generate.128.max_aug_times.4.genm.gpt.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.21043.log
ag.generate.128.max_aug_times.1.genm.t5.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.17132.log
nyt.generate.128.max_aug_times.1.genm.gpt.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.23102.log
ag.generate.128.max_aug_times.4.genm.gpt.genft.tc.filter.dvrl.abundance.3.num_return_sequences.8.29871.log
uci.generate.128.max_aug_times.1.genm.gpt.genft.no.filter.dvrl.abundance.3.num_return_sequences.8.31781.log
ag.generate.128.max_aug_times.1.genm.t5.genft.no.filter.dvrl.abundance.3.num_return_sequences.8.558.log













