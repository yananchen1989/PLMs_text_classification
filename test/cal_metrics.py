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
            if 'nli_check' in dic.keys():
                if dic['nli_check'] == '0':
                    dic.update({'filter':'no'})
                elif dic['nli_check'] == '1':
                    dic.update({'filter':'nli'})

            if dic.get('genft', '*') not in ['no','pp','tc','entire', 'lambda']:
                continue

            if int(dic['samplecnt'])==128 and dic['model']=='albert' and  int(dic['max_aug_times'])==1 \
                and dic['aug']=='generate':
                infos.append((dic['dsn'], dic.get('genm','*'), dic.get('genft', '*'), \
                dic.get('filter', '*'), int(dic.get('threads', 0)), float(dic['acc_aug'])))
                # infos.append((dic['dsn'], dic.get('aug','*'), \
                # float(dic['acc_base']), float(dic['acc_aug'])))
df = pd.DataFrame(infos, columns=['dsn','genm','genft', 'filter', 'threads', 'acc_aug'])
#df = pd.DataFrame(infos, columns=['dsn','aug','acc_base','acc_aug'])



# baselines 
for dsn in ['ag', 'uci', 'nyt']:
    acc_base = df.loc[df['dsn']==dsn]['acc_base'].mean()
    print(dsn, 'acc_base', df.loc[df['dsn']==dsn].shape[0], acc_base)
    for aug in ['eda','bt','cbert']:
        dfi = df.loc[(df['dsn']==dsn) & (df['aug']==aug)]
        print(dsn, aug, "{}({})".format(round(dfi['acc_aug'].mean()*100,2), dfi.shape[0])   )
    print()

filters = ['no','nli', 'nsp', 'enc','cls','dvrl']

# no finetune
for dsn in ['ag', 'uci', 'nyt']:
    for genm in ['t5', 'gpt']:
        for genft in ['no']:
            for fil in filters: 
                dfi = df.loc[(df['dsn']==dsn) & (df['genm']==genm) & (df['genft']==genft) & (df['filter']==fil)]
                print(dsn, genm, genft, fil, "{}({})".format(round(dfi['acc_aug'].mean()*100,2) , dfi.shape[0] ) )
    print('\n')

#internal finetune
for dsn in ['ag', 'uci', 'nyt']:
    for genm in [ 'gpt']:
        for genft in ['lambda', 'entire']:
            for fil in filters: 
                dfi = df.loc[(df['dsn']==dsn) & (df['genm']==genm) & (df['genft']==genft) & (df['filter']==fil)]
                print(dsn, genm, genft, fil, "{}({})".format(round(dfi['acc_aug'].mean()*100,2) , dfi.shape[0] ) )
    print('\n')



#external finetune
for dsn in ['ag', 'uci', 'nyt']:
    for genm in ['t5', 'gpt']:
        for genft in ['pp','tc']:
            for fil in filters: 
                dfi = df.loc[(df['dsn']==dsn) & (df['genm']==genm) & (df['genft']==genft) & (df['filter']==fil)]
                print(dsn, genm, genft, fil, "{}({})".format(round(dfi['acc_aug'].mean()*100,2) , dfi.shape[0] ) )
    print('\n')





























