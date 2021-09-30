sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''



import pandas as pd 
infos = []
for file in ['logb_','logb']:
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
                dic.get('filter', '*'), float(dic['acc_aug'])))
                # infos.append((dic['dsn'], dic.get('aug','*'), \
                # float(dic['acc_base']), float(dic['acc_aug'])))
df = pd.DataFrame(infos, columns=['dsn','genm','genft', 'filter','acc_aug'])
#df = pd.DataFrame(infos, columns=['dsn','aug','acc_base','acc_aug'])



# baselines 
for dsn in ['ag', 'uci', 'nyt']:
    acc_base = df.loc[df['dsn']==dsn]['acc_base'].mean()
    print(dsn, 'acc_base', df.loc[df['dsn']==dsn].shape[0], acc_base)
    for aug in ['eda','bt','cbert']:
        dfi = df.loc[(df['dsn']==dsn) & (df['aug']==aug)]
        print(dsn, aug, "{}({})".format(round(dfi['acc_aug'].mean()*100,2), dfi.shape[0])   )
    print()


# no finetune
for dsn in ['ag', 'uci']:
    for genm in ['t5', 'gpt', 'ctrl']:
        for genft in ['no']:
            for fil in ['no','nli']: 
                dfi = df.loc[(df['dsn']==dsn) & (df['genm']==genm) & (df['genft']==genft) & (df['filter']==fil)]
                print(dsn, genm, genft, fil, "{}({})".format(round(dfi['acc_aug'].mean()*100,2) , dfi.shape[0] ) )
    print('\n')

#internal finetune
for dsn in ['ag', 'uci', 'nyt']:
    for genm in [ 'gpt']:
        for genft in ['lambda', 'entire']:
            for fil in ['no','nli']: 
                dfi = df.loc[(df['dsn']==dsn) & (df['genm']==genm) & (df['genft']==genft) & (df['filter']==fil)]
                print(dsn, genm, genft, fil, "{}({})".format(round(dfi['acc_aug'].mean()*100,2) , dfi.shape[0] ) )
    print('\n')



#external finetune
for dsn in ['ag', 'uci', 'nyt']:
    for genm in ['t5', 'gpt', 'ctrl']:
        for genft in ['no','pp','tc','entire', 'lambda']:
            for fil in ['no','nli']: 
                dfi = df.loc[(df['dsn']==dsn) & (df['genm']==genm) & (df['genft']==genft) & (df['filter']==fil)]
                print(dsn, genm, genft, fil, "{}({})".format(round(dfi['acc_aug'].mean()*100,2) , dfi.shape[0] ) )
    print('\n')

################################## get acc no aug ######################
import GPUtil
DEVICE_ID_LIST = GPUtil.getFirstAvailable()
DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
from utils.load_data import * 
from utils.transblock import * 
print("use gpu id==>", DEVICE_ID)
for _ in range(5):
    for dsn in ['ag', 'uci','nyt']:
        print(dsn)
        ds = load_data(dataset=dsn, samplecnt= 256)
        acc_noaug, _ = do_train_test(ds.df_train, ds.df_test, 100, 10, 0, \
                   3, 256, 'max', 'albert')
        print(acc_noaug)




for ix, row in ds.df_train.sample(frac=1).iterrows():

    contents_trunk_ = gen_nlp(row['content'], max_length=256, do_sample=True, top_p=0.9, top_k=0, temperature=1,\
        repetition_penalty=1.0, num_return_sequences=8, clean_up_tokenization_spaces=True) 

    pairs = [[row['content'], ii['generated_text']] for ii in contents_trunk_]
    #pairs = [["This is {} News".format(row['label_name']), ii['generated_text']] for ii in contents_trunk_]
    pairs_ids = get_ids(pairs,  512)
    print(pairs_ids)
    content_syn = [ii['generated_text'] for ii in contents_trunk_]

    preds = model_cls_pair.predict(pairs_ids, batch_size=64)
    if preds[:,0].min() < 0.3:    
        scores = preds[:,0]
        df_nsp = pd.DataFrame(zip(content_syn, list(scores)), columns=['content','nsp_score'])
        df_nsp.sort_values(by=['nsp_score'], ascending=False, inplace=True)
        print(row['label_name'])
        print(df_nsp)


        print(df_nsp['nsp_score'].tolist()[-1])
        print(df_nsp['content'].tolist()[-1]) 
        break 




























