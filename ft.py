import pandas as pd 
import glob
from sklearn.model_selection import train_test_split
from utils.load_data import * 



df = get_cc_text_double('ss')


df_cc_train, df_cc_test =  train_test_split(df_text2text, test_size=0.05)
 # all: 614805 
print(df_cc_train.shape[0], df_cc_test.shape[0])


df_cc_train.to_csv("/scratch/w/wluyliu/yananc/finetunes/df_cc_train_ss.csv", index=False)
df_cc_test.to_csv("/scratch/w/wluyliu/yananc/finetunes/df_cc_test_ss.csv", index=False)







PATH_HOME = "/home/w/wluyliu/yananc/topic_classification_augmentation"
ds_yelp2 = load_data(dataset='yelp2', samplecnt= -1, path='{}/torch_ds'.format(PATH_HOME))
ds_amzon2 = load_data(dataset='amazon2', samplecnt= -1, path='{}/torch_ds'.format(PATH_HOME))
ds_yelp5 = load_data(dataset='yelp5', samplecnt= -1, path='{}/torch_ds'.format(PATH_HOME))
ds_amzon5 = load_data(dataset='amazon5', samplecnt= -1, path='{}/torch_ds'.format(PATH_HOME))
ds_imdb = load_data(dataset='imdb', samplecnt= -1, path='{}/torch_ds'.format(PATH_HOME))


df_all = pd.concat([ds_yelp2.df_train, ds_yelp2.df_test, ds_amzon2.df_train, ds_amzon2.df_test, \
          ds_yelp5.df_train, ds_yelp5.df_test, ds_amzon5.df_train, ds_amzon5.df_test, \
          ds_imdb.df_train, ds_imdb.df_test])


df_all.drop_duplicates(['content'], inplace=True)
df_st_train, df_st_test =  train_test_split(df_all, test_size=0.05)
print(df_st_train.shape[0], df_st_test.shape[0])




rr = df_all['content'].map(lambda x: para_split2(x, True)).tolist()
rr_ = [i for i in rr if i]

df_text12 = pd.DataFrame(zip([c[0] for c in rr_], [c[1] for c in rr_]), columns=['text1', 'text2'])


df_st_train, df_st_test =  train_test_split(df_all, test_size=0.05)

df_st_train.to_csv("/scratch/w/wluyliu/yananc/finetunes/df_st_pp_train.csv", index=False)
df_st_test.to_csv("/scratch/w/wluyliu/yananc/finetunes/df_st_pp_test.csv", index=False)


df_st_train.to_csv("/scratch/w/wluyliu/yananc/finetunes/df_st_pps_train.csv", index=False)
df_st_test.to_csv("/scratch/w/wluyliu/yananc/finetunes/df_st_pps_test.csv", index=False)








'''
with open ("df_cc_ners_train.txt", 'w') as f:
    for ix, row in df_cc_train.iterrows():
        text = row['ners'] + ': '+ remove_str(row['content'])
        f.write(text + '\n')

with open ("df_cc_ners_test.txt", 'w') as f:
    for ix, row in df_cc_test.iterrows():
        text = row['ners'] + ': '+ remove_str(row['content'])
        f.write(text + '\n')




with open ("df_cc_title_train.txt", 'w') as f:
    for ix, row in df_cc_train.iterrows():
        text = "This document is about {} :".format(row['title']) +  remove_str(row['content']) 
        f.write(text + '\n')

with open ("df_cc_title_test.txt", 'w') as f:
    for ix, row in df_cc_test.iterrows():
        text = "This document is about {} :".format(row['title']) + remove_str(row['content']) 
        f.write(text + '\n')
'''





#######natcat

files = glob.glob("./torch_ds/natcat-data/*/train.tsv*.data")
infos = []
for file in files:
    with open(file, 'r') as f:
        for line in f: 
            tokens = line.strip().split('\t')
            assert len(tokens) == 9 
            if not tokens[-1] or len(tokens[-1].split(" "))<=5:
                continue
            infos.append(tokens)

df_nat = pd.DataFrame(infos, columns=['label'] + ['neg_label_{}'.format(i) for i in range(7)] + ['content'] )

df_nat = df_nat.loc[(~df_nat['content'].isnull()) & (df_nat['content']!='')]
df_nat['content'] = df_nat['content'].map(lambda x: remove_str(x))
df_nat['label'] = df_nat['label'].map(lambda x: x.replace('_',' ').lower())

print(df_nat.loc[df_nat['content'].isnull()].shape[0])


df_nat_train, df_nat_test =  train_test_split(df_nat, test_size=0.002)

df_nat_train.to_csv("./finetunes/df_nat_train.csv", index=False)
df_nat_test.to_csv("./finetunes/df_nat_test.csv", index=False)

# label ===> content

###### for gpt

'''
df_nat['text'] = df_nat['label'].map(lambda x: "This document is about {} : ".format(x)) \
                    + df_nat['content']  

df_nat_train, df_nat_test =  train_test_split(df_nat, test_size=0.001)

print(df_nat_train.shape[0], df_nat_test.shape[0])



with open ("df_nat_train.txt", 'w') as f:
    for line in df_nat_train['text'].tolist():
        f.write(remove_str(line) + '\n')

with open ("df_nat_test.txt", 'w') as f:
    for line in df_nat_test['text'].tolist():
        f.write(remove_str(line) + '\n')

with open ("df_nat_train_sample.txt", 'w') as f:
    for line in df_nat_train.sample(200000)['text'].tolist():
        f.write(remove_str(line) + '\n')

'''













'''


df_nat_train = pd.read_csv("df_nat_train.csv")
df_nat_test = pd.read_csv("df_nat_test.csv")



with open("nat4gptzsl_train.txt", 'w') as f:
    for ix, row in df_nat_train.iterrows():
        line = row['content'] + '. ' + row['prefix'] + tokenizer.eos_token
        f.write(line+'\n')


with open("nat4gptzsl_test.txt", 'w') as f:
    for ix, row in df_nat_test.iterrows():
        line = row['content'] + '. ' + row['prefix'] + tokenizer.eos_token
        f.write(line+'\n')



'''










