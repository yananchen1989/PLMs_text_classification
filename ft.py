import pandas as pd 
import glob
from sklearn.model_selection import train_test_split
from utils.load_data import * 


df = pd.read_csv("./torch_ds/df_cc_news_ners.csv", lineterminator='\n')



dfl = df.loc[(~df['ners'].isnull()) & (df['ners'].str.contains('<=>')) & (~df['title'].isnull()) ]



dfl['ners'] = dfl['ners'].map(lambda x:  "This document is about {}".format(', '.join(x.split("<=>"))))

df_cc_train, df_cc_test =  train_test_split(dfl, test_size=0.02)


df_cc_train[['ners', 'content']].to_csv("df_cc_ners_train.csv", index=False)
df_cc_test[['ners', 'content']].to_csv("df_cc_ners_test.csv", index=False)


df_cc_train[['title', 'content']].to_csv("df_cc_title_train.csv", index=False)
df_cc_test[['title', 'content']].to_csv("df_cc_title_test.csv", index=False)



from datasets import load_dataset


data_files = {}

data_files["train"] = './df_cc_ners_train.txt'
data_files["validation"] = './df_cc_ners_test.txt'

extension = "text"
raw_datasets = load_dataset(extension, data_files=data_files)




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


print(df_nat.loc[df_nat['content'].isnull()].shape[0])



###### for gpt
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





####### for t5



df_nat['prefix'] = df_nat['label'].map(lambda x: "This document is about {}".format(x))

df_nat_train, df_nat_test =  train_test_split(df_nat[['label', 'content']], test_size=0.001)

df_nat_train.to_csv("./finetunes/nat4zsl_train.csv", index=False)
df_nat_test.to_csv("./finetunes/nat4zsl_test.csv", index=False)














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




df_nat_test = pd.read_csv("./finetunes/df_nat_test.csv")




import fasttext
model = fasttext.load_model('lid.176.bin')

cc = 0
for text in df_nat_test['content'].tolist():
    preds = model.predict(text, k=5)
    if (not preds[0][0].endswith("__en")) or (preds[0][0].endswith("__en") and preds[1][0] <= 0.5):
        print(text)
        cc += 1
print(cc)




'''
t5:
t5_natcat     This document is about [] [] [] ==> content
t5_ners_cc     [] [] [] ==> content


'''








