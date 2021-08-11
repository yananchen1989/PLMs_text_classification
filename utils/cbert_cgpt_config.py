from sklearn.model_selection import train_test_split
import os 
import pandas as pd 

def write_for_cbert(df_train, df_test, target_path, boostsample_ft):
    if boostsample_ft > 0:
        df_train_ft_aug = boost_for_ft(df_train, boostsample_ft, model=None, use_ent=0)
    else:
        df_train_ft_aug = df_train

    os.makedirs(target_path, exist_ok=True)
    df_train_ft_aug[['label','content']].to_csv(target_path+'/train.tsv', index=False, header=None, sep='\t')
    df_test, df_dev = train_test_split(df_test, test_size=0.1) 
    df_dev[['label','content']].to_csv(target_path+'/dev.tsv', index=False, header=None, sep='\t')
    df_test[['label','content']].to_csv(target_path+'/test.tsv', index=False, header=None, sep='\t')


from scipy.stats import entropy
import nltk,random
def pick_prefix(para, model=None, use_ent=0):
    sents = nltk.sent_tokenize(para)
    if len(sents)==1:
        return para

    if not use_ent:
        random.shuffle(sents)
        return ' '.join(sents)
    else:
        preds = model.predict(sents, batch_size=64, verbose=0)
        preds_1 = list(preds.reshape(-1))
        preds_0 = list(1 - preds.reshape(-1))
        ent = [entropy([ii[0], ii[1]], base=2) for ii in zip(preds_0, preds_1)]
        df_sents = pd.DataFrame(zip(sents, ent), columns=['sent','ent'])
        dfs = df_sents.sample(frac=1, weights='ent')
        return ' '.join(dfs['sent'].tolist())

def boost_for_ft(df_train_ft, boostsample_ft, model, use_ent=0):
    df_train_ft_ll = []
    df_train_ft_ll.append(df_train_ft)
    for _ in range(boostsample_ft):
        df_train_ft['content'] = df_train_ft['content'].map(lambda x: pick_prefix(x, model, use_ent=use_ent))
        df_train_ft_ll.append(df_train_ft)
    df_train_ft_aug = pd.concat(df_train_ft_ll)
    return df_train_ft_aug