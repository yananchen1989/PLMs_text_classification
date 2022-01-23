

import glob
import pandas as pd 
files = glob.glob("./log_lambda/ag.128.64.*.log")

infos = []
for file in files:
    with open(file,'r') as f: 
        for line in f:
            if 'success summary===>' in line:
                #print(line)

                tokens = line.strip().split('summary==')[-1].split(' ')
                fmark = tokens[-3].split(':')[-1]
                acc_base = float(tokens[-2].split(':')[-1])
                acc_aug = float(tokens[-1].split(':')[-1])
                infos.append((fmark, acc_base, acc_aug))

df = pd.DataFrame(infos, columns=['fmark','acc_base','acc_aug'])



for fmark in df['fmark'].unique():
    dfi = df.loc[df['fmark']==fmark]
    print( fmark, dfi.shape[0], dfi['acc_base'].mean(), dfi['acc_aug'].mean()) 



















