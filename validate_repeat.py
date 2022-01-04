import os,argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="uci", type=str, choices=['uci','ag','agt','nyt','yelp2','amazon2','stsa'])
parser.add_argument("--backbone", default="albert", type=str)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from utils.load_data import * 
from utils.transblock import * 



ds = load_data(dataset=args.dsn, samplecnt= 32)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))
ds, proper_len = process_ds(ds, 128)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))

ixl = {ii[0]:ii[1] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
ixl_rev = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}


infos = []
for samplecnt in [32, 64, 128]:
    for candidates in [64, 256, 512]:
        files = glob.glob("./log_arxiv_clsembednlinsp/{}.{}.{}.*.log".format(args.dsn, samplecnt, candidates))
        if not files:
            continue
        
        for file in files:

            lines = []
            with open(file,'r') as f: 
                for line in f:
                    if not line.strip():
                        continue
                    if 'ori====>' in line or 'ft-nlisp ==>' in line or 'ft-nli ==>' in line or 'ft-nsp ==>' in line or \
                        'ft-cls ==>' in line or 'ft-embed ==>' in line or 'nofil ==>' in line:
                        #print(line)
                        lines.append(line.strip())

            cutix = []
            for ix, line in enumerate(lines):
                if 'ori====>' in line:
                    cutix.append(ix)

            
            for i in range(len(cutix)-1):
                assert cutix[i+1] - cutix[i] == 13
                lines_ = lines[cutix[i]:cutix[i+1]]
                 
                ori_content = lines_[0].split('ori====>')[-1].strip().split('<===')[0].strip()
                label = lines_[0].split('ori====>')[-1].strip().split('<===')[1].strip()
                infos.append((ori_content, 'ori',label, samplecnt, candidates ))
                for line in lines_[1:]:
                    fmark = line.split('==>')[0].strip()
                    content = line.split('==>')[1].strip()
                    infos.append((content, fmark, label, samplecnt, candidates))

df_tmp = pd.DataFrame(infos, columns=['content','fmark','label_name', 'samplecnt', 'candidates'])
df_tmp['label'] = df_tmp['label_name'].map(lambda x: ixl_rev[x])

for col in ['samplecnt','candidates','max_aug_times','candidates']:
    if col in df_tmp.columns:
        print(col)
        df_tmp[col] = df_tmp[col].astype('int')



for ite in range(12):
    for samplecnt in df_tmp['samplecnt'].unique():
        for candidates in df_tmp['candidates'].unique():

            df_tmpi = df_tmp.loc[(df_tmp['samplecnt']==samplecnt) & (df_tmp['candidates']==candidates)]
            for fmark in df_tmp['fmark'].unique():
                if fmark == 'ori':
                    continue

                df_ori = sample_stratify(df_tmpi.loc[df_tmpi['fmark']=='ori'], \
                            min(df_tmpi.loc[df_tmpi['fmark']=='ori'].label_name.value_counts().min(), samplecnt) )
                df_fmark =  sample_stratify(df_tmpi.loc[df_tmpi['fmark']==fmark], \
                            min(df_tmpi.loc[df_tmpi['fmark']==fmark].label_name.value_counts().min(), samplecnt))
                print(samplecnt, candidates, fmark, "===>")
                print(df_ori['label_name'].value_counts())
                print(df_fmark['label_name'].value_counts())

                df_train_aug = pd.concat([df_ori, df_fmark ] ).sample(frac=1)

                acc_noaug, _  = do_train_test_thread(df_ori,       ds.df_test, args.backbone, 16)
                acc_aug, _  = do_train_test_thread(df_train_aug,   ds.df_test, args.backbone, 16)

                print("summary=={}".format(ite), args.dsn,  samplecnt, candidates, fmark, \
                            acc_noaug, acc_aug)






