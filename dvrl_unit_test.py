import sklearn,random,sys,argparse
from sklearn.model_selection import train_test_split
from utils.load_data import * 
#from utils.transblock import * 
from utils.encoders import *

enc = encoder('cmlm-large')
# print('cmlm loaded')

parser = argparse.ArgumentParser()
parser.add_argument('--dsn', default='ag', type=str)
parser.add_argument( '--threds', default=32, type=int)
args = parser.parse_args()

seed = random.sample(list(range(10000)), 1)[0]

if not os.path.exists('dvrl_np_array'):
    os.makedirs('dvrl_np_array')
##################################

ds = load_data(dataset=args.dsn, samplecnt=128)
ds.df_train['groudtruth'] = 1


df_syn_tmp = pd.read_csv("df_syn_tmp.csv")
df_syn_tmp['groudtruth'] = 9
del df_syn_tmp['score']


df_train_valid_noise = pd.concat([ds.df_train,  df_syn_tmp])

embeds = enc.infer(df_train_valid_noise['content'].values)

for ii in range(embeds.shape[1]):
    df_train_valid_noise['embed_{}'.format(ii)] = embeds[:, ii]



from threading import Thread


def run_dvrl_thread(dsn, ii, seed):
    os.system('python dvrl_iter.py --dsn {} --seed {} --ite {}'.format(dsn, seed, ii))

threads = []
for ii in range(args.threds):
    t = Thread(target=run_dvrl_thread, args=(args.dsn, ii, seed))
    t.start()
    threads.append(t)

# join all threads
for t in threads:
    t.join()
print("after join")


df_train_noise_files = glob.glob("./dvrl_np_array/df_train_noise_{}_{}_*.csv".format(args.dsn, seed))
print("valid output==>", len(df_train_noise_files), df_train_noise_files)





ll = []
for file in df_train_noise_files:
    dfi = pd.read_csv(file)
    auc = float(file.split('_')[-1].replace('.csv','')) 
    if dfi['dve_out'].std() == 0 or auc <= 0.8:
        continue
    ll.append(dfi)
    print(file, auc, dfi.shape[0], dfi['content'].unique().shape[0])


df_train_syn = pd.concat(ll)
df_syn = df_train_syn.loc[df_train_syn['groudtruth']==9]

#sent = ' the state is demanding that diebold for being involved in the election hacking case end their ties to the scheme that currently prevents an electronic voter from being counted. #Chess2016 That is very nice.'

#df_syn_agg.loc[df_syn_agg['content']==sent]
df_syn_agg = df_syn.groupby(['content', 'label', 'label_name'])['dve_out'].mean().reset_index()

df_syn_agg.sort_values(by=['dve_out'], ascending=False, inplace=True) 


for l in df_syn_agg['label_name'].unique():
    print(l, df_syn_agg.loc[df_syn_agg['label_name']==l]['dve_out'].mean())


print("head====> ")
for ix, row in df_syn_agg.head(14).iterrows():
    print(row['label_name'], '===>')
    print(row['content'].replace('\n',' '))

print("tail====> ")
for ix, row in df_syn_agg.tail(14).iterrows():
    print(row['label_name'], '===>')
    print(row['content'].replace('\n',' '))






