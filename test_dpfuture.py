import argparse,os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default='uci', type=str)
parser.add_argument("--gpu", default="7", type=str)
args = parser.parse_args()
print('args==>', args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import tensorflow as tf 
gpus = tf.config.experimental.list_physical_devices('GPU')
print('======>',gpus,'<=======')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

from transformers import top_k_top_p_filtering
from torch.nn import functional as F
import os,string,torch,math,time

from utils.load_data import * 
from utils.transblock import * 
ds = load_data(dataset=args.dsn, samplecnt= -1)
#ds, proper_len = process_ds(ds, 64)

ixl = {ii[0]:ii[1] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}
ixl_rev = {ii[1]:ii[0] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}


from threading import Thread
testbed_func = {"test":do_train_test_thread, "valid":do_train_test_valid_thread}


def thread_testing(testvalid, df_train, df_test):
    best_test_accs = []
    models = []

    for ddi in range(3):
        threads = []
        for di in range(1):
            t = Thread(target=testbed_func[testvalid], \
                        args=(df_train, df_test, best_test_accs, models, di + ddi*2, 100,  0))
            t.start()
            threads.append(t)

        # join all threads
        for t in threads:
            t.join()

    acc = round(np.array(best_test_accs).max(), 4)

    #model_best = models[np.array(best_test_accs).argmax()]
    return  acc


import glob
files = glob.glob("./log_dpfuture/dpfuture.{}.samplecnt_128.max_aug_times_1.candidates_128.test_beams_128.*.log".format(args.dsn))

infos = []
for file in files:

    with open(file, 'r') as f:
        lines = f.readlines()
     
    start_ix = []
    for i, line in enumerate(lines):
        if 'eval_result_oris==>' in  line:
            start_ix.append(i)

    for i, j in zip(start_ix[0:len(start_ix)-1], start_ix[1:len(start_ix)] ):

        label_name = lines[i:j][2].split("==>")[0].strip()
        content_ori = lines[i:j][2].split("==>")[1].strip()
        content_syn = lines[i:j][4].replace("gen==>","").strip()
        infos.append((label_name, content_ori, content_syn))


df = pd.DataFrame(infos, columns=['label_name', 'content', 'content_syn'])
df['label'] = df['label_name'].map(lambda x: ixl_rev[x])

ds = load_data(dataset=args.dsn, samplecnt= -1)

ds, proper_len = process_ds(ds, 64)
ds.df_train['content'] = ds.df_train['content'].map(lambda x: x.strip(string.punctuation).strip())
ds.df_test['content'] = ds.df_test['content'].map(lambda x: x.strip(string.punctuation).strip())

if args.dsn == 'uci':
    df_all = pd.concat([ds.df_train, ds.df_test])
    df_test = df_all.loc[~df_all['content'].isin(df['content'].tolist())]
    print(df_test.shape[0], df_all.shape[0])

elif args.dsn == 'agt':
    df_test = ds.df_test


for samplecnt in [32, 64, 128]:
    for trial in range(7):
        df_ = df.sample(frac=1).drop_duplicates(['content','label_name'])
        df_train = sample_stratify(df_, samplecnt)

        acc_noaug = thread_testing('test', df_train[['label_name','label', 'content']], df_test)

        acc_aug = thread_testing('test', \
            pd.concat([df_train[['label_name','label', 'content']], \
                      df_train[['label_name','label', 'content_syn']].rename(columns={'content_syn': 'content'}) ]), \
            df_test) 

        gain = (acc_aug - acc_noaug) / acc_noaug
        print("samplecnt:", samplecnt, "trial:", trial, 'gain:', gain, "<==", acc_noaug, acc_aug)

