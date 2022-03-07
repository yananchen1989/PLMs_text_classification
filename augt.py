import sys,os,logging,glob,csv,datetime,gc,argparse,math,time,operator,traceback,string
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="ag", type=str, choices=['uci','ag','agt','nyt','yelp2','amazon2','stsa'])
parser.add_argument("--samplecnt", default=64, type=int)
parser.add_argument("--backbone", default="former", type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--gpu", default="", type=str)


args = parser.parse_args()
print('args==>', args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#args.filter = args.gpu.split(',')

import numpy as np
import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras


gpus = tf.config.list_physical_devices('GPU')
print('======>',gpus,'<=======')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

print("number of gpus==>", len(gpus))

from utils.load_data import * 
from utils.transblock import * 

ds = load_data(dataset=args.dsn, samplecnt= 8)


files_csvs = []
for samplecnt in [32, 64, 128, 256, 512]:
    files_csv = glob.glob("./augf_csvs/{}_{}_*.csv".format(args.dsn, samplecnt))
    files_csvs.extend(files_csv)

file_csv = random.sample(files_csvs, 1)[0]
 
seed = file_csv.split('.')[1].split('_')[-1]
samplecnt = file_csv.split(args.dsn)[-1].split('.')[0].split('_')[1]


df_train = pd.read_csv(file_csv)
print(df_train['fmark'].value_counts())

df_train_noaug = df_train.loc[df_train['fmark'] == 'ori']
# acc_noaug, _  = do_train_test_thread(df_train_noaug, ds.df_test, args.backbone, 32, args.epochs)
# print('seed:', seed, '==>', samplecnt, 'noaug', acc_noaug)

fmarks = df_train['fmark'].unique().tolist()
random.shuffle(fmarks)

for fmark in fmarks:
    if fmark == 'ori':
        continue
    if fmark in ['eda', 'bt']:
        continue
    df_train_fmark = df_train.loc[df_train['fmark'] == fmark]
    df_train_aug = pd.concat([df_train_noaug, df_train_fmark]).sample(frac=1)
    acc_aug, _  = do_train_test_thread(df_train_aug.loc[~df_train_aug['content'].isnull()], ds.df_test, args.backbone, 32, args.epochs)
    print('seed:', seed, '==>', samplecnt, fmark, acc_aug)













