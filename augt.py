import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math,time,operator,traceback,shutil,string
from sklearn import metrics
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# import GPUtil
# GPUtil.showUtilization()
# deviceIDs = [0,1,2,3]
# #deviceIDs = GPUtil.getAvailable(order = 'memory', limit = 4, maxLoad = 1, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
# print("deviceIDs ==> ", deviceIDs)
# assert len(deviceIDs) >= 2

parser = argparse.ArgumentParser()
parser.add_argument("--aug", default="generate", type=str)
parser.add_argument("--dsn", default="ag", type=str, choices=['uci','ag','agt','nyt','yelp2','amazon2','stsa'])
parser.add_argument("--samplecnt", default=64, type=int)
parser.add_argument("--max_aug_times", default=1, type=int)

parser.add_argument("--backbone", default="former", type=str)
parser.add_argument("--verbose", default=0, type=int)
parser.add_argument("--basemode", default="max", type=str) # rank or thres

#parser.add_argument("--nlim", default="joeddav/bart-large-mnli-yahoo-answers", type=str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--testbed", default=1, type=int)
parser.add_argument("--testvalid", default='test', type=str)
parser.add_argument("--seed", default=0, type=int)
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
from transformers import pipeline
from threading import Thread
#tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
#tf.keras.backend.set_floatx('float16')
import nltk 
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity
#nltk.download('wordnet')
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
device0 = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
#assert device0.type=='cuda' 

from utils.load_data import * 
from utils.transblock import * 
from utils.encoders import *
from utils.cbert_cgpt_config import * 

ds = load_data(dataset=args.dsn, samplecnt= 8)

for epoch in range(100):
    print('epoch:', epoch)
    files_csv = glob.glob("./augf_csvs/{}_{}_*.csv".format(args.dsn, args.samplecnt))

    for file_csv in files_csv:
         
        seed = file_csv.split('.')[1].split('_')[-1]
        df_train = pd.read_csv(file_csv)
        
        df_train_noaug = df_train.loc[df_train['fmark'] == 'ori']
        acc_noaug, _  = do_train_test_thread(df_train_noaug, ds.df_test, args.backbone, 32, args.epochs)

        print('seed:', seed, '==>', 'noaug', acc_noaug)
        for fmark in df_train['fmark'].unique():
            if fmark == 'ori':
                continue
            df_train_fmark = df_train.loc[df_train['fmark'] == fmark]
            df_train_aug = pd.concat([df_train_noaug, df_train_fmark]).sample(frac=1)
            acc_aug, _  = do_train_test_thread(df_train_aug.loc[~df_train_aug['content'].isnull()], ds.df_test, args.backbone, 32, args.epochs)
            print('seed:', seed, '==>', fmark, acc_aug)

    print('\n')











