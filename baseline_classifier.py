import sys,os,logging,glob,pickle,torch
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
import gc,argparse,datetime

parser = argparse.ArgumentParser()
parser.add_argument("--aug", default="no", type=str)
parser.add_argument("--ds", default="", type=str)
parser.add_argument("--samplecnt", default=-1, type=int)
parser.add_argument("--ite", default=5, type=int)
parser.add_argument("--ner_set", default=0, type=int)
parser.add_argument("--lang", default="zh", type=str)
parser.add_argument("--generate_m", default="", type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--model", default="former", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--generate_use_label", default=0, type=int)

args = parser.parse_args()
print('args==>', args)


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)


from aug_fillinmask import *
from aug_generation import * 
from aug_translation import *
from load_data import * 
from transblock import * 


def run_benchmark(dataset, augmentor, samplecnt):
    accs = []
    for ite in range(args.ite): 

        print("iter ==> {}".format(ite))

        ds = load_data(dataset=dataset, samplecnt=samplecnt)

        if augmentor is not None:
            # augmentation
            print("augmentating...")
            if args.generate_use_label and args.aug == 'generate':
                ds.df_train['content_aug'] = (ds.df_train['label'] +' '+ ds.df_train['content'] ).map(lambda x: augmentor.augment(x))
            else:
                ds.df_train['content_aug'] = ds.df_train['content'].map(lambda x: augmentor.augment(x))
            print("augmentated...")
            
            ds.df_train_aug = pd.DataFrame(zip(ds.df_train['content_aug'].tolist()+ds.df_train['content'].tolist(), \
                                                     ds.df_train['label'].tolist()*2),
                                          columns=['content','label']).sample(frac=1)
        else:
            print("do not augmentation...")
            ds.df_train_aug = ds.df_train

        (x_train, y_train),  (x_test, y_test), num_classes = get_keras_data(ds.df_train_aug, ds.df_test)

        if args.model in ['albert','electra', 'dan']:
            model = get_model_bert(num_classes, args.model)
        elif args.model == 'former':
            model = get_model_transormer(num_classes)
        else:
            raise KeyError("input model illegal!")

        print("train begin==>")

        history = model.fit(
            x_train, y_train, batch_size=args.batch_size, epochs=50, \
            validation_batch_size=64,
            validation_data=(x_test, y_test), verbose=1,
            callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
        )

        best_val_acc = max(history.history['val_acc'])
        print("iter completed, tranin acc ==>{}".format(best_val_acc))
        accs.append(best_val_acc)
    # print("accs==>", accs)
    # print("dataset:{} mean acc ==>".format(dataset), sum(accs) / len(accs))
    return round(sum(accs) / len(accs), 4)


print("aug_method started ==> {} on dataset==>{}".format(args.aug, args.ds))

if args.aug == 'fillin':
    # model_name='/root/yanan/berts/transformers/examples/language-modeling/finetuned_bert'
    augmentor = fillInmask(ner_set=args.ner_set, device=args.device)

elif args.aug == 'generate':
    augmentor = generation(model_name=args.generate_m,  device=args.device)

elif args.aug == 'translate':
    augmentor = backTranslate(lang=args.lang, device=args.device)

elif args.aug == 'no':
    augmentor = None

else:
    raise KeyError("args.aug illegal!")
print("model loaded")


print("dataset begin ==> {}".format(args.ds))
acc_mean = run_benchmark(args.ds, augmentor, args.samplecnt)
print("summary aug:{} dataset:{} samplecnt:{} acc=>{}".format(args.aug, args.ds, args.samplecnt, acc_mean))








