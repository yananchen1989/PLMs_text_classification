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
import gc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

import datetime
def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
logging.Formatter.converter = beijing


from aug_fillinmask import *
from aug_generation import * 
from aug_translation import *
from load_data import * 
from transblock import * 


def run_benchmark(dataset, augmentor, logger):
    accs = []
    for ite in range(3): 

        logger.info("iter ==> {}".format(ite))

        ds = load_data(dataset=dataset, samplecnt=1000)

        if augmentor is not None:
            # augmentation
            logger.info("augmentating...")
            ds.df_train['content_aug'] = ds.df_train['content'].map(lambda x: augmentor.augment(x))
            logger.info("augmentated...")
            
            ds.df_train_aug = pd.DataFrame(zip(ds.df_train['content_aug'].tolist()+ds.df_train['content'].tolist(), \
                                                     ds.df_train['label'].tolist()*2),
                                          columns=['content','label']).sample(frac=1)
        else:
            logger.info("do not augmentation...")
            ds.df_train_aug = ds.df_train

        (x_train, y_train),  (x_test, y_test), num_classes = get_keras_data(ds.df_train_aug, ds.df_test)

        model = get_model(num_classes)

        logger.info("train begin==>")
        history = model.fit(
            x_train, y_train, batch_size=32, epochs=12, validation_data=(x_test, y_test), verbose=2,
            callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
        )
        best_val_acc = max(history.history['val_acc'])
        logger.info("iter completed, tranin acc ==>{}".format(best_val_acc))
        accs.append(best_val_acc)
    # print("accs==>", accs)
    # print("dataset:{} mean acc ==>".format(dataset), sum(accs) / len(accs))
    return round(sum(accs) / len(accs), 4)




# AUGMENTATION_FUNCTIONS = {
#     "fillin": fillInmask(ner_set=False),
#     "generate": generation(model_name='ctrl'),
#     "translate": backTranslate(lang='zh')
# }

parser = argparse.ArgumentParser()
parser.add_argument("--aug", default="", type=str)
parser.add_argument("--ds", default="", type=str)
parser.add_argument("--ner_set", default=False, type=bool)
parser.add_argument("--lang", default="zh", type=str)
parser.add_argument("--generate_m", default="ctrl", type=str)

args = parser.parse_args()

logging.basicConfig(
    filename='log_{}_{}'.format(args.aug, args.ds),
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO, filemode='w'
)

logger = logging.getLogger()
logger.info("aug_method started ==> {} on dataset==>{}".format(args.aug, args.ds))



if args.aug == 'fillin':
    augmentor = fillInmask(ner_set=args.ner_set)

elif args.aug == 'generate':
    augmentor = generation(model_name=args.generate_m)

elif args.aug == 'translate':
    augmentor = backTranslate(lang=args.lang)

elif args.aug == 'no':
    augmentor = None

else:
    raise KeyError("args.aug illegal!")
logger.info("model loaded")
#for aug_method, augmentor in AUGMENTATION_FUNCTIONS.items():


logger.info("dataset begin ==> {}".format(args.ds))
acc_mean = run_benchmark(args.ds, augmentor, logger)
logger.info("summary aug:{} dataset:{}  acc=>{}".format(args.aug, args.ds, acc_mean))

'''
nohup python baseline_classifier.py --aug generate --ds yahoo --generate_m gpt2 &
nohup python baseline_classifier.py --aug fillin --ds ag --ner_set True &
'''

# unit test
# augmentor = fillInmask(ner_set=False)

# dataset = 'ag'
# df_train, df_test = LOADDATA_FUNCTIONS[dataset]()
# for title in df_train['content'].tolist():
#     t = augmentor.augment(title)





# intermediate_layer_model = tf.keras.Model(inputs=model.input,
#                                  outputs=model.get_layer('keras_layer_4').output['pooled_output'])
# intermediate_output = intermediate_layer_model.predict(x_val, verbose=1,  batch_size=1024)








