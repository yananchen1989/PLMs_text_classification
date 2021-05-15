import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--aug", default="no", type=str)
parser.add_argument("--dsn", default="", type=str)
parser.add_argument("--samplecnt", default=-1, type=int)
parser.add_argument("--ite", default=5, type=int)
parser.add_argument("--ner_set", default=0, type=int)
parser.add_argument("--lang", default="zh", type=str)
parser.add_argument("--generate_m", default="gpt2", type=str)
parser.add_argument("--batch_size", default=64, type=int)
#parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--model", default="former", type=str)
#parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--beams", default=1, type=int)
parser.add_argument("--rp", default=1.0, type=float)
parser.add_argument("--check", default='enc', type=str)
parser.add_argument("--enc_m", default='dan', type=str)
parser.add_argument("--nli_m", default="joeddav/bart-large-mnli-yahoo-answers", type=str)
parser.add_argument("--thres", default=0.65, type=float)


args = parser.parse_args()
print('args==>', args)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
#from aug_generation import * 
from aug_translation import *
from load_data import * 
from transblock import * 
from encoders import *

if args.check == 'enc':
    enc = encoder(args.enc_m)
else:
    enc = None

if args.check == 'nli':
    nlp_nli = pipeline("zero-shot-classification", model=args.nli_m, device=0) #  1.8.1+cu102
else:
    nlp_nli = None


accs = []
for ite in range(args.ite): 

    print("iter ==> {}".format(ite))

    ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)

    print("augmentating...")

    if args.aug == 'generate':
        if args.generate_m == 'ctrl':
            args.rp = 1.2
        nlp  = pipeline("text-generation", model=args.generate_m, device=0, return_full_text=False)
        results = nlp(ds.df_train['content'].tolist(), max_length=250, do_sample=True, top_p=0.9, top_k=0, \
                    repetition_penalty=args.rp, num_return_sequences=args.beams)
        assert len(results) == ds.df_train.shape[0] and len(results[0]) == args.beams
        train_labels = ds.df_train['label'].tolist()
        labels_candidates = list(ds.df.label.unique())
        ori_sentences = ds.df_train['content'].tolist()
        infos = []

        for ii in range(ds.df_train.shape[0]):
            if args.check == 'enc':
                ori_sentence = ori_sentences[ii]
                ori_embed = enc.infer([ori_sentence])
                syn_sentences = [sentence['generated_text'] for sentence in results[ii]]
                syn_embeds = enc.infer(syn_sentences)
                simis = cosine_similarity(ori_embed, syn_embeds)
                df_simi = pd.DataFrame(zip(syn_sentences, simis[0]), columns=['content','simi'])
                df_simi.sort_values(by=['simi'], ascending=False, inplace=True)
                df_simi_filer = df_simi.loc[df_simi['simi']>= args.thres]

                for sentence in df_simi_filer['content'].tolist():
                    infos.append((sentence, train_labels[ii] ))

            elif args.check == 'nli':
                for sentence in results[ii]:
                    result_nli = nlp_nli(sentence['generated_text'], labels_candidates, multi_label=False, hypothesis_template="This text is about {}.")
                    if result_nli['scores'][0] > args.thres and result_nli['labels'][0] == train_labels[ii]:                    
                        infos.append((sentence['generated_text'], train_labels[ii] ))

            else:
                for sentence in results[ii]:
                    infos.append((sentence['generated_text'], train_labels[ii] ))

        df_synthesize = pd.DataFrame(infos, columns = ['content','label'])
        remain_ratio = df_synthesize.shape[0] / (ds.df_train.shape[0] * args.beams)
        aug_ratio = df_synthesize.shape[0] / ds.df_train.shape[0]
        print('check==>',args.check)
        print(df_synthesize['label'].value_counts())
        ds.df_train_aug = pd.concat([ds.df_train, df_synthesize])
        #assert ds.df_train_aug.shape[0] == ds.df_train.shape[0] * (args.beams + 1)

    elif args.aug in ['fillin','translate']:

        if args.aug == 'fillin':
            augmentor = fillInmask(ner_set=args.ner_set)
        
        if args.aug == 'translate':
            augmentor = backTranslate(lang=args.lang)

        ds.df_train['content_aug'] = ds.df_train['content'].map(lambda x: augmentor.augment(x))        
        ds.df_train_aug = pd.DataFrame(zip(ds.df_train['content_aug'].tolist()+ds.df_train['content'].tolist(), \
                                                 ds.df_train['label'].tolist()*2),
                                      columns=['content','label']).sample(frac=1)

    else:
        print("do not augmentation...")
        ds.df_train_aug = ds.df_train

    print(" begin to train ")
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
    print("iter completed, tranin acc ==> {}".format(best_val_acc))
    accs.append(best_val_acc)

acc_mean = round(sum(accs) / len(accs), 4)
record_log('log', ['summary==>'] + ['{}:{}'.format(k, v) for k, v in vars(args).items()] \
                 +['remain_ratio:{}'.format(remain_ratio), 'aug_ratio:{}'.format(aug_ratio)] \
                 + ['acc=> {}'.format(acc_mean)])









