import tensorflow as tf
#import tensorflow_datasets as tfds
#import matplotlib.image as mpimg
import io,sys
import json,argparse
import numpy as np
import datetime
import random,os,time
from sklearn.metrics import *
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
from utils.load_data import * 
from utils.transblock import * 
from utils.encoders import *

parser = argparse.ArgumentParser()
parser.add_argument('--dsn', default='ag', type=str)
parser.add_argument('--samplecnt', default=128, type=int)
parser.add_argument('--noise_cnt', default=32, type=int)
parser.add_argument('--dropbase', default='val', type=str)
args = parser.parse_args()


ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt) # ag news
# ds contains df_train and df_test, officially provided, no intersection between them
# randomly pick 64 samples per class

ixl = {ii[0]:ii[1] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}

num_classes = ds.df_test['label'].unique().shape[0]
# if ds.df_test['label'].unique().shape[0] == 2:
#     loss_fn = tf.keras.losses.BinaryCrossentropy()
# else:
#     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()


df_train, df_valid = train_test_split(ds.df_train, test_size=0.3)
df_train['groudtruth'] = 1
df_valid['groudtruth'] = 1

# fabricate noisy dataset, borrowed from ds.df_test
df_noise = ds.df_test.sample( args.noise_cnt )
df_noise['label'] = df_noise['label'].map(lambda x: noisy_label(x, ds.df_test['label'].unique() ))
df_noise['groudtruth'] = 0


# inject df_noise into the pure df_train, to get a dataset with some samples of wrong labels
df_train_noise = pd.concat([df_train, df_noise]).sample(frac=1)
#df_train_with_noise = df_train
enc = encoder('cmlm-large','cpu')
x_train = enc.infer(df_train_noise['content'].values)
x_valid = enc.infer(df_valid['content'].values)

y_train, y_valid = df_train_noise['label'].values, df_valid['label'].values


seed = int(time.time())
def get_model_mlp_(x_train, num_classes):
    text_input = tf.keras.layers.Input(shape=(x_train.shape[1]), dtype=tf.float32) 
    if num_classes == 2:
        out = layers.Dense(1, activation=None)(text_input)
        out_sigmoid = layers.Activation('sigmoid')(out)
        model = tf.keras.Model(inputs=text_input, outputs=out_sigmoid)
        model.compile('adam', "binary_crossentropy", metrics=["binary_accuracy"])
    else:
        out = layers.Dense(num_classes, activation=None)(text_input)
        out_softmax = layers.Activation('softmax')(out)
        model = tf.keras.Model(inputs=text_input, outputs=out_softmax)
        model.compile('adam', "sparse_categorical_crossentropy", metrics=["acc"])
    return model

model_cls = get_model_mlp_(x_train, num_classes)
# for i, layer in enumerate(model_cls.layers):
#     print(i, layer)

if num_classes == 2:
    monitor_acc = 'val_binary_accuracy'
else:
    monitor_acc = 'val_acc'


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=  os.path.join('./trackin_checkpoints', "model_seed_{}".format(seed)+ "__{epoch:d}.h5"),
    save_weights_only=True,
    monitor=monitor_acc,
    mode='max', save_freq='epoch',
    save_best_only=False)


hist_records = model_cls.fit(x_train, y_train, validation_data=(x_valid, y_valid),  \
        batch_size=32, epochs=50, verbose=1, callbacks=[model_checkpoint_callback,\
                    EarlyStopping(monitor=monitor_acc, patience=4, mode='max')])


trackin_epochs = []
for ix, ii in enumerate( zip(hist_records.history['val_loss'], hist_records.history[monitor_acc])): 
    epoch = ix+1
    loss, acc = ii[0], ii[1]
    if epoch <= 2:
        continue 
    loss_delta = hist_records.history['val_loss'][ix-1] - loss
    acc_delta = acc - hist_records.history[monitor_acc][ix-1]
    print(epoch, loss_delta, acc_delta)
    if loss_delta >= 0.001 :#and acc_delta >= 0.01:
        trackin_epochs.append(epoch)

assert len(trackin_epochs) >=3

checkpoints_files = []
for epoch in trackin_epochs:
    checkpoints_files.append("./trackin_checkpoints/model_seed_{}__{}.h5".format(seed, epoch))
# list.sort(checkpoints_files)
print("checkpoints_files==>", len(checkpoints_files))
# models_penultimate = []
# models_last = []
# for cf in checkpoints_files:
#     model_cls.load_weights(cf)
#     print(model_cls.get_weights()[-1].sum())
#     models_penultimate.append(tf.keras.Model(model_cls.layers[0].input, model_cls.layers[-3].output))
#     models_last.append(model_cls.layers[-2]) # logits: before softmax

#Find Proponents and Opponents for a given test example
#@tf.function
def get_trackin_grad(df, x_train, y_train):
    #imageids, images, labels = inputs
    contents_tensor = tf.convert_to_tensor(x_train)
    labels_tensor = tf.convert_to_tensor(y_train)
    # ignore bias for simplicity
    loss_grads = []
    activations = []
    # ml: (768, 4) 
    for cf in checkpoints_files:
        model_cls.load_weights(cf)
        #print(mp.get_weights()[-1].sum())
        mp = tf.keras.Model(model_cls.layers[0].input, model_cls.layers[-2].output)
        mpp = tf.keras.Model(model_cls.layers[0].input, model_cls.layers[-3].output)
    #for mp, ml in zip(models_penultimate, models_last):
        logits = mp(contents_tensor)
        embedding = mpp(contents_tensor)
        #logits = ml(h)
        if num_classes > 2:
            probs = tf.nn.softmax(logits)
            loss_grad = tf.one_hot(labels_tensor, num_classes) - probs
        elif num_classes == 2:
            probs = tf.nn.sigmoid(logits)
            loss_grad = tf.cast(labels_tensor, tf.float32) - probs
        activations.append(embedding)
        loss_grads.append(loss_grad)

        probs_cp, predicted_labels_cp = tf.math.top_k(probs, k=1)
        acc_cp = accuracy_score(y_train, predicted_labels_cp.numpy().reshape(-1))
        print('acc_cp:', round(acc_cp, 4), probs_cp.numpy().reshape(-1).sum(), \
            mp.get_weights()[0].sum())

      # Using probs from last checkpoint
    probs, predicted_labels = tf.math.top_k(probs, k=1)

    #return tf.stack(loss_grads, axis=-1), tf.stack(activations, axis=-1), labels, probs, predicted_labels
    return {'image_ids': df['content'].tolist(), 
            'groudtruth': df['groudtruth'].tolist(),
            'loss_grads': tf.stack(loss_grads, axis=-1).numpy(),
          'activations':  tf.stack(activations, axis=-1).numpy(),
          'labels': labels_tensor.numpy(),
          'probs': probs.numpy(),
          'predicted_labels': predicted_labels.numpy()
            } 

print('trackIn for train set')
trackin_train = get_trackin_grad(df_train_noise, x_train, y_train)
print('\ntrackIn for valid set')
trackin_val = get_trackin_grad(df_valid, x_valid, y_valid)


def find(loss_grad=None, activation=None):
    if loss_grad is None and activation is None:
        raise ValueError('loss grad and activation cannot both be None.')
    scores = []
    scores_lg = []
    scores_a = []
    for i in range(len(trackin_train['image_ids'])):
        if loss_grad is not None and activation is not None:
            lg_sim = np.sum(trackin_train['loss_grads'][i] * loss_grad)
            a_sim = np.sum(trackin_train['activations'][i] * activation)
            scores.append(lg_sim * a_sim)
            scores_lg.append(lg_sim)
            scores_a.append(a_sim)
        elif loss_grad is not None:
            scores.append(np.sum(trackin_train['loss_grads'][i] * loss_grad))
        elif activation is not None:
            scores.append(np.sum(trackin_train['activations'][i] * activation))    
    df_ppop = pd.DataFrame(zip(trackin_train['image_ids'], scores, scores_lg, scores_a), \
            columns=['content','score','score_lg','score_a'])
    df_ppop['prob'] = trackin_train['probs']
    df_ppop['predicted_labels'] = trackin_train['predicted_labels']
    df_ppop['label'] = trackin_train['labels']
    df_ppop['label_name'] = df_ppop['label'].map(lambda x: ixl[x])
    df_ppop['groudtruth'] = trackin_train['groudtruth']
    return df_ppop


# find pro and oppo for all samples of validation set, which is clean
df_ppop_ll = []
for idx in range(len(trackin_val['loss_grads'])):
    #if vector == 'influence':
    df_ppop_idx = find(trackin_val['loss_grads'][idx], trackin_val['activations'][idx])
    # elif vector == 'encoding':
    #     df_ppop = find(None, trackin_val['activations'][idx])  
    # elif vector == 'error':
    #     df_ppop = find(trackin_val['loss_grads'][idx], None)
    # else:
    #     raise ValueError('Unsupported vector type.')  
    df_ppop_ll.append(df_ppop_idx)


df_ppop = df_ppop_ll[0][['content','label','label_name','groudtruth']]

for col in ['score', 'score_lg', 'score_a']:
    df_ppop[col] = 0
    for df_ in df_ppop_ll:
        df_ppop[col] += df_[col]

# it is anticipated that samples of lower score(influence) shoule be samples from df_noise
for col in ['score', 'score_lg', 'score_a']:
    df_ppop[col] /= len(df_ppop_ll)


df_ppop.sort_values(by=['score'], ascending=True, inplace=True)
assert df_ppop.shape[0] == df_train_noise.shape[0]

for col in ['score', 'score_lg', 'score_a']:
    fpr, tpr, thresholds = roc_curve(df_ppop['groudtruth'].values, df_ppop[col].values, pos_label=1)
    auc_v = auc(fpr, tpr)
    print(col, auc_v)
















