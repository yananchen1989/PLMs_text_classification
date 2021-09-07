import tensorflow as tf
#import tensorflow_datasets as tfds
#import matplotlib.image as mpimg
import io,sys
import json
import numpy as np
import datetime
import random,os,time
from sklearn.metrics import *
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
from utils.load_data import * 
from utils.transblock import * 

ds = load_data(dataset='ag', samplecnt= 64) # ag news
# ds contains df_train and df_test, officially provided, no intersection between them
# randomly pick 64 samples per class

ixl = {ii[0]:ii[1] for ii in ds.df_test[['label','label_name']].drop_duplicates().values}


def noisy_label(l, unique_labels):
    unique_labels_ = [j for j in unique_labels if j!=l]
    assert l not in unique_labels_
    return random.sample(unique_labels_, 1)[0]


def get_model_bert_(num_classes, m='albert'):

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string

    encoder = hub.KerasLayer("./resource/albert_en_base_2", trainable=True)

    encoder_inputs = preprocessor_layer(text_input)
    outputs = encoder(encoder_inputs)
    embed = outputs["pooled_output"]  

    if num_classes == 2:
        out = layers.Dense(1, activation='sigmoid')(embed)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        #model.compile(Adam(lr=2e-5), "binary_crossentropy", metrics=["binary_accuracy"])
    else:
        out = layers.Dense(num_classes, activation=None)(embed)
        out_softmax = layers.Activation('softmax')(out)
        model = tf.keras.Model(inputs=text_input, outputs=out_softmax)
        #model.compile(Adam(lr=2e-5), "sparse_categorical_crossentropy", metrics=["acc"])
    return model

#base_optimizer = tf.keras.optimizers.Adam(learning_rate=4e-5)
base_optimizer = tf.keras.optimizers.SGD(learning_rate=5e-5)
@tf.function
def train_step_base(prompts, labels, model):
    # generator_ral update
    with tf.GradientTape() as tape:
        predictions = model(prompts)
        loss = keras.losses.SparseCategoricalCrossentropy()(labels, predictions)
    grads = tape.gradient(loss, model.trainable_weights)
    base_optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss

seed = int(time.time())

model_cls = get_model_bert_(ds.df_test.label.unique().shape[0], 'albert')

for i, layer in enumerate(model_cls.layers):
    print(i, layer)

loss_val_init = -1
acc_val_init = -1
df_train, df_valid = train_test_split(ds.df_train, test_size=0.3)
batch_size = 32

# fabricate noisy dataset, borrowed from ds.df_test
df_noise = ds.df_test.sample( int(df_train.shape[0] * 0.3) )
df_noise['label'] = df_noise['label'].map(lambda x: noisy_label(x, list(ixl.keys()) ))
df_noise['groudtruth'] = 0

df_train['groudtruth'] = 1
df_valid['groudtruth'] = 1
# inject df_noise into the pure df_train, to get a dataset with some samples of wrong labels
df_train_with_noise = pd.concat([df_noise, df_train]).sample(frac=1)

#loss_delta_model = []
for epoch in range(50):
    ix = 0
    while ix < df_train_with_noise.shape[0]:
        rows = df_train_with_noise[ix:ix+batch_size]
        content_tensor = tf.convert_to_tensor(rows['content'].tolist())
        label_tensor = tf.convert_to_tensor(rows['label'].tolist())
        loss = train_step_base(content_tensor, label_tensor, model_cls)
        ix += batch_size

    predictions = model_cls(tf.convert_to_tensor(df_valid['content'].tolist()), training=False)
    labels_val = tf.convert_to_tensor(df_valid['label'].tolist())
    loss_val= keras.losses.SparseCategoricalCrossentropy()(labels_val, predictions)
    preds = predictions.numpy().argmax(axis=1)
    acc_val = round(accuracy_score(labels_val.numpy(), preds), 4)
    
    if epoch > 0:
        loss_delta = loss_val_init - loss_val.numpy() 
        acc_delta = round((acc_val - acc_val_init)*100, 2)
        #loss_delta_model.append([loss_delta, epoch, model_cls.get_weights()])
        if loss_delta > 0.01 and acc_delta > 0.01:
            model_cls.save_weights("trackin_checkpoints/checkpont_seed{0}_epoch_{1:02d}.h5".format(seed, epoch))
        print('epoch:', epoch, loss_val.numpy(), acc_val, 'delta:', loss_delta, acc_delta)  
    loss_val_init, acc_val_init = loss_val.numpy(), acc_val




checkpoints_files = glob.glob("./trackin_checkpoints/checkpont_seed{}_epoch_*.h5".format(seed))
list.sort(checkpoints_files)
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
def get_trackin_grad(df):
    #imageids, images, labels = inputs
    contents_tensor = tf.convert_to_tensor(df['content'].tolist())
    labels_tensor = tf.convert_to_tensor(df['label'].tolist())
    # ignore bias for simplicity
    loss_grads = []
    activations = []
    # ml: (768, 4) 
    for cf in checkpoints_files:
        model_cls.load_weights(cf)
        #print(mp.get_weights()[-1].sum())
        mp = tf.keras.Model(model_cls.layers[0].input, model_cls.layers[-3].output)
        ml = model_cls.layers[-2]
    #for mp, ml in zip(models_penultimate, models_last):
        h = mp(contents_tensor)
        logits = ml(h['pooled_output'])
        probs = tf.nn.softmax(logits)
        loss_grad = tf.one_hot(labels_tensor, df['label'].unique().shape[0] ) - probs
        activations.append(h['pooled_output'])
        loss_grads.append(loss_grad)

        probs_cp, predicted_labels_cp = tf.math.top_k(probs, k=1)
        acc_cp = accuracy_score(df['label'].values, predicted_labels_cp.numpy().reshape(-1))
        print('acc_cp:', round(acc_cp, 4), probs_cp.numpy().reshape(-1).sum(), \
            mp.get_weights()[-1].sum(), ml.get_weights()[0].sum())

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
trackin_train = get_trackin_grad(df_train_with_noise)
print('\ntrackIn for valid set')
trackin_val = get_trackin_grad(df_valid)


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


for col in ['score', 'score_lg', 'score_a']:
    fpr, tpr, thresholds = roc_curve(df_ppop['groudtruth'].values, df_ppop['score'].values, pos_label=1)
    auc_v = auc(fpr, tpr)
    print(col, auc_v)
















