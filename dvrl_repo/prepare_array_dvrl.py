import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

# Data loading
#from dvrl import data_loading

###########################dsn 
from sklearn.model_selection import train_test_split
from utils.load_data import * 
from utils.encoders import *
from utils.transblock import * 

enc = encoder('cmlm-large','cpu')

##################################
ds = load_data(dataset='stsa', samplecnt= 64)
df_train, df_valid = train_test_split(ds.df_train, test_size=0.3)
df_train['groudtruth'] = 1



df_noise['label'] = df_noise['label'].map(lambda x: noisy_label(x, ds.df_test['label'].unique() )  )
df_noise['groudtruth'] = 0 

df_train_noise = pd.concat([df_train, df_noise]).sample(frac=1)


x_train = enc.infer(df_train_noise['content'].values)
x_valid = enc.infer(df_valid['content'].values)

y_train, y_valid = df_train_noise['label'].values, df_valid['label'].values

groudtruth = df_train_noise['groudtruth'].values



###############
ds = load_data(dataset='ag', samplecnt= 64)
df_train, df_valid = train_test_split(ds.df_train, test_size=0.3)

df_incremental = ds.df_test.sample(64)
df_incre_clean, df_incre_corrupt = train_test_split(df_incremental, test_size=0.5)
df_incre_corrupt['label'] = df_incre_corrupt['label'].map(lambda x: noisy_label(x, ds.df_test['label'].unique() )  )

x_train = enc.infer(df_train['content'].values)
x_valid = enc.infer(df_valid['content'].values)

y_train, y_valid = df_train['label'].values, df_valid['label'].values


def get_model_mlp(x_train, num_classes):
    text_input = tf.keras.layers.Input(shape=(x_train.shape[1]), dtype=tf.float32) 
    if num_classes == 2:
        out = tf.keras.layers.Dense(1, activation='sigmoid')(text_input)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(tf.keras.optimizers.SGD(lr=0.01), "binary_crossentropy", metrics=["binary_accuracy"])
    else:
        out = tf.keras.layers.Dense(num_classes, activation="softmax")(text_input)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(tf.keras.optimizers.SGD(lr=0.01), "sparse_categorical_crossentropy", metrics=["acc"])
    return model

model = get_model_mlp(x_train, ds.df_test['label'].unique().shape[0] )

model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=16, epochs=100, verbose=1, \
            callbacks = [EarlyStopping(monitor='val_acc', patience=7, mode='max')] )


loss_valid, acc_valid = model.evaluate(x_valid, y_valid, batch_size=64)
print("init==>", loss_valid, acc_valid)

model.save_weights("incremental_ori_model.h5")


model.compile(tf.keras.optimizers.SGD(lr=0.01), "sparse_categorical_crossentropy", metrics=["acc"])

for _ in range(20):
    dfi = df_incre_clean.sample(16)

    x_tmp = enc.infer(dfi['content'].values)
    y_tmp =  dfi['label'].values

    model.load_weights("incremental_ori_model.h5")
    
    model.train_on_batch(x_tmp, y_tmp)
    loss_after, acc_after = model.evaluate(x_valid, y_valid, batch_size=64)
    print("acc delta:", acc_after - acc_valid)









# np.save('./dvrl_np_array/x_train_{}.npy'.format(dsn), x_train)
# np.save('./dvrl_np_array/x_valid_{}.npy'.format(dsn), x_valid)

# np.save('./dvrl_np_array/y_train_{}.npy'.format(dsn), y_train)
# np.save('./dvrl_np_array/y_valid_{}.npy'.format(dsn), y_valid)

# np.save('./dvrl_np_array/groudtruth_{}.npy'.format(dsn), groudtruth)




# model = get_model_mlp(x_train, ds.df_test['label'].unique().shape[0])
# model.fit(x_train, y_train, validation_data=(x_valid, y_valid),  batch_size=64, epochs=100, verbose=1)

# acc, model_bert = do_train_test(df_train_, df_valid, epochs=100, freq=1, verbose=1, \
#                basetry=3, samplecnt=128, basemode='max', model_name='albert', gpu=0)


#############
# ds = load_data(dataset='stsa', samplecnt= -1)
# x_valid = enc.infer(ds.df_test['content'].values)
# y_valid = ds.df_test['label'].values


# x_train = enc.infer(ds.df_train['content'].values)
# y_train = ds.df_train['label'].values

# model = get_model_mlp(x_train, ds.df_test['label'].unique().shape[0])
# model.fit(x_train, y_train, validation_data=(x_valid, y_valid),  batch_size=64, epochs=200, verbose=1)



# Data preprocessing
# Normalization methods: 'minmax' or 'standard'
#normalization = args.normalization

# Extracts features and labels. Then, normalizes features
# x_train, y_train, x_valid, y_valid, x_test, y_test, col_names = \
# data_loading.preprocess_data(normalization, 'train.csv',
#                            'valid.csv', 'test.csv')


#from dvrl import dvrl_v2
# Run DVRL
# Resets the graph
# tf.compat.v1.reset_default_graph()
# tf.compat.v1.keras.backend.clear_session()

# # Here, we assume a classification problem and we assume a predictor model
# # in the form of a simple multi-layer perceptron.
# # Predictive model define
# pred_model = tf.compat.v1.keras.models.Sequential()
# # pred_model.add(keras.layers.Dense(parameters['hidden_dim'],
# #                                 activation='relu'))
# # pred_model.add(keras.layers.Dense(parameters['hidden_dim'],
# #                                 activation='relu'))
# pred_model.add(tf.compat.v1.keras.layers.Dense(np.unique(y_train).shape[0], activation='softmax'))
# pred_model.compile(optimizer='adam', loss='categorical_crossentropy',
#                  metrics=['accuracy'])

# # Flags for using stochastic gradient descent / pre-trained model
# flags = {'sgd': True, 'pretrain': False}

# # Initializes DVRL
# dvrl_class = dvrl_v2.Dvrl(x_train, y_train, x_valid, y_valid,
#                      'classification', pred_model, vars(args),
#                      args.checkpoint_file_name, flags)

# Trains DVRL
# dvrl_class.train_dvrl(args.perf_metric, vars(args) )

# print('Finished dvrl training.')

# # Outputs
# # Data valuation
# dve_out = dvrl_class.data_valuator(x_train, y_train)

# print('Finished data valuation.')

# df_train_noise['score'] = dve_out

# df_train_noise.to_csv("df_train_noise.csv", index=False)
# print(df_train_noise)


#df_train_noise = pd.read_csv("df_train_noise.csv")

# df_train_noise.sort_values(by=['score'], ascending=False, inplace=True)

# fpr, tpr, thresholds = metrics.roc_curve(df_train_noise['groudtruth'].values, \
#                                     df_train_noise['score'].values, pos_label=1)
# print('auc:', metrics.auc(fpr, tpr))

#df_train_noise.loc[df_train_noise['groudtruth']==0].sample(1)

# # Evaluations
# # 1. Data valuation
# # Data valuation
# sorted_idx = np.argsort(-dve_out)
# sorted_x_train = x_train[sorted_idx]

# # Indices of top n high valued samples
# print('Indices of top ' + str(n_exp) + ' high valued samples: '
#     + str(sorted_idx[:n_exp]))
# print(pd.DataFrame(data=sorted_x_train[:n_exp, :], index=range(n_exp),
#                  columns=col_names).head())

# # Indices of top n low valued samples
# print('Indices of top ' + str(n_exp) + ' low valued samples: '
#     + str(sorted_idx[-n_exp:]))
# print(pd.DataFrame(data=sorted_x_train[-n_exp:, :], index=range(n_exp),
#                  columns=col_names).head())

# # 2. Performance after removing high/low values
# # Here, as the evaluation model, we use LightGBM.
# eval_model = lightgbm.LGBMClassifier()

# # Performance after removing high/low values
# _ = dvrl_metrics.remove_high_low(dve_out, eval_model, x_train, y_train,
#                                x_valid, y_valid, x_test, y_test,
#                                'accuracy', plot=True)





