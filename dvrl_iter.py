import tensorflow as tf
import argparse,os
import numpy as np 
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd 
from utils.load_data import * 
parser = argparse.ArgumentParser()
parser.add_argument('--dsn', default='ag', type=str)
parser.add_argument( '--ite', default=1, type=int)
parser.add_argument( '--seed', default=333, type=int)
args = parser.parse_args()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

flags = {'sgd': True, 'pretrain': False}
parameters = dict()
parameters['hidden_dim'] = 100
parameters['comb_dim'] = 10
parameters['iterations'] = 2000 ###
parameters['activation'] = tf.nn.relu
parameters['inner_iterations'] = 100
parameters['layer_number'] = 5
parameters['learning_rate'] = 0.01
parameters['batch_size'] = 256
parameters['batch_size_predictor'] = 256
checkpoint_file_name = './tmp_{}/model.ckpt'.format(args.seed)

df_train_valid_noise = pd.read_csv("./dvrl_np_array/df_train_valid_noise_{}_{}.csv".format(args.dsn, args.seed))

df_syn = df_train_valid_noise.loc[df_train_valid_noise['groudtruth'].isin([9])]

df_train, df_valid = train_test_split(df_train_valid_noise.loc[df_train_valid_noise['groudtruth']==1], test_size=0.5)

df_train_1, df_train_0 = train_test_split(df_train, test_size=0.5)

df_train_0['label'] = df_train_0['label'].map(lambda x: noisy_label(x, df_train_valid_noise['label'].unique() )  )
df_train_0['groudtruth'] = 0


df_train_syn_noise = pd.concat([df_train_1, df_train_0, df_syn]).sample(frac=1)
assert df_train_syn_noise['groudtruth'].unique().shape[0] == 3


embed_shape = len([i for i in df_train_syn_noise.columns if i.startswith('embed_')])
embed_cols = ['embed_{}'.format(ii) for ii in range(embed_shape)]

x_train = df_train_syn_noise[embed_cols].values 
x_valid = df_valid[embed_cols].values

y_train, y_valid = df_train_syn_noise['label'].values, df_valid['label'].values
print("embeding inferred ite:{}".format(args.ite))

from dvrl import dvrl_v2
# Run DVRL
# Resets the graph
tf.compat.v1.reset_default_graph()
tf.compat.v1.keras.backend.clear_session()


# Predictive model define
pred_model = tf.compat.v1.keras.models.Sequential()
pred_model.add(tf.compat.v1.keras.layers.Dense(np.unique(y_train).shape[0], activation='softmax'))
pred_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Flags for using stochastic gradient descent / pre-trained model


# Initializes DVRL
dvrl_class = dvrl_v2.Dvrl(x_train, y_train, x_valid, y_valid,
                     'classification', pred_model, parameters,
                     checkpoint_file_name , flags, args.seed+args.ite)

# Trains DVRL
dvrl_class.train_dvrl('accuracy')

print('ite:{} Finished dvrl training.'.format(args.ite))

# Outputs
# Data valuation
dve_out = dvrl_class.data_valuator(x_train, y_train)
df_train_syn_noise['dve_out'] = dve_out
print('ite:{} Finished data valuation.'.format(args.ite))

df_train_noise = df_train_syn_noise.loc[df_train_syn_noise['groudtruth'].isin([0,1])]

fpr, tpr, thresholds = metrics.roc_curve(df_train_noise['groudtruth'].values,  df_train_noise['dve_out'].values, pos_label=1)
auc = metrics.auc(fpr, tpr)
mean_dve_out = np.array(dve_out).mean()
std_dve_out = np.array(dve_out).std()
print('dvrl iter:{} auc:{} mean:{} std:{}'.format(args.ite, auc, mean_dve_out, std_dve_out))
#np.save("./dvrl_np_array/dve_out_{}_{}_{}.npy".format(args.dsn, args.seed, args.ite), np.array(dve_out))
df_train_syn_noise.to_csv("./dvrl_np_array/df_train_noise_{}_{}_{}_{}.csv"\
    .format(args.dsn, args.seed, args.ite, round(auc, 2)), index=False)










