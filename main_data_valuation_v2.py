from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

#import lightgbm
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

# Inputs for the main function
parser = argparse.ArgumentParser()
parser.add_argument('--dsn', default='stsa', type=str)
parser.add_argument('--samplecnt', default=128, type=int)
parser.add_argument('--noise_cnt', default=32, type=int)
parser.add_argument('--perf_metric', default='accuracy', type=str)

parser.add_argument(
    '--hidden_dim',
    help='dimensions of hidden states',
    default=512,
    type=int)
parser.add_argument(
    '--comb_dim',
    help='dimensions of hidden states after combinding with prediction diff',
    default=10,
    type=int)

parser.add_argument(
    '--outer_iterations',
    help='number of iterations',
    default=1000,
    type=int)
parser.add_argument(
    '--batch_size',
    help='number of batch size for RL',
    default=256,
    type=int)
parser.add_argument(
    '--inner_iterations',
    help='number of iterations',
    default=7,
    type=int)
parser.add_argument(
    '--batch_size_predictor',
    help='number of batch size for predictor',
    default=32,
    type=int)
parser.add_argument(
    '--n_exp',
    help='number of examples',
    default=5,
    type=int)
parser.add_argument(
    '--learning_rate',
    help='learning rates for RL',
    default=0.001,
    type=float)
parser.add_argument(
    '--checkpoint_file_name',
    help='file name for saving and loading the trained model',
    default='./tmp/model.ckpt',
    type=str)

args = parser.parse_args()

"""Main function of DVRL for data valuation experiment.

Args:
args: data_name, train_no, valid_no,
      normalization, network parameters, number of examples
"""

# Network parameters
# parameters = dict()
# parameters['iterations'] = args.iterations
# parameters['inner_iterations'] = args.inner_iterations
# parameters['learning_rate'] = args.learning_rate
# parameters['batch_size'] = args.batch_size
# parameters['batch_size_predictor'] = args.batch_size_predictor

# The number of examples
n_exp = args.n_exp

# Checkpoint file name
checkpoint_file_name = args.checkpoint_file_name

# Data loading
#_ = data_loading.load_tabular_data(data_name, dict_no, 0.0)



from sklearn.model_selection import train_test_split
from utils.load_data import * 
from utils.encoders import *
from utils.transblock import * 
ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)

df_train_, df_valid = train_test_split(ds.df_train, test_size=0.3)
df_train_['groudtruth'] = 1


df_noise = ds.df_test.sample(args.noise_cnt)
df_noise['label'] = df_noise['label'].map(lambda x: noisy_label(x, ds.df_test['label'].unique() )  )
df_noise['groudtruth'] = 0 


df_train_noise = pd.concat([df_train_, df_noise])

enc = encoder('cmlm-large','cpu')
x_train = enc.infer(df_train_noise['content'].values)
x_valid = enc.infer(df_valid['content'].values)

y_train, y_valid = df_train_noise['label'].values, df_valid['label'].values

print('Finished data loading.')


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


from dvrl import dvrl_v2
# Run DVRL
# Resets the graph
tf.compat.v1.reset_default_graph()
tf.compat.v1.keras.backend.clear_session()

# Here, we assume a classification problem and we assume a predictor model
# in the form of a simple multi-layer perceptron.
# Predictive model define
pred_model = tf.compat.v1.keras.models.Sequential()
# pred_model.add(keras.layers.Dense(parameters['hidden_dim'],
#                                 activation='relu'))
# pred_model.add(keras.layers.Dense(parameters['hidden_dim'],
#                                 activation='relu'))
pred_model.add(tf.compat.v1.keras.layers.Dense(ds.df_test.label.unique().shape[0], activation='softmax'))
pred_model.compile(optimizer='adam', loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Flags for using stochastic gradient descent / pre-trained model
flags = {'sgd': True, 'pretrain': False}

# Initializes DVRL
dvrl_class = dvrl_v2.Dvrl(x_train, y_train, x_valid, y_valid,
                     'classification', pred_model, vars(args),
                     checkpoint_file_name, flags)

# Trains DVRL
dvrl_class.train_dvrl(args.perf_metric)

print('Finished dvrl training.')

# Outputs
# Data valuation
dve_out = dvrl_class.data_valuator(x_train, y_train)

print('Finished data valuation.')

df_train_noise['score'] = dve_out

# df_train_noise.to_csv("df_train_noise.csv", index=False)
# print(df_train_noise)


#df_train_noise = pd.read_csv("df_train_noise.csv")

df_train_noise.sort_values(by=['score'], ascending=False, inplace=True)

fpr, tpr, thresholds = metrics.roc_curve(df_train_noise['groudtruth'].values, \
                                    df_train_noise['score'].values, pos_label=1)
print('auc:', metrics.auc(fpr, tpr))

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





