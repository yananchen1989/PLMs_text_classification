# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main experiment for data valuation.

Main experiment of a data valuation application
using "Data Valuation using Reinforcement Learning (DVRL)"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

#import lightgbm
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import keras

from dvrl import data_loading
from dvrl import dvrl
from dvrl.dvrl_metrics import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)
#assert gpus

# Inputs for the main function
parser = argparse.ArgumentParser()

parser.add_argument('--dsn', default='adult', type=str)
parser.add_argument('--normalization',  choices=['minmax', 'standard'], default='minmax', type=str)

parser.add_argument( '--hidden_dim',  default=100, type=int)
parser.add_argument( '--comb_dim', default=10, type=int)
parser.add_argument( '--layer_number', help='number of network layers', default=5, type=int)

parser.add_argument('--train_no', default=1000, type=int)
parser.add_argument('--valid_no', default=400, type=int)

parser.add_argument('--perf_metric', default='auc', type=str)

parser.add_argument( '--iterations',  help='number of iterations',  default=2000,  type=int)
parser.add_argument(
    '--batch_size',
    help='number of batch size for RL',
    default=2000,
    type=int)
parser.add_argument(
    '--inner_iterations',
    help='number of iterations',
    default=100,
    type=int)
parser.add_argument(
    '--batch_size_predictor',
    help='number of batch size for predictor',
    default=256,
    type=int)
parser.add_argument(
    '--n_exp',
    help='number of examples',
    default=5,
    type=int)
parser.add_argument(
    '--learning_rate',
    help='learning rates for RL',
    default=0.01,
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
# Data loading and sample corruption

# The number of training and validation samples

# Network parameters
parameters = dict()
parameters['hidden_dim'] = args.hidden_dim
parameters['comb_dim'] = args.comb_dim
parameters['iterations'] = args.iterations
parameters['activation'] = tf.nn.relu
parameters['inner_iterations'] = args.inner_iterations
parameters['layer_number'] = args.layer_number
parameters['learning_rate'] = args.learning_rate
parameters['batch_size'] = args.batch_size
parameters['batch_size_predictor'] = args.batch_size_predictor

# The number of examples
n_exp = args.n_exp

# Checkpoint file name
args.checkpoint_file_name = './dvrl_tmp_{}/model.ckpt'.format(args.dsn)
# Data loading

if args.dsn not in ['blog', 'adult']:
    x_train = np.load('../dvrl_np_array/x_train_{}.npy'.format(args.dsn))
    x_valid = np.load('../dvrl_np_array/x_valid_{}.npy'.format(args.dsn))

    y_train = np.load('../dvrl_np_array/y_train_{}.npy'.format(args.dsn))
    y_valid = np.load('../dvrl_np_array/y_valid_{}.npy'.format(args.dsn))

    groudtruth = np.load('../dvrl_np_array/groudtruth_{}.npy'.format(args.dsn))
        
else:
    dict_no = dict()
    dict_no['train'] = args.train_no
    dict_no['valid'] = args.valid_no

    noise_idx = data_loading.load_tabular_data(args.dsn, dict_no, 0.2)
    # Extracts features and labels. Then, normalizes features
    x_train, y_train, x_valid, y_valid, x_test, y_test, col_names = \
    data_loading.preprocess_data('minmax', 'train.csv', 'valid.csv', 'test.csv')

    groudtruth = [0 if i in noise_idx else 1 for i in range(x_train.shape[0])]
    #df_train_noise = pd.DataFrame(groudtruth, columns=['groudtruth'])


print('Finished data preprocess.')

# Run DVRL
# Resets the graph
tf.reset_default_graph()
keras.backend.clear_session()

# Here, we assume a classification problem and we assume a predictor model
# in the form of a simple multi-layer perceptron.
problem = 'classification'
# Predictive model define
pred_model = keras.models.Sequential()
# pred_model.add(keras.layers.Dense(parameters['hidden_dim'],
#                                 activation='relu'))
# pred_model.add(keras.layers.Dense(parameters['hidden_dim'],
#                                 activation='relu'))
pred_model.add(keras.layers.Dense(np.unique(y_train).shape[0], activation='softmax'))
pred_model.compile(optimizer='adam', loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Flags for using stochastic gradient descent / pre-trained model
flags = {'sgd': True, 'pretrain': False}

# Initializes DVRL
dvrl_class = dvrl.Dvrl(x_train, y_train, x_valid, y_valid,
                     problem, pred_model, parameters,
                     args.checkpoint_file_name, flags)

# Trains DVRL
dvrl_class.train_dvrl(args.perf_metric)

print('Finished dvrl training.')

# Outputs
# Data valuation
dve_out = dvrl_class.data_valuator(x_train, y_train)

print('Finished data valuation.')

print(dve_out)


# df_train_noise.to_csv("df_train_noise.csv", index=False)
# print(df_train_noise)


#df_train_noise = pd.read_csv("df_train_noise.csv")

#df_train_noise.sort_values(by=['score'], ascending=False, inplace=True)


fpr, tpr, thresholds = metrics.roc_curve(groudtruth,  dve_out, pos_label=1)
#print(df_train_noise)
print('summary==>', args.dsn, ' '.join(['{}:{}'.format(k, v) for k, v in vars(args).items()]), \
   'auc:', metrics.auc(fpr, tpr) )




# Evaluations
# 1. Data valuation
# Data valuation
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





