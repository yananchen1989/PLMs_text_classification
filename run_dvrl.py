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

parser = argparse.ArgumentParser()

parser.add_argument('--dsn', default='stsa', type=str)
parser.add_argument( '--samplecnt',  default=128, type=int)
parser.add_argument( '--samplecnt_bad',  default=128, type=int)

parser.add_argument( '--hidden_dim',  default=100, type=int)
parser.add_argument( '--comb_dim', default=10, type=int)
parser.add_argument( '--layer_number', help='number of network layers', default=5, type=int)

parser.add_argument('--perf_metric', default='accuracy', type=str)

parser.add_argument( '--iterations',  help='number of iterations',  default=2000,  type=int)
parser.add_argument(
    '--batch_size',
    help='number of batch size for RL',
    default=256,
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

# Data loading
#from dvrl import data_loading

###########################dsn 
from sklearn.model_selection import train_test_split
from utils.load_data import * 
from utils.encoders import *
#from utils.transblock import * 

enc = encoder('cmlm-large')
print('cmlm loaded')


##################################
ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)
df_train, df_valid = train_test_split(ds.df_train, test_size=0.4)
df_train['groudtruth'] = 1


df_noise = ds.df_test.sample(args.samplecnt_bad)
df_noise['label'] = df_noise['label'].map(lambda x: noisy_label(x, ds.df_test['label'].unique() )  )
df_noise['groudtruth'] = 0 

df_train_noise = pd.concat([df_train, df_noise]).sample(frac=1)


x_train, x_valid = enc.infer(df_train_noise['content'].values), enc.infer(df_valid['content'].values)
y_train, y_valid = df_train_noise['label'].values, df_valid['label'].values


from dvrl import dvrl_v2
# Run DVRL
# Resets the graph
tf.compat.v1.reset_default_graph()
tf.compat.v1.keras.backend.clear_session()

# Here, we assume a classification problem and we assume a predictor model
# in the form of a simple multi-layer perceptron.
problem = 'classification'
# Predictive model define
pred_model = tf.compat.v1.keras.models.Sequential()
# pred_model.add(keras.layers.Dense(parameters['hidden_dim'],
#                                 activation='relu'))
# pred_model.add(keras.layers.Dense(parameters['hidden_dim'],
#                                 activation='relu'))
pred_model.add(tf.compat.v1.keras.layers.Dense(np.unique(y_train).shape[0], activation='softmax'))
pred_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Flags for using stochastic gradient descent / pre-trained model
flags = {'sgd': True, 'pretrain': False}

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


# Initializes DVRL
dvrl_class = dvrl_v2.Dvrl(x_train, y_train, x_valid, y_valid,
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



fpr, tpr, thresholds = metrics.roc_curve(df_train_noise['groudtruth'].values,  dve_out, pos_label=1)
#print(df_train_noise)
print('summary==>', args.dsn, ' '.join(['{}:{}'.format(k, v) for k, v in vars(args).items()]), \
   'auc:', metrics.auc(fpr, tpr), 'mean_dve_out:', np.array(dve_out).mean(), \
   'std_dve_out:', np.array(dve_out).std() )






