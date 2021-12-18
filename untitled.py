import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity 
import tensorflow as tf 
gpus = tf.config.list_physical_devices('GPU')
print('======>',gpus,'<=======')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

from utils.encoders import *

enc = encoder('dan', 'gpu') 


from utils.load_data import * 
df = get_cc_news(1)
df = df.loc[~df['title'].isnull()]




