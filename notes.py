#ner 

import random
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load("flair/ner-english-fast")


text = '''
Reuters/Vincest West Lucasfilm is eyeing a date for an Obi-Wan Kenobi solo.\nLucasfilm is already eyeing a production date for an Obi-Wan Kenobi solo film.\nThere are plans to create a solo debut for Obi-Wan Kenobi from the "Star Wars" franchise, and according to Omega Underground, Lucasfilm Productions is setting a film date in January 2019. The set location is expected to be at Pinewood Studios in the United Kingdom.\nMeanwhile, the anticipated Obi-Wan debut also has a working title, which is "Joshua Tree." The working title offers hints that the film will feature scenes when Obi-Wan was watching over a young Luke Skywalker in Tatooine.\nAccording to Screenrant, the story of Obi-Wan in his solo film could take place between the saga\'s episode 3, "Star Wars: Revenge of the Sith," and episode 4, "A New Hope." However, there might be difficulty in finding the perfect timeline for the Obi-Wan solo.\nObi-Wan\'s adventures have been featured in many other platforms aside from the "Star Wars" saga. His character and adventures already appeared in the animated "Star Wars: The Clone Wars," as well as the "Star Wars Rebels" TV series. Hopefully the upcoming Obi-Wan spin-off won\'t collide with other stories from the "Star Wars" franchise.\nThere are speculations that Stephen Daldry, director of "The Hours" and "The Reader" is currently in talks to spearhead the upcoming Obi-Wan solo film.\nAlthough it is yet to be confirmed, fans are already expecting Ewan McGregor to fulfil the role of Obi-Wan in his solo debut. There are talks that McGregor was not given the opportunity to embody the character of Obi-Wan in the prequel trilogy of "Star Wars," and that reprising the role in a solo would give him that chance.\nAside from that, fans won\'t be happy if Lucasfilm hires another actor to play Obi-Wan, since he is the original portrayer.\nThe director and screenwriter for the Obi-Wan Kenobi solo film are yet to be confirmed.
'''

def get_ners(text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    ners = list(set([ii['text'] for ii in sentence.to_dict(tag_type='ner')['entities']]))
    random.shuffle(ners)
    return '<=>'.join(ners)

ners = get_ners(text)

df = pd.read_csv("./torch_ds/df_cc_news.csv", lineterminator='\n')

df['ners'] = df['content'].map(lambda x: get_ners(x))
df.to_csv("./torch_ds/df_cc_news_ners.csv", index=False)


    


# translation
from fairseq.models.transformer import TransformerModel
en_de_model = TransformerModel.from_pretrained("transformer.wmt19.en-de",
    tokenizer='moses',
    bpe='fastbpe', cache_dir="./cache_wmt19", local_files_only=True
)

de_en_model = TransformerModel.from_pretrained("transformer.wmt19.de-en",
    tokenizer='moses',
    bpe='fastbpe', cache_dir="./cache_wmt19", local_files_only=True
)

import torch
# List available models
torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]

# Load a transformer trained on WMT'16 En-De
# Note: WMT'19 models use fastBPE instead of subword_nmt, see instructions below
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de',
                       tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en',
                       tokenizer='moses', bpe='fastbpe')








#rouge
rouge = datasets.load_metric('rouge')
predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]
results = rouge.compute(predictions=predictions, references=references)
print(list(results.keys()))
    ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
print(results["rouge1"])
AggregateScore(low=Score(precision=1.0, recall=1.0, fmeasure=1.0), mid=Score(precision=1.0, recall=1.0, fmeasure=1.0), high=Score(precision=1.0, recall=1.0, fmeasure=1.0))
print(results["rouge1"].mid.fmeasure)








# kl divergence

y_true = np.random.randint(0, 2, size=(1, 3)).astype(np.float64)
y_pred = np.random.random(size=(1, 3))
p = np.array([[0.99, 0.005, 0.005]])
q = np.array([[0.005, 0.99, 0.005]])

p = np.array([1] + 15*[0])
q = np.array(15*[0] + [1])
m = (p+q)/2
loss_pm = tf.keras.losses.kullback_leibler_divergence(p, m)
loss_qm = tf.keras.losses.kullback_leibler_divergence(q, m)
jsd = (loss_pm.numpy() + loss_qm.numpy())/2
print(jsd)





# token generation by step
from transformers import TFAutoModelWithLMHead, AutoTokenizer, tf_top_k_top_p_filtering
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = TFAutoModelWithLMHead.from_pretrained("gpt2")

sequence = f"Hugging Face is based in DUMBO, New York City, and "

input_ids = tokenizer.encode(sequence, return_tensors="tf")

# get logits of last hidden state
next_token_logits = model(input_ids)[0][:, -1, :]

# filter
filtered_next_token_logits = tf_top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

# sample
next_token = tf.random.categorical(filtered_next_token_logits, dtype=tf.int32, num_samples=1)

generated = tf.concat([input_ids, next_token], axis=1)

resulting_string = tokenizer.decode(generated.numpy().tolist()[0])








#  tokenizer
tokens = tokenizer.tokenize(text) 
# ['hello', 'world', '!']
token_ids = tokenizer.convert_tokens_to_ids(tokens)
# [7592, 2088, 999]

# Encode and Encode+
tokens = tokenizer.encode(text)
# [101, 7592, 2088, 999, 102]

tokens = tokenizer.encode_plus(text)
# {'input_ids': [101, 7592, 2088, 999, 102], 'token_type_ids': [0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1]}

tokens = tokenizer.encode(text, max_length=512, padding='max_length', return_tensors='pt')
# tensor([ 101, 7592, 2088,  999,  102,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0])


tokens = tokenizer.encode_plus(text, max_length=512, padding='max_length',
                               return_tensors='pt')
tokens.keys()
# dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
tokens['input_ids'][0,:50]
# tensor([ 101, 7592, 2088,  999,  102,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0])



# For Batches
text_list = ['hello world!', 'hello mars!']
tokens = tokenizer.batch_encode_plus(text_list)
# {'input_ids': [[101, 7592, 2088, 999, 102], [101, 7592, 7733, 999, 102]], 
#    'token_type_ids': [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 
#    'attention_mask': [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]}

tokens = tokenizer.batch_encode_plus(text_list, max_length=8,
                                     padding='max_length',
                                     return_tensors='pt')
# {'input_ids': tensor([[ 101, 7592, 2088,  999,  102,    0,    0,    0],
#         [ 101, 7592, 7733,  999,  102,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0, 0, 0]])}

# calling our tokenizer class directly.
tokens = tokenizer(text)
# {'input_ids': [101, 7592, 2088, 999, 102], 'token_type_ids': [0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1]}
tokens = tokenizer(text_list)
# {'input_ids': [[101, 7592, 2088, 999, 102], [101, 7592, 7733, 999, 102]], 'token_type_ids': [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]}



# https://github.com/anderskm/gputil
import GPUtil
GPUtil.showUtilization()

deviceIDs = GPUtil.getAvailable(order = 'memory', limit = len(gpus), maxLoad = 1, maxMemory = 0.9, includeNan=False, excludeID=[], excludeUUID=[])

# Get the first available GPU
DEVICE_ID_LIST = GPUtil.getFirstAvailable()
DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list

# Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

Gpus = GPUtil.getGPUs()
for gpu in Gpus:
    print('gpu.id:', gpu.id)
    print('GPU总量：', gpu.memoryTotal)
    print('GPU使用量：', gpu.memoryUsed)
    print('gpu使用占比:', gpu.memoryUtil * 100)



while True:
    memoryUtil = min([gpu.memoryUtil for gpu in Gpus])
    if memoryUtil < 0.3:
        break
    else:
        time.sleep(60)




# log 
#日志
import datetime
def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
logging.Formatter.converter = beijing


logging.basicConfig(
    filename='log_test',
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger()