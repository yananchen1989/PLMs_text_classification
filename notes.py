#ner 




    


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


# write and read json
import json
data = {
    'employees' : [
        {
            'name' : 'John Doe',
            'department' : 'Marketing',
            'place' : [1,2,3,4]
        },
        {
            'name' : 'Jane Doe',
            'department' : 'Software Engineering',
            'place' : [1,2,3,4]
        },
        {
            'name' : 'Don Joe',
            'department' : 'Software Engineering',
            'place' : 'Office'
        }
    ]
}


json_string = json.dumps(data)

for _ in range(5):
    with open('test.json', 'a') as f:
        f.write(json_string+'\n')


with open('test.json', 'r') as f:
    for line in f:
        data = json.loads(line.strip()) 
        print(data)











