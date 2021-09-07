#ner 
from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load("flair/ner-english-large")


def get_ners(text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    ners = list(set([ii['text'] for ii in sentence.to_dict(tag_type='ner')['entities']]))
    return ners




    


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
p = np.array([[0.10, 0.40, 0.50]])
q = np.array([[0.80, 0.15, 0.05]])

m = (p+q)/2
loss_pm = tf.keras.losses.kullback_leibler_divergence(p, m)
loss_qm = tf.keras.losses.kullback_leibler_divergence(q, m)
jsd = (loss_pm.numpy()[0] + loss_qm.numpy()[0])/2
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







