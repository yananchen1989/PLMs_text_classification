
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