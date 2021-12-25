

from nltk.corpus import wordnet as wn
import joblib,operator
import numpy as np 


import gensim
model_google = gensim.models.KeyedVectors.load_word2vec_format('./resource/GoogleNews-vectors-negative300.bin',binary=True)


from utils.load_data import * 

dsn = 'yahoo'
ds = load_data(dataset=dsn, samplecnt= 2048)
labels_candidates = ds.df_train['label_name'].unique().tolist()
print(labels_candidates)


from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache__')
bert_nsp = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', cache_dir='./cache__')


def nsp_infer(sent1, sent2, bert_nsp, bert_tokenizer):
    encoding = bert_tokenizer(sent1, sent2, return_tensors='pt', max_length=512, truncation=True)

    outputs = bert_nsp(**encoding, labels=torch.LongTensor([1]))
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.detach().numpy()[0][0]



infos = []
while 1:
    labels = random.sample(labels_candidates, 2)
    sent1 = ds.df_train.loc[ds.df_train['label_name']==labels[0]].sample(1)['content'].tolist()[0]


    sent2 = "This text is about {}.".format(labels[0])
    nsp_prob12 = nsp_infer(sent1, sent2, bert_nsp, bert_tokenizer)
    nsp_prob21 = nsp_infer(sent2, sent1, bert_nsp, bert_tokenizer)
    nsp_prob = (nsp_prob12+nsp_prob21) / 2

    sent2_ = "This text is about {}.".format(labels[1])
    nsp_prob12_ = nsp_infer(sent1, sent2_, bert_nsp, bert_tokenizer)
    nsp_prob21_ = nsp_infer(sent2_, sent1, bert_nsp, bert_tokenizer)
    nsp_prob_ = (nsp_prob12_+nsp_prob21_ ) / 2

    infos.append((nsp_prob, nsp_prob_))

    if len(infos) > 0 and len(infos) % 50  == 0:

        result = np.array(infos)
        print(result[:, 0].mean(), result[:, 1].mean())


from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_nli = AutoModelForSequenceClassification.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
tokenizer_nli = AutoTokenizer.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=-1)

label_expands = label_expands_auto

grams_candidates = []
for l, grams in label_expands.items():
    grams_candidates.extend(grams)
grams_candidates = list(set(grams_candidates))

######## evaluate ###########
assert set(labels_candidates) == set(label_expands.keys())
accs_noexpand = []
accs_expand = []
for ix, row in ds.df_train.reset_index().iterrows():
    content = row['content']

    # ori
    nli_result = nli_nlp([content],  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
    pred_label =  nli_result['labels'][0]
    if pred_label == row['label_name']:
        accs_noexpand.append(1)
    else:
        accs_noexpand.append(0)

    nli_result_ = nli_nlp([content],  grams_candidates, multi_label=True, hypothesis_template="This text is about {}.")
    nli_result_.pop('sequence')

    gram_score_dic = {gram:score for gram, score in zip(nli_result_['labels'], nli_result_['scores'])}

    infos = []
    for l in label_expands.keys():
        l_score = np.array(([gram_score_dic[gram] for gram in label_expands[l]]  )).mean()
        infos.append((l, l_score))

    df_expand = pd.DataFrame(infos, columns=['label','score'])

    pred_label = df_expand.sort_values(by=['score'], ascending=False).head(1)['label'].tolist()[0]

    if pred_label == row['label_name']:
        accs_expand.append(1)
    else:
        accs_expand.append(0)
        # print(row['label_name'])
        # print(content)
        # print(df_expand.sort_values(by=['score'], ascending=False))
        # print()

    if ix % 32 == 0 and ix > 0:
        print(ix, sum(accs_noexpand) / len(accs_noexpand), sum(accs_expand)/len(accs_expand))

print("final_summary==>", ' '.join(['{}:{}'.format(k, v) for k, v in vars(args).items()]),
     sum(accs_noexpand) / len(accs_noexpand), sum(accs_expand)/len(accs_expand) )





embed_label = enc.infer(["Business"])
embed_grams = enc.infer([ii[0] for ii in grams])

simis = cosine_similarity(embed_label, embed_grams)

df_simi = pd.DataFrame(zip([ii[0] for ii in grams], list(simis[0])), columns=['gram', 'simi'])




