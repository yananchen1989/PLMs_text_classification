
def nsp_infer(sent1, sent2, bert_nsp, bert_tokenizer):
    encoding = bert_tokenizer(sent1, sent2, return_tensors='pt', max_length=256, truncation=True).to(device0)
    outputs = bert_nsp(**encoding, labels=torch.LongTensor([1]).cpu().to(device0) )
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.cpu().detach().numpy()[0][0]



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



from utils.transblock import *

with tf.distribute.MirroredStrategy().scope():
    bert_nsp  = get_model_nsp(256)

pairs = [[row['content'], sent] for sent in contents_syn]
pairs_ids = get_ids(pairs, 256, tokenizer_bert )
preds = bert_nsp.predict(pairs_ids, batch_size=8)
nsp_scores = preds[:,0]






