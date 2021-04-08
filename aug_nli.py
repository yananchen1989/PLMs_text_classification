#from transformers import AutoModelForSequenceClassification, AutoTokenizer
#https://huggingface.co/joeddav/bart-large-mnli-yahoo-answers
# MODEL = "joeddav/bart-large-mnli-yahoo-answers"
# #'facebook/bart-large-mnli'  'joeddav/xlm-roberta-large-xnli'
# from transformers import BartForSequenceClassification, BartTokenizer
# tokenizer = BartTokenizer.from_pretrained(MODEL)
# nli_model = BartForSequenceClassification.from_pretrained(MODEL)
# nli_model.to(device)



'''
infos = []
ix = 0
while ix < len(sentences):
    premises = sentences[ix:ix+batch_size]

    for cate in labels:
        hypothesises = ['This text is about {}.'.format(cate)] * batch_size
        # run through model pre-trained on MNLI
        x = tokenizer(premises, hypothesises, return_tensors='pt', padding=True ,truncation=True,max_length=64)

        logits = nli_model(**x.to(device))[0] 
        entail_contradiction_logits = logits[:,[0,2]].softmax(dim=1)
        true_prob = entail_contradiction_logits[:,1].cpu().detach().numpy()
        
        if true_prob.max() >= 0.9: 
            pos = np.where(true_prob>=0.9)[0]
            for p in pos:
                #print(premises[p])
                infos.append((premises[p], cate))
    ix += batch_size
    logger.info('ix==>{}'.format(ix))

    gc.collect()


'''


#nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# 

#content = "Johnson Helps D-Backs End Nine-Game Slide (AP) AP - Randy Johnson took a four-hitter into the ninth inning to help the Arizona Diamondbacks end a nine-game losing streak Sunday, beating Steve Trachsel and the New York Mets 2-0."
# generate selected samples which belong to the defined categories
# import csv
# save_train_file = open('cnn_dm_nli.csv', 'w')
# writer = csv.writer(save_train_file, delimiter='\t')
'''
for content in sentences:
    result = nlp(content, labels_candidate, multi_label=False, hypothesis_template="This text is about {}.")

    if result['scores'][0] < 0.4:
        continue
    writer.writerow([content.strip().replace('\t',' '), result['labels'][0]])
'''







import os, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="", type=str)
parser.add_argument("--check", default=True, type=bool)
parser.add_argument("--gpu", default="1", type=str)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from load_data import * 
from transblock import * 

from transformers import pipeline
nlp = pipeline("zero-shot-classification", model="joeddav/bart-large-mnli-yahoo-answers", device=0) #  



def check_premise(content, labels_candidate):
    result = nlp(content, labels_candidate, multi_label=False, hypothesis_template="This text is about {}.")
    if result['scores'][0] >= 0.7:
        return True 
    else:
        return False




label_ix = {label:ix+1 for ix, label in enumerate(labels[:10])}

for args.dsn in ['yahoo','ag']*10:
    infos = []
    with open('gpt_zsl.tsv','r') as f:
        for line in f:
            if '\t' not in line:
                continue 

            tokens = line.strip().split('\t') 
            if len(tokens)!=2:
                print(line)
                continue
            content = tokens[1].strip()

            if args.dsn == 'yahoo':
                if tokens[0].strip() not in label_ix.keys():
                    continue
                label = label_ix[tokens[0].strip()]
                
            if args.dsn == 'pop':
                if tokens[0].strip() not in labels[10:14]:
                    continue
                label = tokens[0].lower().replace('president ','').strip()


            if args.dsn == 'ag':
                if tokens[0].strip() == "world news":
                    label = 1
                    continue # do not use this category
                elif tokens[0].strip() == "Sports":
                    label = 2
                elif tokens[0].strip() in ["Business","Business & Finance"]:
                    label = 3
                elif tokens[0].strip() in ["Science & Mathematics", "science and technology"]:
                    label = 4
                else:
                    continue 
            if args.check and args.dsn in ['ag', 'yahoo']:
                if not check_premise(content, [tokens[0].strip()]) :
                    continue

            infos.append((label, content))

    ds = load_data(dataset=args.dsn, samplecnt=100)
    # train
    df = pd.DataFrame(infos, columns=['label','content'])

    #df.to_csv("ag_nli_filter.csv", index=False)

    #df = pd.read_csv("yahoo_nli_filter.csv")
    if args.dsn == 'ag':
        df = df.loc[df['label']!=1]
        ds.df_test = ds.df_test.loc[ds.df_test['label']!=1]

    assert set(list(ds.df_test.label.unique())) == set(list(df['label'].unique()))
    (x_train, y_train),  (x_test, y_test), num_classes = get_keras_data(df, ds.df_test)
    model = get_model_albert(num_classes)

    print("train begin==>")
    history = model.fit(
        x_train, y_train, batch_size=64, epochs=12, validation_data=(x_test, y_test), verbose=1,
        callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
    )
    best_val_acc = max(history.history['val_acc'])
    print('dsn:', args.dsn, 'check:{}', args.check)
    print("iter completed, tranin acc ==>{}".format(best_val_acc))
    print("training cnt==", df.shape[0])
















