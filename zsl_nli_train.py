


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
parser.add_argument("--check", default=1, type=int)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--model", default="", type=str)
args = parser.parse_args()

print("args==>", args)
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

infos = []
with open('zsl_{}_contents.tsv'.format(args.model),'r') as f:
    for line in f:
        if '\t' not in line:
            continue 

        tokens = line.strip().split('\t') 
        if len(tokens)!=4:
            continue
        content = tokens[-1].strip()
        dsn = tokens[0].strip()
        label = tokens[1].strip()
        code = tokens[2].strip()
        if dsn != args.dsn:
            continue
        if args.check and args.dsn in ['ag', 'yahoo']:
            if not check_premise(content, [code]) :
                continue
        if args.dsn != 'pop':
            infos.append((int(label), content))
        else:
            infos.append((label, content))
            
df = pd.DataFrame(infos, columns=['label','content'])
# df.to_csv("df_nli_filter_{}_{}.csv".format(args.model, args.dsn), index=False)
# print(args.model, ' ', args.dsn, '==>', df.shape[0])

# args.dsn = 'yahoo'
# args.model = 'ctrl'

#df = pd.read_csv("df_nli_filter_{}_{}.csv".format(args.model, args.dsn))

#df = df.sample(146057)

ds = load_data(dataset=args.dsn, samplecnt=-1)

if args.dsn == 'ag':
    ds.df_test = ds.df_test.loc[ds.df_test['label']!=1]
    ds.df_train = ds.df_train.loc[ds.df_train['label']!=1]

assert set(list(ds.df_test.label.unique())) == set(list(df['label'].unique()))

df_all = pd.concat([df, ds.df_train])
(x_train, y_train),  (x_test, y_test), num_classes = get_keras_data(df_all, ds.df_test)
model = get_model_bert(num_classes)

print("train begin==>")
history = model.fit(
    x_train, y_train, batch_size=64, epochs=12, validation_data=(x_test, y_test), verbose=1,
    callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
)
best_val_acc = max(history.history['val_acc'])
print('dsn:', args.dsn, 'check:', args.check, 'model:', args.model)
print("iter completed, tranin acc ==>{}".format(best_val_acc))
print("training cnt==", df_all.shape[0])
















