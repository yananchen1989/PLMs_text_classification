


#nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# 

#content = "Johnson Helps D-Backs End Nine-Game Slide (AP) AP - Randy Johnson took a four-hitter into the ninth inning to help the Arizona Diamondbacks end a nine-game losing streak Sunday, beating Steve Trachsel and the New York Mets 2-0."
# generate selected samples which belong to the defined categories



import os, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="", type=str)
parser.add_argument("--check", default=1, type=int)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--model", default="", type=str)
parser.add_argument("--thres", default=0.7, type=float)
args = parser.parse_args()

print("args==>", args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from load_data import * 
from transblock import * 
print(torch.__version__)



args.dsn = 'yahoo'
args.model = 'gpt2'
ds = load_data(dataset=args.dsn, samplecnt=-1)
labels = ds.df.label.unique()

#files = glob.glob("./generation_samples_{}*.tsv".format(args.model))
infos = []
with open('pseudos_{}.tsv'.format(args.model),'r') as f:
    for line in f:
        if '\t' not in line:
            continue 

        tokens = line.strip().split('\t') 
        
        if len(tokens)!=3 or tokens[0].strip() not in labels:
            continue
        
        content = tokens[1].strip()
        label = tokens[0].strip()
        score = float(tokens[2].strip())
        if score < args.thres:
            continue 

        infos.append((label, content))

df = pd.DataFrame(infos, columns=['label','content'])
print(df.label.value_counts())
print("df==>", df.shape[0])

assert set(list(ds.df_test.label.unique())) == set(list(df['label'].unique()))

(x_train, y_train),  (x_test, y_test), num_classes = get_keras_data(df, ds.df_test)
model = get_model_transormer(num_classes)

print("train begin==>")
history = model.fit(
    x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), verbose=1, validation_batch_size=64,
    callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
)
best_val_acc = max(history.history['val_acc'])
print('dsn:', args.dsn, 'check:', args.check, 'model:', args.model)
print("iter completed, tranin acc ==>{}".format(best_val_acc))
print("training cnt==", df_all.shape[0])
















