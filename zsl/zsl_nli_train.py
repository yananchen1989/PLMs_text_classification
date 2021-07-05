


#nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# 

#content = "Johnson Helps D-Backs End Nine-Game Slide (AP) AP - Randy Johnson took a four-hitter into the ninth inning to help the Arizona Diamondbacks end a nine-game losing streak Sunday, beating Steve Trachsel and the New York Mets 2-0."
# generate selected samples which belong to the defined categories



import os, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="yahoo", type=str)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--model", default="gpt2", type=str)
#parser.add_argument("--thres", default=0.7, type=float)
args = parser.parse_args()

print("args==>", args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from load_data import * 
from transblock import * 
print(torch.__version__)


for args.dsn in ['yahoo','dbpedia','nyt','pop','20news','uci']:

    ds = load_data(dataset=args.dsn, samplecnt=-1)
    labels = ds.df.label.unique()

    #args.thres = 0
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
            # if score < 0.7:
            #     continue 

            infos.append((label, content, score))

    df_synthe = pd.DataFrame(infos, columns=['label','content', 'score'])

    #ds.df_train['score'] = 1

    assert set(list(ds.df_test.label.unique())) == set(list(df_synthe['label'].unique()))
    #print(df_synthe.label.value_counts())

    #df_fuse = pd.concat([df_synthe, ds.df_train])

    sample_over = df_synthe.label.value_counts().mean()  / ds.df_train.label.value_counts().mean()
    print('dsn==>', args.dsn)
    print("sample_over==>", sample_over)
    print('df_synthe per class==>', df_synthe.label.value_counts().mean())

    (x_train, y_train),  (x_test, y_test), num_classes = get_keras_data(df_synthe, ds.df_test)

    model = get_model_transormer(num_classes)

    history = model.fit(
        x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), verbose=1, validation_batch_size=64,
        sample_weight = df_synthe['score'].values, #use_multiprocessing=True, workers=8,
        callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
    )
    best_val_acc = max(history.history['val_acc'])
    print(df_synthe.label.value_counts())
    print('dsn:', args.dsn, 'model:', args.model, ' acc:', best_val_acc )








#0.7: 0.4520
#0.5: 0.44871
# weights: 0.4635






