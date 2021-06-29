
target_path = '/root/topic_classification_augmentation/TransformersDataAugmentation/src/utils/datasets'

for dsn in ['ag','yahoo','dbpedia']:
    ds = load_data(dataset=dsn, samplecnt=-1)

    df_train, df_dev = train_test_split(ds.df_train, test_size=0.3)

    df_train['content'] = df_train['content'].map(lambda x: x.replace('\t',' ').replace('\n',' '))
    df_dev['content'] = df_dev['content'].map(lambda x: x.replace('\t',' ').replace('\n',' '))
    ds.df_test['content'] = ds.df_test['content'].map(lambda x: x.replace('\t',' ').replace('\n',' '))


    df_train[['label','content']].to_csv(target_path+'/{}/train.tsv'.format(dsn), sep='\t', header=None, index=False)
    ds.df_test[['label','content']].to_csv(target_path+'/{}/test.tsv'.format(dsn), sep='\t', header=None, index=False)
    df_dev[['label','content']].to_csv(target_path+'/{}/dev.tsv'.format(dsn), sep='\t', header=None, index=False)






cc_news = datasets.load_dataset('cc_news', split="train")
dfcc = pd.DataFrame(cc_news['text'], columns=['content'])
dfcnndm = pd.read_csv("../datasets_aug/cnn_dailymail_stories.csv")
#dfcc = pd.concat([dfcc, dfcnndm])
df_batch = dfcnndm.sample(32)


ds = load_data(dataset='dbpedia', samplecnt=1000)
(x_train, y_train),  (x_test, y_test), num_classes, label_idx = get_keras_data(ds.df_train.sample(10000), ds.df_test.sample(5000), sparse=True)
model = get_model_bert(num_classes, 'albert')
model.compile(Adam(lr=1e-5), 'sparse_categorical_crossentropy', metrics=["acc"])

model.fit(
        x_train, y_train, batch_size=32, epochs=50, \
        validation_data=(x_test, y_test), verbose=1,
        callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
    )

intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                 outputs=model.get_layer('keras_layer_2').output['pooled_output'])



preds = intermediate_layer_model.predict(x_test, verbose=1, batch_size=64)
dff = pd.DataFrame(preds, columns=['c{}'.format(ii) for ii in range(preds.shape[1])])
dff['label'] = y_test
dff.to_csv('mnnbenchdata_test.csv',index=False)

from sklearn.model_selection import train_test_split
import pandas as pd 
df = pd.read_csv("HIGGS.csv.gz", error_bad_lines=False, header=None, nrows=800000) #  0.6371
df.columns = ['label'] + [str(i+1) for i in range(28)]
df_train, df_test = train_test_split(df, test_size=0.2)

labels_train = df_train.pop('label').values
labels_test = df_test.pop('label').values
input_embed = keras.Input(shape=(df_train.shape[1], ))
outputs = layers.Dense(1, activation="sigmoid")(input_embed)
model = keras.Model(inputs=input_embed, outputs=outputs)
model.compile('adam', 'binary_crossentropy', metrics=["acc"])
model.fit(
        df_train.values, labels_train, batch_size=32, epochs=50, \
        validation_data=(df_test.values, labels_test), verbose=1,
        callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
    )



df_train = pd.read_csv('mnnbenchdata_train.csv')
df_test = pd.read_csv('mnnbenchdata_test.csv')

input_embed = keras.Input(shape=(df_train.shape[1], ))
outputs = layers.Dense(14, activation="softmax")(input_embed)
model = keras.Model(inputs=input_embed, outputs=outputs)
model.compile('adam', 'sparse_categorical_crossentropy', metrics=["acc"])

feat_columns = [c for c in df_train.columns if c != 'label']
model.fit(
        df_train[feat_columns].values, df_train['label'].values, batch_size=32, epochs=50, \
        validation_data=(df_test[feat_columns].values, df_test['label'].values), verbose=1,
        callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
    )





sent = "Edelman Partners. New York NY\n\nJ.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc\n\nFrequent speaker, former U.S. Ambassador"

content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''

























