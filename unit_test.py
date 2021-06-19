

for samplecnt in [100, 1000]:
    for dsn in ['ag','yahoo']:
        for ite  in range(3):  
            ds = load_data(dataset=dsn, samplecnt=samplecnt)
            (x_train, y_train),  (x_test, y_test), num_classes, label_idx = get_keras_data(ds.df_train, ds.df_test)
            model = get_model_bert(num_classes)

            history = model.fit(
                                x_train, y_train, batch_size=64, epochs=100, \
                                validation_batch_size=64,
                                validation_data=(x_test, y_test), verbose=1,
                                callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
                            )
            best_val_acc = max(history.history['val_acc'])
            print(samplecnt, dsn, ite, best_val_acc) 








from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
import tensorflow as tf
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = TFAlbertForSequenceClassification.from_pretrained('albert-base-v2')
inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1)) # Batch size 1
outputs = model(inputs)
loss = outputs.loss
logits = outputs.logits






nlp = pipeline("ner", model="flair/ner-english-fast")
nlp = pipeline("ner", model="dslim/bert-base-NER")
content = random.sample(contents,1)[0]
print(content)
nlp(content)



content = "Edelman Partners. New York NY\n\nJ.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc\n\nFrequent speaker, former U.S. Ambassador"

content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''




# https://github.com/GT-SALT/MixText/blob/master/data/yahoo_answers_csv/back_translate.ipynb
import torch
while 1:
    en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
    ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')


en2ru.cuda()
ru2en.cuda()

en2de.cuda()
de2en.cuda()

ru2en.translate(en2ru.translate(content,  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)







prompts = [s.decode() for s in prompts.numpy()] 
labels = list(labels.numpy())



