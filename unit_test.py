



sent = "Federal jury orders tech giant Samsung to pay"

sent1 = 'FDA gives green light to migraine prevention tool'

from transformers import GPT2Tokenizer, GPT2LMHeadModel#TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)

gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
gpt2.trainable = False
gpt2.config.pad_token_id=50256
gen_nlp  = pipeline("text-generation", model=gpt2, tokenizer=tokenizer_gpt2, device=1, return_full_text=False)

ds.df_train.sample(10)['content'].tolist()

results_trunk = gen_nlp([sent], max_length=64, do_sample=True, top_p=0.9, top_k=0, temperature=1.0, \
                repetition_penalty=1.0, num_return_sequences=4, clean_up_tokenization_spaces=True, skip_special_tokens=True)









ds = load_data(dataset='dbpedia', samplecnt=1000)
(x_train, y_train),  (x_test, y_test), num_classes, label_idx = get_keras_data(ds.df_train.sample(10000), ds.df_test.sample(5000), sparse=True)
model = get_model_bert(num_classes)
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
df_train, df_test = train_test_split(df, test_size=0.125)

df_train[[str(i+1) for i in range(28)] + ['label']].to_csv('higgs_train.csv', index=False)
df_test[[str(i+1) for i in range(28)] + ['label']].to_csv('higgs_test.csv', index=False)


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




sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''





ds = load_data(dataset='ag', samplecnt= 32)


best_val_accs = []
models = []
from threading import Thread
threads = []
for ii in range(3):
    t = Thread(target=do_train_test_, args=(ds.df_train, ds.df_test, 100, 20, 0, 2, 64))
    t.start()
    threads.append(t)

# join all threads
for t in threads:
    t.join()
print("dvrl after join")






import requests
requests.get('https://huggingface.co/bert-base-uncased/resolve/main/config.json')
