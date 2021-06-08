
for seed in range(5):
    for dsn in ['uci']:
        for ite  in range(7):  
            ds = load_data(dataset=dsn, samplecnt=100, seed=seed)
            (x_train, y_train),  (x_test, y_test), num_classes = get_keras_data(ds.df_train, ds.df_test)
            model = get_model_transormer(num_classes)

            history = model.fit(
                                x_train, y_train, batch_size=8, epochs=100, \
                                validation_batch_size=64,
                                validation_data=(x_test, y_test), verbose=0,
                                callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
                            )
            best_val_acc = max(history.history['val_acc'])
            print(seed, dsn, ite, best_val_acc) 



from load_data import * 
ds = load_data(dataset='ag', samplecnt=100, seed=45)
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TFGPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
modeltf = TFGPT2LMHeadModel.from_pretrained('gpt2')
model.train()
# when generating, we will use the logits of right-most token to predict the next token
# so the padding should be on the left
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token # to avoid an error
max_length = 300
prompts = ds.df_test.sample(10)['content'].tolist()
inputs = tokenizer(prompts, padding=True, truncation=True, max_length=max_len, return_tensors="tf")


output_sequences = modeltf.generate(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length= max_length,
    temperature=1,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1,
    do_sample=True,
    num_return_sequences=1,
)
assert output_sequences.shape[0] == len(prompts) and output_sequences[0].shape[0] == max_length

for seq in output_sequences:
    text_generated = tokenizer.decode(seq[len(encoded_prompt[0]):], clean_up_tokenization_spaces=True)

syn_sents = tokenizer.batch_decode(output_sequences, clean_up_tokenization_spaces=True, skip_special_tokens=True)


loss = black_box(output_sequences)
loss.backward()















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

















