
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





#### paraphrasing 
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences,num_beams, maxlen):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=maxlen, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=maxlen,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

num_beams = 10
num_return_sequences = 10
context = '''
It's barely dawn when Mike Fitzpatrick starts his shift with a blur of colorful maps, figures and endless charts, but already he knows what the day will bring. Lightning will strike in places he expects. Winds will pick up, moist places will dry and flames will roar.
'''
get_response(context, num_return_sequences, num_beams, 100)




















