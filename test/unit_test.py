sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''


tokenizer(sent, padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=500,  
                                      return_tensors='pt') 



import os, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="yahoo", type=str)
parser.add_argument("--gpu", default=0, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
from utils.transblock import * 
from utils.load_data import * 

ds = load_data(dataset=args.dsn, samplecnt= -1)
(x_train, y_train),  (x_test, y_test)= get_keras_data(ds.df_train, ds.df_test)
model = get_model_bert(ds.df_test['label'].unique().shape[0])
history = model.fit(
        x_train, y_train, batch_size=32, epochs=30, \
        validation_data=(x_test, y_test), verbose=1, validation_batch_size=64, validation_freq=5)




model.save_weights('./cls/model_former_{}.h5'.format(args.dsn) )


model.load_weights('model_full_{}.h5'.format(args.dsn))

model.evaluate()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer.padding_side = "left" 
#tokenizer.pad_token = tokenizer.eos_token
gpt2_ft = GPT2LMHeadModel.from_pretrained('finetune_gpt2_ppo_ag_1024_1628217816', local_files_only=True)
gpt2_ft.trainable = False
gpt2_ft.config.pad_token_id=50256

gpt2_ori = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
gpt2_ori.trainable = False
gpt2_ori.config.pad_token_id=50256


from transformers import pipeline
gpt2_ori_nlp  = pipeline("text-generation", model=gpt2_ori, tokenizer=tokenizer, device=0, return_full_text=False)
gpt2_ft_nlp  = pipeline("text-generation", model=gpt2_ft, tokenizer=tokenizer, device=0, return_full_text=False)

dfs = ds.df_train.sample(50)



#gpt2_imdb_pos  = pipeline("text-generation", model=gpt2_imdb_pos, tokenizer=tokenizer, device=0, return_full_text=False)
#gpt2_imdb_ctrl  = pipeline("text-generation", model=gpt2_imdb_ctrl, tokenizer=tokenizer, device=0, return_full_text=False)

continuations = gpt2_ft_nlp(dfs['content'].tolist(), max_length=256, do_sample=True, top_p=0.0, top_k=0, \
                    repetition_penalty=2.0, num_return_sequences=1, clean_up_tokenization_spaces=True)

# for i in zip(dfs['content'].tolist(), [ii[0]['generated_text'] for ii in continuations]):
#     print('ori==>', i[0])
#     print('syn==>', i[1])
#     print('\n')

gpt2_ft.to(device)
for ix, row in ds.df_train.sample(frac=1).iterrows():
    prompt = row['label_name'] + ' {} '.format(tokenizer.bos_token) + ' '.join(row['content'].split(' ')[:3])
    print('ori ==> ', row['content'])

    context_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    out = gpt2_ft.generate(
        input_ids=context_tokens,
        max_length=tokenizer.model_max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        pad_token_id=50256
    )

    out = out[:, len(context_tokens):].tolist()
    #for o in out:
    text = tokenizer.decode(out[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
    aug_text = text.split(tokenizer.bos_token )[-1]
    print('syn ==> ', aug_text)
    print('label==>', row['label_name'])
    print('\n')








# import pandas as pd
# df = pd.read_csv("imdb-dataset.csv")
# imdb_str = " <|endoftext|> ".join(df['review'].tolist())

# with open ('imdb.txt', 'w') as f:
#     f.write(imdb_str)
























