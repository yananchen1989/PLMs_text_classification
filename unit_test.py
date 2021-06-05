args.aug = 'generate'
args.beams = 128
args.thres = 0.6



for dsn in ['uci','yahoo']:
    for ite  in range(7):  
        ds = load_data(dataset=dsn, samplecnt=-1)
        (x_train, y_train),  (x_test, y_test), num_classes = get_keras_data(ds.df_train, ds.df_test)
        model = get_model_transormer(num_classes)

        history = model.fit(
                            x_train, y_train, batch_size=64, epochs=100, \
                            validation_batch_size=64,
                            validation_data=(x_test, y_test), verbose=1,
                            callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
                        )
        best_val_acc = max(history.history['val_acc'])
        print(dsn, ite, best_val_acc) 

for samplecnt in [1000, -1]:
    for ite in range(3):
        for dsn in ['ag','stsa','dbpedia', 'pop','uci','yahoo']:
            ds = load_data(dataset=dsn, samplecnt=1000)
            (x_train, y_train),  (x_test, y_test), num_classes = get_keras_data(ds.df_train, ds.df_test)
            #model = get_model_transormer(num_classes)
            #model.save_weights('weights.h5')
            for batch_size in [64, 32, 8]:
                model = get_model_transormer(num_classes)
                #model.load_weights('weights.h5')

                history = model.fit(
                    x_train, y_train, batch_size=batch_size, epochs=50, \
                    validation_batch_size=64,
                    validation_data=(x_test, y_test), verbose=0,
                    callbacks = [EarlyStopping(monitor='val_acc', patience=3, mode='max')]
                )
                best_val_acc = max(history.history['val_acc'])
                print(dsn, samplecnt, batch_size, best_val_acc)






for dsn in ['ag','stsa','dbpedia','pop','uci','yahoo']:

    ds = load_data(dataset=dsn, samplecnt=64)
    max_len = get_tokens_len(ds, args.cap3rd)

    contents = ds.df_train['content'].tolist()
    labels = ds.df_train['label'].tolist()
    labels_candidates = list(ds.df.label.unique())

    results = []
    for i in range(0, ds.df_train.shape[0], args.trunk_size):
        contents_trunk = contents[i:i+args.trunk_size]
        labels_trunk = labels[i:i+args.trunk_size] 
        results_trunk = nlp(contents_trunk, max_length=max_len, do_sample=True, top_p=0.9, top_k=0, \
                repetition_penalty=args.rp, num_return_sequences=args.beams)
        results.extend(results_trunk)
        print('generate trunk==>', i, i+args.trunk_size, 'of', ds.df_train.shape[0])
    assert len(results) == ds.df_train.shape[0] and len(results[0]) == args.beams
    retain = []
    for ii in range(ds.df_train.shape[0]):
        if args.check in ['enc','double']:
            ori_sentence = contents[ii]
            ori_embed = enc.infer([ori_sentence])
            syn_sentences = [sentence['generated_text'] for sentence in results[ii]]
            syn_embeds = enc.infer(syn_sentences)
            simis = cosine_similarity(ori_embed, syn_embeds)
            df_simi = pd.DataFrame(zip(syn_sentences, simis[0]), columns=['content','simi'])
            df_simi.sort_values(by=['simi'], ascending=False, inplace=True)
            df_simi_filer = df_simi.loc[df_simi['simi']>= args.thres]
            df_simi_filer_enc = df_simi_filer
            #print(df_simi.shape[0], '==>', df_simi_filer.shape[0])
            retain.append((df_simi.shape[0], df_simi_filer.shape[0]))

        if args.check in ['nli','double']:
            infos_trunk = []
            for sentence in results[ii]:
                if not sentence['generated_text']:
                    continue
                result_nli = nlp_nli(sentence['generated_text'], labels_candidates, multi_label=False, hypothesis_template="This text is about {}.")
                if result_nli['scores'][0] >= args.thres and result_nli['labels'][0] == labels[ii]:                    
                    infos_trunk.append((sentence['generated_text'], result_nli['scores'][0] ))
            df_simi_filer = pd.DataFrame(infos_trunk, columns=['content','simi'])
            if df_simi_filer.shape[0] == 0:
                print(args.dsn, 'nli ==> noleft')
                continue                     
            df_simi_filer_nli = df_simi_filer

    assert len(retain) == ds.df_train.shape[0]
    df_retain = pd.DataFrame(retain, columns=['ori','fil'])

    print(dsn,  df_retain['fil'].sum() / df_retain['ori'].sum(), df_retain['fil'].sum() / ds.df_train.shape[0])










nlp  = pipeline("text-generation", model='gpt2', device=-1, return_full_text=False)

max_len = get_tokens_len(ds, 0.99)
results_trunk = nlp([content], max_length=max_len, do_sample=True, top_p=0.9, top_k=0, \
                    repetition_penalty=1, num_return_sequences=256)


enc = encoder('dan')

ori_sentence = ds.df_test.sample(1)['content'].tolist()[0]

ori_embed = enc.infer([ori_sentence])
syn_sentences = [sent['generated_text'] for sent in results_trunk]
syn_embeds = enc.infer(syn_sentences)
simis = cosine_similarity(ori_embed, syn_embeds)
df_simi = pd.DataFrame(zip(syn_sentences, simis[0]), columns=['content','simi'])
df_simi.sort_values(by=['simi'], ascending=False, inplace=True)
df_simi_filer = df_simi.loc[df_simi['simi']>= 0.6]


embeds = enc.infer(df_simi_filer['content'].tolist())

sorted_ixs = extract_ix_dpp(embeds, df_simi_filer['simi'].values)
df_simi_filer_dpp = df_simi_filer.reset_index().iloc[sorted_ixs]






infos = [('aaa',1), ('bbb',2), ('ccc',3)]
df_simi_filer = pd.DataFrame(infos, columns=['content','simi'])
df_simi_filer_enc = df_simi_filer


infos = [('aaa',1), ('zzz',2), ('ddd',3)]
df_simi_filer = pd.DataFrame(infos, columns=['content','simi'])
df_simi_filer_nli = df_simi_filer

df_simi_filer_enc.join(df_simi_filer_nli)

df_simi_filer = pd.merge(df_simi_filer_enc, df_simi_filer_nli, on='content', how='inner')


tokenizer.tokenize(content)

content = ds.df_test['content'].tolist()[10]


nlp = pipeline("fill-mask" , model = 'distilbert-base-cased', device=0)


text_mask = text.replace('Fassa Bortolo', nlp.tokenizer.mask_token, 1)

pred_tokens = nlp(text_mask)

text_mask.replace( nlp.tokenizer.mask_token, pred_tokens[0]['token_str'])


ds.df_train.sample(10000)['content'].map(lambda x: augmentor.augment(x)) 

augmentor = fillInmask(ner_set= 0 )






results = nlp(ds.df.sample(1000)['content'].tolist(), max_length=128, do_sample=False)
print(results[0]['translation_text'])




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

















