import sys,os,logging,glob,pickle,torch,csv,datetime,gc,argparse,math,time,operator

parser = argparse.ArgumentParser()
parser.add_argument("--aug", default="eda", type=str)
parser.add_argument("--dsn", default="ag", type=str)
parser.add_argument("--samplecnt", default=64, type=int)
parser.add_argument("--lang", default="zh", type=str)
parser.add_argument("--temp", default=1.0, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--model", default="former", type=str)
parser.add_argument("--verbose", default=0, type=int)
parser.add_argument("--basemode", default="max", type=str) # rank or thres
parser.add_argument("--beams", default=1, type=int)
parser.add_argument("--rp", default=1.0, type=float)
#parser.add_argument("--nli_m", default="joeddav/bart-large-mnli-yahoo-answers", type=str)
parser.add_argument("--cbert_sample_ratio", default=0.3, type=float)
parser.add_argument("--epochs_ft", default=7, type=int)
parser.add_argument("--cap3rd", default=0.99, type=float)
parser.add_argument("--trunk_size", default=32, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--freq", default=20, type=int)
parser.add_argument("--boostsample_ft", default=0, type=int)
parser.add_argument("--cbertgpt_batct_size", default=8, type=int)
parser.add_argument("--nli_check", default=1, type=int)
#parser.add_argument("--dpp_retain", default=0.7, type=float)
parser.add_argument("--max_aug_times", default=4, type=int)
parser.add_argument("--basetry", default=3, type=int)
parser.add_argument("--eda_times", required=False, type=int, default=1, help="number of augmented sentences per original sentence")
parser.add_argument("--eda_sr", required=False, type=float, default=0.2, help="percent of words in each sentence to be replaced by synonyms")
parser.add_argument("--eda_ri", required=False, type=float, default=0.2, help="percent of words in each sentence to be inserted")
parser.add_argument("--eda_rs", required=False, type=float, default=0.2, help="percent of words in each sentence to be swapped")
parser.add_argument("--eda_rd", required=False, type=float, default=0.2, help="percent of words in each sentence to be deleted")
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--encm", default='dan', type=str, \
     choices=['dan', 'cmlm', \
     'paraphrase-distilroberta-base-v2','paraphrase-mpnet-base-v2','paraphrase-TinyBERT-L6-v2',\
     'paraphrase-MiniLM-L3-v2', 'average_word_embeddings_glove.6B.300d','nli-distilroberta-base-v2'])

args = parser.parse_args()
print('args==>', args)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import numpy as np
import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
from transformers import pipeline
#tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
#tf.keras.backend.set_floatx('float16')
import nltk 
from sklearn.metrics.pairwise import cosine_distances
#nltk.download('wordnet')

gpus = tf.config.experimental.list_physical_devices('GPU')
print('======>',gpus,'<=======')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)
#assert gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type=='cuda'

from utils.load_data import * 
from utils.transblock import * 
from utils.encoders import *
from utils.cbert_cgpt_config import * 
#from utils.dpp_model import * 
if args.aug == 'eda':
    from utils.eda import *

if args.aug == 'generate':
    # if args.generate_m == 'ctrl':
    #     args.rp = 1.2
    from utils.gan_config import *  
    print('generate model loaded')
    nlp_nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0) #  1.8.1+cu102
    print('nli model loaded')

if args.aug == 'bt':
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer_backward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-{}-en".format(args.lang), cache_dir="./cache", local_files_only=True)
    model_backward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-{}-en".format(args.lang), cache_dir="./cache", local_files_only=True)
    tokenizer_forward = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(args.lang), cache_dir="./cache", local_files_only=True)
    model_forward = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-{}".format(args.lang), cache_dir="./cache", local_files_only=True)
    nlp_backward = pipeline("translation", model=model_backward, tokenizer=tokenizer_backward, device=0)
    nlp_forward = pipeline("translation", model=model_forward, tokenizer=tokenizer_forward, device=0)
    print('bt model loaded')

if args.aug == 'fillin':
    from utils.aug_fillinmask import *

if args.aug == 'cbert':
    from utils.cbert_config import * 
    if args.dsn in ['yelp2','stsa','imdb', 'amazon2']:
        label_list_len = 2
    elif args.dsn == 'ag':
        label_list_len = 4
    elif args.dsn == 'yahoo':
        label_list_len = 10

    if label_list_len > 2:
        model.bert.embeddings.token_type_embeddings = torch.nn.Embedding(label_list_len, 768)
        model.bert.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)
    model.to(device)

if args.aug == 'cgpt':
    from utils.cgpt_config import * 


# def dpp_rerank(df_simi_filer, enc, dpp_retain):
#     embeds = enc.infer(df_simi_filer['content'].tolist())
#     sorted_ixs = extract_ix_dpp(embeds, df_simi_filer['simi'].values)
#     df_simi_filer_dpp = df_simi_filer.reset_index().iloc[sorted_ixs]
#     dpp_sents = df_simi_filer_dpp['content'].tolist()[:math.ceil(df_simi_filer_dpp.shape[0] * dpp_retain)]
#     return dpp_sents
def nli_classify(generated_text, label_name, expand_label_nli):
    assert label_name and  expand_label_nli
    if not generated_text or len(generated_text) <= 10:
        return 0, -99
    labels_candidates = []
    for label_expand in expand_label_nli.values():
        labels_candidates.extend(label_expand)
    result = nlp_nli(generated_text,  labels_candidates, multi_label=False, hypothesis_template="This text is about {}.")
    if result['labels'][0] in expand_label_nli[label_name]:
        return 1, result['scores'][0]
    else:
        return 0, result['scores'][0]



def synthesize(ds, proper_len, syn_df_ll):
    labels = ds.df_train['label'].tolist()
    contents = ds.df_train['content'].tolist()
    label_names = ds.df_train['label_name'].tolist()
    #gpt2_nlp_  = pipeline("text-generation", model='gpt2', device=device, return_full_text=False)

    if args.aug == 'generate':
        samples_syn_all = []
        for itr in range(100):     
            results = []
            for i in range(0, ds.df_train.shape[0], args.trunk_size):
                contents_trunk = contents[i:i+args.trunk_size]
                labels_trunk = labels[i:i+args.trunk_size] 
                # contents_trunk: list of contents
                results_trunk = gpt2_nlp(contents_trunk, max_length=200, do_sample=True, top_p=0.9, top_k=0, \
                        repetition_penalty=args.rp, num_return_sequences=4, clean_up_tokenization_spaces=True)
                results.extend(results_trunk)
                print('generate trunk==>', i, i+args.trunk_size, 'of', ds.df_train.shape[0])
            assert len(results) == ds.df_train.shape[0]

            buffer = []
            for ii in range(ds.df_train.shape[0]):
                for s in results[ii]:
                    generated_text = s['generated_text']
                    #print('==>', generated_text)
                    label = labels[ii]
                    label_name = label_names[ii]
                    if args.nli_check:
                        nli_check, nli_score = nli_classify(generated_text, label_name, expand_label_nli)
                        if nli_check:
                            buffer.append((generated_text, label, label_name, nli_score))
                    else:
                        buffer.append((generated_text, label, label_name, 0))

            print('itr:', itr , 'nli filter ratio:', len(buffer) / ds.df_train.shape[0] )

            samples_syn_all.extend(buffer)
            df_syn_tmp = pd.DataFrame(samples_syn_all, columns=['content','label','label_name','nli_score'])
            print(df_syn_tmp['label_name'].value_counts())

            if df_syn_tmp['label_name'].value_counts().values.min() >= args.samplecnt:
                df_syn_filter_ll = []
                for label_name in df_syn_tmp['label_name'].unique():
                    df_syn_tmp_l = df_syn_tmp.loc[df_syn_tmp['label_name']==label_name].copy()
                    df_syn_tmp_l.sort_values(by=['nli_score'], ascending=False, inplace=True) 
                    df_syn_filter_ll.append(df_syn_tmp_l.head(args.samplecnt))

                df_syn_filter = pd.concat(df_syn_filter_ll)
                
                samples_syn = [(ii[0],ii[1]) for ii in df_syn_filter[['content','label']].values]
                break

    elif args.aug == 'eda':
        aug_sentences = ds.df_train['content'].map(lambda x: eda(x, alpha_sr=args.eda_sr, alpha_ri=args.eda_ri, \
                                   alpha_rs=args.eda_rs, p_rd=args.eda_rd, num_aug=args.beams)).tolist()
        assert len(aug_sentences) == ds.df_train.shape[0] and len(aug_sentences[1]) == args.beams \
                and len(aug_sentences) == len(labels)
        samples_syn = []
        for ii in range(len(aug_sentences)):
            for sent in aug_sentences[ii]:
                samples_syn.append((sent, labels[ii]))

    elif args.aug == 'fillin':
        augmentor = fillInmask()
        samples_syn = []
        for b in range(args.beams):
            sentences = ds.df_train['content'].map(lambda x: augmentor.augment(x)).tolist()
            samples_syn.extend(list(zip(sentences, labels)))
            print('beam:', b)

    elif args.aug == 'bt':
        samples_syn = []
        for i in range(0, ds.df_train.shape[0], args.trunk_size):
            contents_trunk = contents[i:i+args.trunk_size]
            labels_trunk = labels[i:i+args.trunk_size]

            content_ =  nlp_forward(contents_trunk, truncation=True, \
                       do_sample=True, temperature=0.9, max_length=512, num_return_sequences=1)
            content__ =  nlp_backward([ii['translation_text'] for ii in content_], truncation=True, \
                        do_sample=True, max_length=512, temperature=0.9, num_return_sequences=1 )
            infos_trunk = list(zip([ii['translation_text'] for ii in content__], labels_trunk ))
            samples_syn.extend(infos_trunk)
            print('translate trunk==>', i, i+args.trunk_size, 'of', ds.df_train.shape[0])

    elif args.aug in ['cgpt','cbert']:
        temp_path = "augf__{}_{}".format(args.dsn, args.aug)
        temp_path_ft = "augf__{}_{}_ft".format(args.dsn, args.aug)
        
        write_for_cbert(ds.df_train, ds.df_test, temp_path, 0)
        write_for_cbert(ds.df_train, ds.df_test, temp_path_ft, args.boostsample_ft)

        processor = get_task_processor(args.dsn, temp_path)
        processor_ft = get_task_processor(args.dsn, temp_path_ft)

        label_list = processor.get_labels(args.dsn)
        # load train and dev data
        train_examples = processor.get_train_examples()
        train_examples_ft = processor_ft.get_train_examples()
        dev_examples = processor.get_dev_examples()   

        seed = int(time.time())

        if args.aug == 'cgpt':
            train_features =    convert_examples_to_features(train_examples, block_size, tokenizer, seed)
            train_features_ft = convert_examples_to_features(train_examples_ft, block_size, tokenizer, seed)
            dev_features =      convert_examples_to_features(dev_examples, block_size, tokenizer, seed)

        if args.aug == 'cbert':
            train_features =    convert_examples_to_features(train_examples, label_list, proper_len, tokenizer, seed)
            train_features_ft = convert_examples_to_features(train_examples_ft, label_list, proper_len, tokenizer, seed)
            dev_features =      convert_examples_to_features(dev_examples, label_list, proper_len, tokenizer, seed)

        # train data
        train_data = prepare_data(train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.cbertgpt_batct_size)

        train_data_ft = prepare_data(train_features_ft)
        train_sampler_ft = RandomSampler(train_data_ft)
        train_dataloader_ft = DataLoader(train_data_ft, sampler=train_sampler_ft, batch_size=args.cbertgpt_batct_size)

        #dev data
        dev_data = prepare_data(dev_features)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.cbertgpt_batct_size)

        print("***** Running training {} *****".format(args.aug))
        print("  Num examples = %d", len(train_features))
        print("  Batch size = %d", args.cbertgpt_batct_size)

        best_dev_loss = float('inf')
        #model_name = './{}_{}_best_{}.pt'.format(args.dsn, seed, args.aug)

        if args.aug == 'cgpt':
            # finetune
            if not syn_df_ll:
                for epoch in trange(args.epochs_ft, desc="Epoch"):
                    avg_loss = 0.
                    model.train()
                    for step, batch in enumerate(train_dataloader_ft):
                        batch = tuple(t.to(device) for t in batch)

                        inputs = {'input_ids': batch[0],
                                  'labels': batch[1]}

                        outputs = model(**inputs)
                        loss = outputs[0]
                        # loss = model(input_ids, segment_ids, input_mask, masked_ids)
                        optimizer.zero_grad()
                        loss.backward()
                        avg_loss += loss.item()
                        optimizer.step()
                        model.zero_grad()
                        if (step + 1) % 50 == 0:
                            print("avg_loss: {}".format(avg_loss / 50))
                        # avg_loss = 0.

                    # eval on dev after every epoch
                    dev_loss = compute_dev_loss(model, dev_dataloader)
                    print("Epoch {}, Dev loss {}".format(epoch, dev_loss))
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        print("Saving model. Best dev so far {}".format(best_dev_loss))
                        #torch.save(model.state_dict(), model_name)     

            print("generate augmentated samples")
            samples_syn = [] 
            prefix_size = prefix      
            prefix_text = None
            for ex_index, example in enumerate(train_examples):
                model.eval()
                if prefix_size > 0:
                    prefix_text = " ".join(example.text_a.split(' ')[:prefix_size])
                    prompt = example.label + SEP_TOKEN + prefix_text
                else:
                    prompt = example.label + SEP_TOKEN
                # print('cgpt example.text_a ==>', example.text_a)
                # print('cgpt prompt==>', prompt)
                context_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
                out = model.generate(
                    input_ids=context_tokens,
                    max_length=min(proper_len, tokenizer.model_max_length),
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=args.temp,
                    top_k=0,
                    top_p=0.9,
                    repetition_penalty=1.0,
                    pad_token_id=50256
                )

                out = out[:, len(context_tokens):].tolist()
                #for o in out:
                text = tokenizer.decode(out[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                aug_text = text.split(SEP_TOKEN.lower() )[-1]
                # eosn_index = 128
                # for stop_token in STOP_TOKENS:
                #     idx = text.find(stop_token)
                #     if idx > 0:
                #         eosn_index = min(eosn_index, idx)
                # text = text[: eosn_index]
                # text = text.replace("\n", " ").replace(EOS_TOKEN, ' ').strip()
                # if prefix_size > 0:
                #     text = prefix_text + " " + text
                samples_syn.append((aug_text, int(example.label)) )
                #print('cgpt samples_syn==>', aug_text, '<==', example.label, '\n')


        if args.aug == 'cbert':

            # finetune
            if not syn_df_ll:
                for epoch in trange(args.epochs_ft, desc="Epoch"):
                    avg_loss = 0.
                    model.train()
                    for step, batch in enumerate(train_dataloader_ft):
                        batch = tuple(t.to(device) for t in batch)
                        inputs = {'input_ids': batch[1],
                                  'attention_mask': batch[2],
                                  'token_type_ids': batch[3],
                                  'masked_lm_labels': batch[4] # yanan
                                  }

                        outputs = model(**inputs)
                        loss = outputs[0]
                        optimizer.zero_grad()
                        loss.backward()
                        avg_loss += loss.item()
                        optimizer.step()

                        if (step + 1) % 50 == 0:
                            print("avg_loss: {}".format(avg_loss / 50))
                        avg_loss = 0.

                    # eval on dev after every epoch
                    dev_loss = compute_dev_loss(model, dev_dataloader)
                    print("Epoch {}, Dev loss {}".format(epoch, dev_loss))
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        print("Saving model. Best dev so far {}".format(best_dev_loss))
                        #torch.save(model.state_dict(), model_name)

            train_sampler = SequentialSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.cbertgpt_batct_size)
            #model.load_state_dict(torch.load(model_name))

            print("generate augmentated samples")
            MASK_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            #tsv_writer = csv.writer(save_train_file, delimiter='\t')
            samples_syn = []
            for step, batch in enumerate(train_dataloader):
                model.eval()
                batch = tuple(t.to(device) for t in batch)
                init_ids, _, input_mask, segment_ids, _ = batch
                input_lens = [sum(mask).item() for mask in input_mask]
                masked_idx = np.squeeze(
                    [np.random.randint(0, l, max( int(l * args.cbert_sample_ratio), 1) ) for l in input_lens])
                for ids, idx in zip(init_ids, masked_idx):
                    ids[idx] = MASK_id
                # print('mask tokens', [ii.shape[0] for ii in masked_idx])
                # print('ori tokens', input_lens )
                inputs = {'input_ids': init_ids,
                          'attention_mask': input_mask,
                          'token_type_ids': segment_ids}

                outputs = model(**inputs)
                predictions = outputs[0]  # model(init_ids, segment_ids, input_mask)
                predictions = F.softmax(predictions / args.temp, dim=2)

                for ids, idx, preds, seg in zip(init_ids, masked_idx, predictions, segment_ids):
                    preds = torch.multinomial(preds, 1, replacement=True)[idx]
                    if len(preds.size()) == 2:
                        preds = torch.transpose(preds, 0, 1)
                    for pred in preds:
                        ids[idx] = pred
                        new_str = tokenizer.convert_ids_to_tokens(ids.cpu().numpy())
                        new_str = rev_wordpiece(new_str)
                        samples_syn.append((new_str, int(label_list[seg[0].item()])  ))
        
    else:
        raise KeyError("args.aug model illegal!")        
    print('samples_syn done...')
    df_synthesize = pd.DataFrame(samples_syn, columns = ['content','label'])
    assert df_synthesize.shape[0] == ds.df_train.shape[0] #* args.beams 
    if args.aug != 'generate':
        assert df_synthesize['label'].tolist() == ds.df_train['label'].tolist()

        for ix in random.sample(list(range(0, ds.df_train.shape[0])), 10):
            print('label==>',ds.df_train.iloc[ix]['label'], df_synthesize.iloc[ix]['label'])
            print('demo_{} ori sample==>'.format(args.aug), ds.df_train.iloc[ix]['content'])
            print('demo_{} aug sample==>'.format(args.aug), df_synthesize.iloc[ix]['content'])
            print()

    return df_synthesize 


# if args.aug in ['eda']:
#     enc = encoder('paraphrase-mpnet-base-v2','cuda')
# else:
#     enc = encoder('paraphrase-mpnet-base-v2','cpu')

def filterp(df_synthesize, model_base, ds):
    # step1 
    print("<==filterp 1==>")
    preds = model_base.predict(df_synthesize['content'].values, batch_size=args.batch_size, verbose=args.verbose)
    pred_l = np.argmax(preds, axis=1)
    df_synthesize['pred_label'] = pred_l
    df_synthesize_filter = df_synthesize.loc[df_synthesize['label'] == df_synthesize['pred_label']]
    print('self filter==>', df_synthesize_filter.shape[0], df_synthesize.shape[0], args.dsn, args.aug, args.beams )
    print('self vc==>\n', df_synthesize_filter['label'].value_counts() )
    filter_ratio_self = round(df_synthesize_filter.shape[0] / df_synthesize.shape[0], 4)

    # step 2
    print("<==filterp 2==>")
    enc = encoder(args.encm, 'cuda')
    label_embed = {}
    for l in ds.df_train['label'].unique():
        print('label==>', l)
        embeds_ = enc.infer(ds.df_train.loc[ds.df_train['label']==l]['content'].tolist(), batch_size=args.batch_size )
        label_embed[l] = embeds_
    embeds_syn =  enc.infer(df_synthesize_filter['content'].tolist(), batch_size=args.batch_size)
    infos = []
    test_labels = df_synthesize_filter['label'].tolist()
    contens = df_synthesize_filter['content'].tolist()
    for i in range(df_synthesize_filter.shape[0]):
        embed = embeds_syn[i].reshape(1,-1)
        label = test_labels[i]

        label_simi = {}
        for l, embed_anchor in label_embed.items():
            simi = cosine_distances(embed, embed_anchor) 
            label_simi[l] = simi.mean()

        if min(label_simi.items(), key=operator.itemgetter(1))[0] == label:
            infos.append((contens[i], test_labels[i]))
    df_synthesize_filter2 = pd.DataFrame(infos, columns = ['content','label'])
    print('cluster filter==>', df_synthesize_filter2.shape[0], df_synthesize_filter.shape[0], args.dsn, args.aug, args.beams)
    print('cluster vc==>\n', df_synthesize_filter2['label'].value_counts()  )
    filter_ratio_cluster = round(df_synthesize_filter2.shape[0] / df_synthesize_filter.shape[0], 4)
    return df_synthesize_filter2, filter_ratio_self, filter_ratio_cluster

#for args.dsn in ['ag','yahoo','stsa','yelp2','amazon2','imdb']:
ds = load_data(dataset=args.dsn, samplecnt= args.samplecnt)
ds, proper_len = process_ds(ds)
print(ds.df_train.sample(8))
proper_len = min(500, proper_len)
print('proper_len==>', proper_len)

if args.epochs > 0:
    best_val_acc_noaug, _ = do_train_test(ds.df_train, ds.df_test, args.epochs, args.freq, args.verbose, \
               args.basetry, args.samplecnt, args.basemode, args.model)
else:
    best_val_acc_noaug = -1


# if args.aug == 'no':
#     record_log('logg', \
#                  ['noaug==> '] + ['{}:{}'.format(k, v) for k, v in vars(args).items() if not k.startswith('eda_')] +\
#                       ['noaug_acc:{}'.format(best_val_acc_noaug)])
#     os._exit(0)

print("augmentating...")


syn_df_ll = []
accs_iters = []

while 1:

    df_synthesize = synthesize(ds, proper_len, syn_df_ll)

    syn_df_ll.append(df_synthesize)

    df_train_aug = pd.concat([ds.df_train] + syn_df_ll )

    aug_ratio = pd.concat(syn_df_ll).shape[0] / ds.df_train.shape[0]
    if int(aug_ratio) in [1, 2, 4]:
        acc_aug, _ = do_train_test(df_train_aug, ds.df_test, args.epochs, args.freq, args.verbose, \
                                args.basetry, args.samplecnt, args.basemode, args.model)
        accs_iters.append(acc_aug)

    if aug_ratio >= args.max_aug_times:
        print('max_aug_times terminated==>', accs_iters, args.dsn, args.aug)
        break


summary = ['summary===>'] + ['{}:{}'.format(k, v) for k, v in vars(args).items() if not k.startswith('eda_')] + \
    ['acc_base:{}'.format(best_val_acc_noaug), 'acc_aug:{}'.format(max(accs_iters))]


record_log('logb', summary)
print('success', ' '.join(summary))









