base_nli ={
    'politics':['Politics','War', 'Election','Constitution','Democracy','Conflict','Military',\
                'Terrorism', 'Government', 'Ideology', 'fascism', 'Socialism', 'Totalitarian', 'Religion'],
    'law':      ['Law', 'Legitimacy','Court','Crime','Murder','Jurisdiction'],
    'science':  ['Science','Aerospace','Physics','Chemistry','Biology','Scientist','Astronomy','Universe','Big Bang'],
    'technology':['Technology','Biotech', 'IT','Computers','Internet','Algorithm','Space','Bitcoin','artificial Intelligence','Robot'],
    'health': ['Health','Healthcare','Medicine','Clinics','Vaccine','Wellness','Nutrition','Dental','HIV','Disease'],
    'business': ['Business','Finance','Oil price','Supply','Inflation','Dollars','Bank','Wall Street','Bitcoin',
                        'Federal Reserve','Accrual','Accountancy','Sluggishness','Consumerism','Trade','Quarterly earnings',\
                         'Deposit','Revenue','Stocks','Recapitalization','Marketing','Futures'],
    'sports': ['Sports','Athletics','Championships','Football','Olympic','Tournament','Chelsea','League','Golf',
                            'NFL','Super bowl','World Cup'],
    'entertainment':['Entertainment','Pop music','Film','Music','Reality show','Drama','Concert','Rock music','Opera'],
    'education': ['Education', 'Tertiary education', 'University','Curriculum','Lecture'],
    'arts': ['Arts','Music','Painting','Art galleries','Classical music','Art Works','Stitchery'],
    'estate': ['Estate','Estate tax','Real estate']
}

def map_expand_nli(base_nli, dsn):
    label_expands_mannual = {}

    if dsn == 'ag':
        label_expands_mannual['World'] = base_nli['politics'] + base_nli['law']
        label_expands_mannual['Business'] = base_nli['business']
        label_expands_mannual['Sports'] = base_nli['sports']
        label_expands_mannual['science and technology'] = base_nli['science'] + base_nli['technology']

    elif dsn == 'uci':
        label_expands_mannual['entertainment'] = base_nli['entertainment']
        label_expands_mannual['business'] = base_nli['business']
        label_expands_mannual['science and technology'] = base_nli['science'] + base_nli['technology']
        label_expands_mannual['health'] = base_nli['health']

    elif dsn == 'nyt':
        label_expands_mannual['education'] = base_nli['education']
        label_expands_mannual['arts'] = base_nli['arts']
        label_expands_mannual['politics'] = base_nli['politics']
        label_expands_mannual['sports'] = base_nli['sports']
        label_expands_mannual['estate'] = base_nli['estate']
        label_expands_mannual['science'] = base_nli['science'] 
        label_expands_mannual['technology'] = base_nli['technology']
    return label_expands_mannual
import pandas as pd
import time,argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", default="yahoo", type=str)
parser.add_argument("--topk", default=100, type=int)
parser.add_argument("--manauto", default="auto", type=str)
parser.add_argument("--mode", default="test", type=str)
parser.add_argument("--calculate", default="sum", type=str)
parser.add_argument("--embed_cut", default=0.15, type=float)
parser.add_argument("--upper", default=0.85, type=float)
parser.add_argument("--lower", default=0.15, type=float)
parser.add_argument("--gpu", default="", type=str)

args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#from utils.flair_ners import *
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

from utils.load_data import * 
ds = load_data(dataset=args.dsn, samplecnt= 2048)
labels_candidates = ds.df_test['label_name'].unique().tolist()
print(labels_candidates)

from sklearn.metrics.pairwise import cosine_distances,cosine_similarity 
from utils.encoders import *
#if not gpus:
enc = encoder('dan','cpu')
# else:
#     enc = encoder('dan','gpu')


nli_model_name = "facebook/bart-large-mnli"

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_nli = AutoModelForSequenceClassification.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
tokenizer_nli = AutoTokenizer.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=len(gpus)-1)

# if not args.gram_diff:
#     from transformers import T5Tokenizer, AutoModelWithLMHead
#     tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
#     print(tokenizer_t5)
#     t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)    
#     gen_nlp  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=0)


# load dataset
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
# stopwords = stopwords.words('english')
stopwords = joblib.load("./utils/stopwords")

import numpy as np
import datasets,re,operator,joblib

def get_embedding_score(gram, df, enc):
    dfs = df.loc[df['title_lower'].str.contains(gram)]
    titles_contain = dfs.sample(min(2048, dfs.shape[0]))['title'].tolist()
    embeds = enc.infer(titles_contain, batch_size=1024)
    embeds_grams = enc.infer([gram])
    simi = cosine_similarity(embeds, embeds_grams).mean()
    return simi



'''
# prepare balanced df
fbs = 256
infos = []
titles = df.sample(frac=1)['title'].tolist()
for i in range(0, len(titles), fbs):
    torch.cuda.empty_cache()
    nli_result = nli_nlp(titles[i:i+fbs],  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
    for r in  nli_result:
        if r['scores'][0] >= 0.8:
            infos.append((r['sequence'], r['labels'][0] ))   
        
    if i % 2048 == 0:
        df_label = pd.DataFrame(infos, columns=['title','label']) 
        print(df_label['label'].value_counts())
        df_label.to_csv("df_cc_label.csv", index=False)
'''
from nltk.corpus import wordnet as wn
def check_noun(word):
    nets = wn.synsets(word)
    for net in nets:
        if net.name().split('.')[1] == 'n':
            return True 
    return False

############ find support seeds
import joblib,operator
import numpy as np

if args.mode == 'train':
    df = get_cc_news(1)
    df = df.loc[~df['title'].isnull()]
    df['title_lower'] = df['title'].map(lambda x: x.lower())
    gram_diff = {l:{} for l in labels_candidates}
    gram_embed = {}
    for ix, row in df.sample(frac=1).reset_index().iterrows():
        #row = ds.df_train.sample(1)
        #content = row['content'].tolist()[0]
        #label_name = row['label_name'].tolist()[0]
        if not row['title']:
            continue
        try:
            content = row['title'].lower()
        except:
            continue

        if not content or len(content.split(' ')) <=5:
            continue
        if re.search('[a-zA-Z]', content) is None:
            continue

        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=False)
        vectorizer.fit([content])
        grams = [g for g in vectorizer.get_feature_names() \
                        if g not in stopwords \
                            and g not in [ll.lower() for ll in labels_candidates] \
                            and not g.isdigit() \
                            and re.search('[a-zA-Z]', g) is not None \
                            and check_noun(g)]
        if not grams:
            continue
        #ners = get_ners(row['title'])
        #ners_ = [ner.lower() for ner in ners if len(ner.split(' '))>=2 and len(ner.split(' '))<=3]

        #grams = grams + ners_
        #print(grams)
        
        result_ori = nli_nlp(content, labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
        result_ori.pop('sequence')
        df_ori = pd.DataFrame(result_ori)

        # tune
        if df_ori['scores'].max() < 0.8:
            continue

        # gen_result = gen_nlp(content + tokenizer_t5.eos_token,  do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
        #                                         repetition_penalty=1.2, num_return_sequences= args.fbs,\
        #                                         clean_up_tokenization_spaces=True)
        # gen_contents = [s['generated_text'].lower() for s in gen_result if s['generated_text']]

        embeds = enc.infer([content])
        embeds_grams = enc.infer(grams)
        simis = cosine_similarity(embeds, embeds_grams)[0]
        df_gram_simis = pd.DataFrame(zip(grams, list(simis)), columns=['gram','simi'])
        #df_gram_simis['simi'] = (df_gram_simis['simi'] - df_gram_simis['simi'].min()) / (df_gram_simis['simi'].max()-df_gram_simis['simi'].min())
        df_gram_simis.sort_values(by=['simi'], ascending=False, inplace=True)
        # print(ix)
        # print("======>", content)
        # print("====>", grams)
        
        content_ = [content.replace(gram, '') for gram in grams]

        if len(content_)<=1:
            continue

        result_ = nli_nlp(content_,  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")

        for g,r in zip(grams, result_):
            # embedding score
            if not gram_embed.get(g, None):
                embed_score = get_embedding_score(g, df, enc)
                gram_embed[g] = embed_score

            if gram_embed[g] < args.embed_cut:
                continue

            r.pop('sequence')
            df_ = pd.DataFrame(r)
            df_diff = pd.merge(df_ori, df_, on=['labels'], how='inner') 
            df_diff['score_diff'] = df_diff['scores_x'] - df_diff['scores_y']
            df_diff_sel = df_diff.loc[(df_diff['scores_x']>=args.upper) & (df_diff['scores_y']<=args.lower)]
            
            if df_diff_sel.shape[0] == 0:
                continue
            print(g)
            print(df_diff_sel)
            for _, row in df_diff_sel.iterrows():
                if not gram_diff[row['labels']].get(g, None):
                    gram_diff[row['labels']][g] = []
                gram_diff[row['labels']][g].append(row['score_diff'])
            print("embed score:", gram_embed[g])
        print()


        if ix % 100 ==0 and ix > 0 :
            print(ix)
            label_expands = {}
            for l, gram_scores in gram_diff.items():
                gram_scores_mean = {g:round(np.array(scores).mean(),4) for g, scores in gram_scores.items() }
                gram_scores_mean_sort = sorted(gram_scores_mean.items(), key=operator.itemgetter(1), reverse=True) 
                print(l, '===>', gram_scores_mean_sort[:50])
                label_expands[l] = [ii[0] for ii in gram_scores_mean_sort[:20]]
            print('\n')

            # save to disk
            joblib.dump(gram_diff, 'gram_diff___{}'.format(args.dsn))
            joblib.dump(gram_embed, 'gram_embed___{}'.format(args.dsn))



elif args.mode == 'test':
    gram_diff = joblib.load("gram_diff___{}".format(args.dsn))
    gram_embed = joblib.load("gram_embed___{}".format(args.dsn))

    for args.topk in [32, 64, 100]:
        for args.calculate in ['sum', 'mean', 'max']:
            label_expands_auto = {}
            for l, gram_scores in gram_diff.items():
                if args.calculate == 'sum':
                    gram_scores_mean = {g:round(np.array(scores).sum(),4) for g, scores in gram_scores.items() }
                elif args.calculate == 'max':
                    gram_scores_mean = {g:round(np.array(scores).max(),4) for g, scores in gram_scores.items() }
                elif args.calculate == 'mean':
                    gram_scores_mean = {g:round(np.array(scores).mean(),4) for g, scores in gram_scores.items() }

                gram_scores_mean_sort = sorted(gram_scores_mean.items(), key=operator.itemgetter(1), reverse=True) 
                print(l, '===>', gram_scores_mean_sort[:100])
                 
                label_expands_auto[l] = [j[0] for j in gram_scores_mean_sort[:args.topk]]

            print(label_expands_auto)

            # assign
            if args.manauto == 'man':
                label_expands = map_expand_nli(base_nli, args.dsn)
            elif args.manauto == 'auto':
                label_expands = label_expands_auto


            # label_expands = {'business': ['market', 'ceo', 'bank', 'stocks', 'boss', 'sales', 'manager', 'stock', 'amazon', 'shares', 'profit', 'director', 'trump', 'businesses', 'news', 'apple', 'china', 'launches', 'team', 'million', 'loss', 'contract', 'google', 'workers', 'chief', 'restaurant', 'companies', 'banks', 'markets', 'jobs', 'growth', 'earnings', 'week', 'investment', 'launch', 'investors', 'economy', 'macron', 'robbery', 'employee', 'agent', 'revenue', 'dividend', 'production', 'billion', 'services', 'career', 'capital', 'workplace', 'partners', 'customers', 'india', 'finance', 'brewers', 'giants', 'executive', 'farm', 'chiefs', 'sells', 'reports', 'airlines', 'commercial', 'center', 'analysts'], 'science and technology': ['energy', 'technology', 'video', 'google', 'apple', 'study', 'science', 'internet', 'arsenal', 'china', 'launches', 'missile', 'electric', 'scientists', 'news', 'ceo', 'mobile', 'amazon', 'market', 'trump', 'space', 'smart', 'software', 'rockets', 'mining', 'launch', 'latest', 'moon', 'probe', 'gun', 'fireworks', 'researchers', 'technologies', 'zoo', 'lions', 'steel', 'million', 'brewers', 'future', 'stars', 'driving', 'birth', 'penguins', 'machine', 'bomb', 'eagles', 'chip', 'drone', 'korea', 'technical', 'shares', 'cloud', 'titans', 'surgery', 'clippers', 'production', 'giants', 'development', 'pipeline', 'fire', 'gold', 'nasa', 'wins', 'tesla'], 'entertainment': ['video', 'season', 'fans', 'stars', 'festival', 'music', 'sports', 'film', 'party', 'series', 'movie', 'club', 'trump', 'concert', 'nightclub', 'fireworks', 'eagles', 'actress', 'news', 'disney', 'hollywood', 'singer', 'zoo', 'rugby', 'channel', 'weekend', 'week', 'episode', 'bowl', 'league', 'album', 'beer', 'tickets', 'celebrity', 'fox', 'dance', 'song', 'parade', 'trailer', 'lions', 'theatre', 'events', 'hosts', 'streaming', 'comedian', 'summer', 'football', 'players', 'wins', 'director', 'movies', 'team', 'launches', 'fame', 'amazon', 'toys', 'theater', 'preview', 'magic', 'twitter', 'vegas', 'apple', 'performance', 'latest'], 'health': ['hospital', 'medical', 'cancer', 'healthcare', 'injury', 'doctor', 'flu', 'drug', 'heart', 'fitness', 'surgery', 'safety', 'doctors', 'blood', 'marijuana', 'hospitals', 'treatment', 'birth', 'injuries', 'patients', 'brain', 'news', 'medicaid', 'abortion', 'study', 'gym', 'nursing', 'concussion', 'safe', 'glasses', 'wounded', 'drugs', 'smoking', 'swim', 'risk', 'recovery', 'patient', 'smoke', 'diet', 'nurse', 'disease', 'dementia', 'save', 'clinic', 'knee', 'workout', 'medicine', 'week', 'exercise', 'police', 'disabled', 'poisoning', 'wellness', 'hurt', 'fire', 'healing', 'warning', 'illness', 'foods', 'vegan', 'swimming', 'autism', 'sports', 'therapy']}

            # label_expands = {'business': ['market', 'ceo', 'bank', 'stocks', 'boss', 'sales', 'manager', 'stock', 'amazon', 'shares', 'profit', 'director', 'trump', 'businesses', 'apple', 'china', 'launches', 'million', 'loss', 'contract', 'google', 'workers', 'chief', 'companies', 'banks', 'markets', 'jobs', 'growth', 'earnings', 'investment', 'launch', 'investors', 'economy', 'macron', 'robbery', 'employee', 'agent', 'revenue', 'dividend', 'production', 'billion', 'services', 'career', 'capital', 'workplace', 'partners', 'customers', 'india', 'finance', 'brewers', 'giants', 'executive', 'chiefs', 'sells', 'commercial', 'analysts'], \
            # 'science and technology': ['energy', 'technology', 'video', 'google', 'apple', 'study', 'science', 'internet', 'arsenal', 'china', 'launches', 'missile', 'electric', 'scientists', 'mobile', 'amazon', 'space', 'smart', 'software', 'rockets', 'mining', 'moon', 'probe', 'researchers', 'technologies',  'future', 'stars', 'machine', 'bomb', 'chip', 'drone', 'technical', 'cloud', 'titans', 'surgery', 'clippers', 'production', 'giants', 'development', 'pipeline', 'fire', 'gold', 'nasa', 'tesla'], 
            # 'entertainment': ['video', 'season', 'fans', 'stars', 'festival', 'music', 'sports', 'film', 'party', 'series', 'movie', 'club', 'trump', 'concert', 'nightclub', 'fireworks', 'actress', 'disney', 'hollywood', 'singer', 'rugby', 'channel', 'weekend', 'episode', 'bowl', 'league', 'album', 'beer', 'tickets', 'celebrity', 'fox', 'dance', 'song', 'parade', 'trailer', 'theatre', 'hosts', 'streaming', 'comedian', 'players', 'wins', 'director', 'movies', 'team', 'launches', 'fame', 'amazon', 'toys', 'theater', 'preview', 'magic', 'twitter', 'vegas', 'apple', 'performance'], \
            # 'health': ['hospital', 'medical', 'cancer', 'healthcare', 'injury', 'doctor', 'flu', 'drug', 'heart', 'fitness', 'surgery', 'safety', 'doctors', 'blood', 'marijuana', 'hospitals', 'treatment', 'birth', 'injuries', 'patients', 'brain', 'news', 'medicaid', 'abortion', 'gym', 'nursing', 'concussion', 'safe', 'glasses', 'wounded', 'drugs', 'smoking', 'swim', 'recovery', 'patient', 'smoke', 'diet', 'nurse', 'disease', 'dementia', 'clinic', 'knee', 'workout', 'medicine', 'exercise', 'police', 'disabled', 'poisoning', 'wellness', 'hurt', 'fire', 'healing', 'warning', 'illness', 'foods', 'vegan', 'swimming', 'autism', 'sports', 'therapy']}


            grams_candidates = []
            for l, grams in label_expands.items():
                grams_candidates.extend(grams)
            grams_candidates = list(set(grams_candidates))

            ######## evaluate ###########
            assert set(labels_candidates) == set(label_expands.keys())
            accs_noexpand = []
            accs_expand = []
            for ix, row in ds.df_train.reset_index().iterrows():
                content = row['content']

                nli_result = nli_nlp([content],  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")

                pred_label =  nli_result['labels'][0]
                if pred_label == row['label_name']:
                    accs_noexpand.append(1)
                else:
                    accs_noexpand.append(0)

                nli_result_ = nli_nlp([content],  grams_candidates, multi_label=True, hypothesis_template="This text is about {}.")
                nli_result_.pop('sequence')

                gram_score_dic = {gram:score for gram, score in zip(nli_result_['labels'], nli_result_['scores'])}

                infos = []
                for l in label_expands.keys():
                    l_score = np.array(([gram_score_dic[gram] for gram in label_expands[l]]  )).mean()
                    infos.append((l, l_score))

                df_expand = pd.DataFrame(infos, columns=['label','score'])

                pred_label = df_expand.sort_values(by=['score'], ascending=False).head(1)['label'].tolist()[0]

                if pred_label == row['label_name']:
                    accs_expand.append(1)
                else:
                    accs_expand.append(0)
                    # print(row['label_name'])
                    # print(content)
                    # print(df_expand.sort_values(by=['score'], ascending=False))
                    # print()

                if ix % 512 == 0 and ix > 0:
                    print(ix, sum(accs_noexpand) / len(accs_noexpand), sum(accs_expand)/len(accs_expand))

            print("final_summary==>", ' '.join(['{}:{}'.format(k, v) for k, v in vars(args).items()]),
                 sum(accs_noexpand) / len(accs_noexpand), sum(accs_expand)/len(accs_expand) )













