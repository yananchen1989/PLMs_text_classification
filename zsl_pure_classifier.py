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
parser.add_argument("--std_cut", default=0.1, type=float)
parser.add_argument("--topk", default=100, type=int)
parser.add_argument("--gram_diff_file", default="gram_diff_constrain", type=str)
parser.add_argument("--gpu", default="0", type=str)
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
ds = load_data(dataset='uci', samplecnt= 2048)
labels_candidates = ds.df_test['label_name'].unique().tolist()

from sklearn.metrics.pairwise import cosine_distances,cosine_similarity 
from utils.encoders import *
enc = encoder('cmlm-base')


nli_model_name = "facebook/bart-large-mnli"

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_nli = AutoModelForSequenceClassification.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
tokenizer_nli = AutoTokenizer.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=len(gpus)-1)




# load dataset
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
# stopwords = stopwords.words('english')
stopwords = joblib.load("./utils/stopwords")

import numpy as np
import datasets,re,operator,joblib
try:
    cc_news = datasets.load_dataset('cc_news', split="train", cache_dir='./torch_ds')
    df = pd.DataFrame(zip(cc_news['title'], cc_news['text'], cc_news['description'] ))
    df.columns = ['title','content','description']
    df.drop_duplicates(['title','content'], inplace=True) 
    df.to_csv("./torch_ds/df_cc_news.csv", index=False)
except:
    df = get_cc_news(1)

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



############ find support seeds
gram_diff = {l:{} for l in labels_candidates}

for ix, row in df.sample(frac=1).reset_index().iterrows():
    #row = ds.df_train.sample(1)
    #content = row['content'].tolist()[0]
    #label_name = row['label_name'].tolist()[0]
    content = row['title'].lower()
    if not content or len(content.split(' ')) <=5:
        continue
    if re.search('[a-zA-Z]', content) is None:
        continue

    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=False)
    vectorizer.fit([content])
    grams = vectorizer.get_feature_names_out().tolist()
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

    embeds = enc.infer([content])

    grams_sent = []
    for gram in grams:
        if gram not in content:
            print("gram not in ")
            continue
        if gram in stopwords or gram.lower() in stopwords:
           continue
        if gram.lower() in labels_candidates:
            continue
        if gram.isdigit():
            continue
        if re.search('[a-zA-Z]', gram) is None:
            continue
        grams_sent.append((gram, content.replace(gram, '').strip()))

    if len(grams_sent)<=2:
        continue

    results = nli_nlp([ii[1] for ii in grams_sent],  list(labels_candidates), multi_label=True, hypothesis_template="This text is about {}.")
    
    assert len(results) == len(grams_sent)
    for i in range(len(results)):
        gram = grams_sent[i][0]
        result_ = results[i]
        result_.pop('sequence')
        embed_gram = enc.infer([gram])
        simi = cosine_similarity(embed_gram, embeds)
        #print(gram, simi[0][0])

        df_ = pd.DataFrame(result_)
        df_merge = pd.merge(df_ori, df_, on=['labels'], how='inner')
        df_merge['score_diff'] = df_merge['scores_x'] - df_merge['scores_y']
        df_merge.sort_values(by=['score_diff'], ascending=False, inplace=True)

        df_merge_pos = df_merge.loc[df_merge['score_diff']>0]
        if df_merge_pos.shape[0] == 0:
            continue  

        for iix, roww in df_merge_pos.iterrows():
            if not gram_diff[roww['labels']].get(gram, None):
                gram_diff[roww['labels']][gram] = []
            gram_diff[roww['labels']][gram].append(roww['score_diff'] * simi[0][0])

    if ix % 100 ==0:
        print(ix)
        label_expands = {}
        for l, gram_scores in gram_diff.items():
            gram_scores_mean = {g:round(np.array(scores).mean(),4) for g, scores in gram_scores.items() }
            gram_scores_mean_sort = sorted(gram_scores_mean.items(), key=operator.itemgetter(1), reverse=True) 
            print(l, '===>', gram_scores_mean_sort[:50])
            label_expands[l] = [ii[0] for ii in gram_scores_mean_sort[:20]]
        print('\n')
        joblib.dump(gram_diff, 'gram_diff_constrain')

'''


########## get distribution for each gram 
import scipy
df_label = pd.read_csv("df_cc_label.csv")
df_label = df_label.loc[~df_label['title'].isnull()]
df_label_sample = sample_stratify(df_label, df_label.label.value_counts().values.min())

df_label_sample['title_lower'] = df_label_sample['title'].map(lambda x: x.lower())



def cal_gram_entropy(labels_candidates, df_label_sample, gram):
    dftitle = df_label_sample.loc[df_label_sample['title_lower'].str.contains(gram)]

    titles_include = dftitle.sample(min(512, dftitle.shape[0]))['title'].tolist()
    fbs = 256
    infos = []
    for i in range(0,len(titles_include), fbs):
        #print(i)
        if len(titles_include[i:i+fbs]) == 1:
            continue
        nli_result = nli_nlp(titles_include[i:i+fbs], labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
        if isinstance(nli_result, dict):
            continue
        for r in  nli_result:
            lsd = {l:s for l, s in zip(r['labels'], r['scores'])}
            scores = [lsd[l] for l in labels_candidates]
            infos.append(scores)
    if not infos:
        return None, None
    scores_reduce = list(np.array(infos).mean(axis=0))
    df_ls = pd.DataFrame(zip(labels_candidates, scores_reduce), columns=['label','score'])
    df_ls.sort_values(by=['score'], ascending=False, inplace=True)
    #print(gram, len(titles_include), df_ls['score'].std(), scipy.stats.entropy(df_ls['score'].values))
    return df_ls['score'].std(), scipy.stats.entropy(df_ls['score'].values)

'''
grams_entropy = []
for gram in grams_candidates:
    std, entropy = cal_gram_entropy(labels_candidates, df_label_sample, gram)
    grams_entropy.append((gram, len(titles_include), df_ls['score'].std(), scipy.stats.entropy(df_ls['score'].values)))  

df_grams_entropy = pd.DataFrame(grams_entropy, columns=['gram','cnt','std','entropy'])
df_grams_entropy.sort_values(by=['entropy'], ascending=True, inplace=True)

ban_grams = set(df_grams_entropy.loc[df_grams_entropy['std'] < std_cut]['gram'].tolist())
'''
#df_grams_entropy.loc[df_grams_entropy['gram']=='new']
################# filter ######


import joblib,operator
gram_diff = joblib.load(args.gram_diff_file)


label_expands = {}
for l, gram_scores in gram_diff.items():
    gram_scores_mean = {g:round(np.array(scores).sum(),4) for g, scores in gram_scores.items() }
    gram_scores_mean_sort = sorted(gram_scores_mean.items(), key=operator.itemgetter(1), reverse=True) 
    grams_rank = [gs[0] for gs in gram_scores_mean_sort]
    grams_topk = []
    for gram in grams_rank:
        std, entropy = cal_gram_entropy(labels_candidates, df_label_sample, gram)
        if not std:
            continue
        if std < args.std_cut:
            continue
        grams_topk.append(gram)
        if len(grams_topk) == args.topk:
            break 
    label_expands[l] = grams_topk
#print( 'label_expands ===>', label_expands)

grams_candidates = []
for l, grams in label_expands.items():
    grams_candidates.extend(grams)

grams_candidates = list(set(grams_candidates))



######## evaluate ###########

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

    if ix % 2048 == 0:
        print(ix, sum(accs_noexpand) / len(accs_noexpand), sum(accs_expand)/len(accs_expand))

print("final_summary==>", args.gram_diff_file, args.std_cut, args.topk,
     sum(accs_noexpand) / len(accs_noexpand), sum(accs_expand)/len(accs_expand) )

# uci
#4095 noexpand:0.7263 manual_expand:0.76147

'''
ag:
facebook/bart-large-mnli
acc: 0.7199046483909416
time cost: 2.5168240070343018

vicgalle/xlm-roberta-large-xnli-anli
acc: 0.7612236789829162
time cost: 2.5296759605407715

joeddav/xlm-roberta-large-xnli
acc: 0.6893126738180373
time cost: 2.215322971343994



uci:

facebook/bart-large-mnli
acc: 0.6745514798308765
time cost: 0.43288516998291016

vicgalle/xlm-roberta-large-xnli-anli
acc: 0.7370586218717861
time cost: 0.6152443885803223

joeddav/xlm-roberta-large-xnli
acc: 0.6927208319049252
time cost: 0.3963637351989746
'''
















