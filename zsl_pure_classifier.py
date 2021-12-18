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
parser.add_argument("--dsn", default="uci", type=str)
parser.add_argument("--fbs", default=32, type=int)
parser.add_argument("--topk", default=100, type=int)
parser.add_argument("--manauto", default="auto", type=str)
parser.add_argument("--gpu", default="4", type=str)
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
#from sklearn.metrics.pairwise import cosine_distances,cosine_similarity 
# from utils.encoders import *
# enc = encoder('cmlm-base')


nli_model_name = "facebook/bart-large-mnli"

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_nli = AutoModelForSequenceClassification.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
tokenizer_nli = AutoTokenizer.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='./cache', local_files_only=True)
nli_nlp = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=0)


from transformers import T5Tokenizer, AutoModelWithLMHead
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
print(tokenizer_t5)
t5 = AutoModelWithLMHead.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)    
gen_nlp  = pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=0)


# load dataset
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
# nltk.download('stopwords')
# stopwords = stopwords.words('english')
stopwords = joblib.load("./utils/stopwords")

import numpy as np
import datasets,re,operator,joblib
# try:
#     cc_news = datasets.load_dataset('cc_news', split="train", cache_dir='./torch_ds')
#     df = pd.DataFrame(zip(cc_news['title'], cc_news['text'], cc_news['description'] ))
#     df.columns = ['title','content','description']
#     df.drop_duplicates(['title','content'], inplace=True) 
#     df.to_csv("./torch_ds/df_cc_news.csv", index=False)
# except:
df = get_cc_news(1)
df = df.loc[~df['title'].isnull()]
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
def check_noun(word):
    nets = wn.synsets(word)
    for net in nets:
        if net.name().split('.')[1] == 'n':
            return True 
    return False

############ find support seeds
from nltk.corpus import wordnet as wn
gram_diff = {l:{} for l in labels_candidates}


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
    if df_ori['scores'].max() < 0.7:
        continue

    gen_result = gen_nlp(content + tokenizer_t5.eos_token,  do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                            repetition_penalty=1.2, num_return_sequences= args.fbs,\
                                            clean_up_tokenization_spaces=True)
    gen_contents = [s['generated_text'].lower() for s in gen_result if s['generated_text']]

    # embeds = enc.infer([content])
    # embeds_grams = enc.infer(grams)
    # simis = cosine_similarity(embeds, embeds_grams)[0]
    # df_gram_simis = pd.DataFrame(zip(grams, list(simis)), columns=['gram','simi'])
    # #df_gram_simis['simi'] = (df_gram_simis['simi'] - df_gram_simis['simi'].min()) / (df_gram_simis['simi'].max()-df_gram_simis['simi'].min())
    # df_gram_simis.sort_values(by=['simi'], ascending=False, inplace=True)
    print(ix)
    print("======>", content)
    print("====>", grams)
    
    for gram in grams:
        lscores = {l:[] for l in labels_candidates}
        contents_yes = []
        contents_no = []
        for sent in gen_contents:
            if gram not in sent.split(' '):
                continue
            contents_yes.append(sent)
            contents_no.append(sent.replace(gram, ''))

        contents_yes.append(content)
        contents_no.append(content.replace(gram, '').strip())

        result_yes = nli_nlp(contents_yes,  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
        result_no = nli_nlp(contents_no,  labels_candidates, multi_label=True, hypothesis_template="This text is about {}.")
        
        if len(contents_yes) == 1:
            result_yes = [result_yes]
            result_no = [result_no]
        assert len(result_yes) == len(result_no)
        for ryes, rno in zip(result_yes, result_no):
            ryes.pop('sequence')
            rno.pop('sequence')
            df_yes = pd.DataFrame(ryes)
            df_no = pd.DataFrame(rno)
            df_merge_yesno = pd.merge(df_yes, df_no, on=['labels'], how='inner')
            df_merge_yesno['score_diff'] = df_merge_yesno['scores_x'] - df_merge_yesno['scores_y']
            for _, row in df_merge_yesno.iterrows():
                if row['score_diff'] > 0 :
                    lscores[row['labels']].append(row['score_diff'])

        lscores_final = {l:sum(s)/len(s) for l, s in lscores.items() if len(s) > 0}
        #print(gram, lscores_final)


        for l, s in lscores_final.items():
            if not gram_diff[l].get(gram, None):
                gram_diff[l][gram] = []
            gram_diff[l][gram].append(s)


    if ix % 100 ==0 and ix > 0 :
        print(ix)
        label_expands = {}
        for l, gram_scores in gram_diff.items():
            gram_scores_mean = {g:round(np.array(scores).mean(),4) for g, scores in gram_scores.items() }
            gram_scores_mean_sort = sorted(gram_scores_mean.items(), key=operator.itemgetter(1), reverse=True) 
            print(l, '===>', gram_scores_mean_sort[:50])
            label_expands[l] = [ii[0] for ii in gram_scores_mean_sort[:20]]
        print('\n')
        joblib.dump(gram_diff, 'gram_diff_gen__{}_{}'.format(args.dsn, args.fbs))













'''

########## get distribution for each gram 
import scipy
df_label = pd.read_csv("df_cc_label.csv")
df_label = df_label.loc[~df_label['title'].isnull()]
df_label_sample = sample_stratify(df_label, df_label.label.value_counts().values.min())

df_label_sample['title_lower'] = df_label_sample['title'].map(lambda x: x.lower())

df_label_sample.loc[df_label_sample['title_lower'].str.contains('vaccine')]


def cal_gram_entropy(labels_candidates, df_label_sample, gram):
    dftitle = df_label_sample.loc[df_label_sample['title_lower'].str.contains(gram)]

    titles_include = dftitle.sample(min(1024, dftitle.shape[0]))['title'].tolist()
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


import joblib,operator
gram_diff = joblib.load(args.gram_diff_file)

# expansion automatic
label_expands_auto = {}
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
    label_expands_auto[l] = grams_topk

# expansion mannual 
label_expands_mannual = map_expand_nli(base_nli, args.dsn)


# assign
if args.manauto == 'man':
    label_expands = label_expands_mannual
elif args.manauto == 'auto':
    label_expands = label_expands_auto


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

    if ix % 256 == 0 and ix > 0:
        print(ix, sum(accs_noexpand) / len(accs_noexpand), sum(accs_expand)/len(accs_expand))

print("final_summary==>", args.dsn, args.gram_diff_file, args.std_cut, args.topk,
     sum(accs_noexpand) / len(accs_noexpand), sum(accs_expand)/len(accs_expand) )



'''














