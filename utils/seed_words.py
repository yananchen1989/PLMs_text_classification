import joblib
stopwords = joblib.load("./utils/stopwords")
stopwords = set(stopwords)  

from transformers.file_utils import cached_path
# gram_diff = joblib.load("gram_diff___{}".format('ag'))
# model_w2v = gensim.models.KeyedVectors.load_word2vec_format('./resource/GoogleNews-vectors-negative300.bin',binary=True)



label_expands_self ={
    'politics':['Politics','War', 'Election','Constitution','Democracy','Conflict','Military', 'Peace', 'Iran', \
                'Syria', 'Israel', 'Crisis', 'EU','NATO', 'korea', 'Afghanistan', 'Taliban',
                'Terrorism', 'Government', 'Ideology', 'fascism', 'Socialism', 'Totalitarian', 'Religion'],

    'law':      ['Law', 'Legitimacy','Court','Crime','Murder','Jurisdiction'],

    'science':  ['Science','Aerospace','Physics','Chemistry','Biology','Scientist','Astronomy','Universe','Big Bang'],

    'technology':['Technology','Biotech', 'IT','Computers','Internet','Algorithm','Space','Bitcoin',\
                    'artificial Intelligence','Robot'],

    'health': ['Health','Healthcare','Medicine','Clinics','Vaccine','Wellness','Nutrition','Dental','HIV','Disease'],

    'business': ['Business','Finance','Oil price','Supply','Inflation','Dollars','Bank','Wall Street','Bitcoin',
                        'Federal Reserve','Accrual','Accountancy','Consumerism','Trade','Quarterly earnings',\
                         'Deposit','Revenue','Stocks','Recapitalization','Marketing','Futures', 
                          'competition', 'billionaire', 'retailer', 'entrepreneurs', 'merger', \
                    'acquisition', 'manufacturers', 'businesses', 'economist', 'sponsor', 'broker', 'growth',\
                     'corporation', 'sells', 'profit', 'trader', 'brands', 'clients', \
                     'pharma', 'bank', 'investor', 'market', 'client', 'shareholder',\
                      'earnings', 'businesswoman', 'marketing', 'entrepreneur', 'directors', 'dollar',\
                       'professional', 'manager', 'finance', 'estate', 'warehouse', 'companies', 'distributor', \
                       'chain', 'sales', 'supplier', 'commerce',  'retail', \
                        'brewers', 'production', 'traders', 'operations', 'executive',\
                        'productivity', 'contractors', 'corp', 'stocks', 'managers', 'employment', 'division',\
                         'recruitment', 'mining', 'profits', 'employees', 'manufacturer', 'partnership', \
                         'contractor', 'boss', 'shareholders', 'team', 'enterprise', 'investors', 'workers', 'marketers',\
                          'economy', 'ceo', 'shares', 'services', 'franchise', 'businessman',\
                    'exporters', 'partners', 'conference', 'commercial', 'markets', 'tycoon', \
                    'partner', 'retailers','executives', 'insurer', 'dealer', \
                    'workplaces', 'industries', 'technology', 'trading', 'customers', 'contract', 'analysts',\
                      'dividend', 'bucks', 'stock', 'supermarket', 'capital', 'customer', 'banks', 'equities', \
                      'economics', 'revenue', 'investment', 'angel funds', 'Venture', 'Capital', 
                     'VC'],


    'sports': ['Sports','Athletics','Championships','Football','Olympic','Tournament','Chelsea','League','Golf',
                            'NFL','Super bowl','World Cup', 'formula',\
                            'marathon', 'curling', 'athlete', 'riding', 'swimmers', 'tennis', 'players', 'manchester',
                    'cyclists', 'scores', 'olympian', 'nike', 'footballers', 'liverpool', 'soccer',
                    'climber', 'receiver', 'hoops', 'pitching', 'volleyball', 
                    'lineman', 'tournament', 'champions', 'rodeo', 'pitcher', 'stunt', 'stars',
                    'gym', 'weight', 'softball', 'quarterback', 'colts', 
                     'cyclist', 'bike', 'surfer', 'marlins', 'championship', 'hockey', 
                     'sportsman', 'matches', 'darts', 'broncos', 'bets',
                      'workout', 'titans', 'skiers', 'playoff', 'footballer', 'teams', 'redskins', 
                     'barcelona', 'swimming', 'skate', 'speedway', 'bikes', 'golfers', 
                     'champion', 'cowboys', 'raceway', 'coaching', 'pitchers', 'swimsuit', 
                'atlanta', 'surfing', 'cycling', 'baseball', 'season', 'racing', 'gymnastics',
                 'lineup', 'bowling', 'finals', 'wrestling', 'trainer', 'playoffs', 'boxing', 
                 'teammates', 'caps', 'fans', 'wrestler', 'chess', 'fitness', 'lacrosse',
                  'dancers', 'trophy', 'athletes', 'wrestlers', 'batting', 'club',  'training',
                  'yankees', 'basketball', 'jones', 'olympics', 'cricketer',
                   'team',  'league', 'cricketers', 'golfer', 'exercise', 'derby', 'workouts',
                    'match', 'jersey', 'riders', 'coaches', 'swim', 'bowl', 'skating', 'coach', 
                    'linebacker', 'stadium', 'diving', 'boxer', 'captain', 'skateboard', 'golf',
                     'athletics', 'runners', 'rugby', 'wimbledon', 'quarterbacks', 'cricket'],

    'science and technology': ['equipment', 'smart', 'rocket', 'weld', 'tools', 'submarine', 'materials', 'lego', 'battery', \
                    'uranium', 'android', 'transplant', 'experiment', 'researchers', 'nasa', 'nobel prize',\
                     'internet', 'chip', 'newton', 'innovation', 'engineering', 'telecom', 'silicon', \
                     'moon', 'software', 'speed', 'giants', 'puppet', 'development', 'phones', 'arsenal', \
                      'cameras', 'streaming', 'rockets', 'developers', 'robots', 'manufacturing', 'zoo', 'pharma', \
                      'revolution', 'mobile', 'balloon', 'brain', 'hacker', 'gps', 'toys', 'weapons', 'titans',\
                       'documentary', 'synthetic', 'cybercrime', 'electric', 'coal', 'marvel', 'cellphone', 'energy',\
                        'factories', 'electronics', 'builds', 'laboratories', 'google', 'scientists', 'hackers',\
                         'laptop', 'space', 'trump', 'robotics', 'pilot', 'diesel', 'researcher', 'pipeline',\
                          'rays', 'nuke', 'chemistry', 'semiconductor', 'machines', 'camera', 'cloud', 'penguins',\
                           'chargers', 'mining', 'engineers', 'chemical', 'gaming', 'technical', 'olympics', 'nukes',\
                            'hack', 'drilling', 'aircraft', 'drones', 'model', 'galaxy', 'science', 'machine', \
                            'technologies', 'tesla', 'weapon', 'hacks', 'tool', 'launches', 'electricity',\
                     'screening', 'startup', 'laser', 'engineer', 'patent', 'future', 'scientist',\
                      'discovery', 'wireless', 'technology', 'satellite',\
                       'biotech', 'probe', 'magic', 'robot', 'training', 'apple', 'automation', 'artificial Intelligence', 'Musk',
                    'mars', 'solar system', 'arxiv', 'quantum', 'computing', 'nuclear']
}



     

def get_s3_words():
    BAG_OF_WORDS_ARCHIVE_MAP = {
        'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
        'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
        'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
        'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
        'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
        'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
        'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
        'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
        'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
        }
    label_expands = {}
    for l in BAG_OF_WORDS_ARCHIVE_MAP.keys():
        filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[l])
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        #print(words)
        label_expands[l] = words
    return label_expands

label_expands_s3 =  get_s3_words()


label_expand_ag = {
    'World': 
        label_expands_s3['religion'] + label_expands_s3['politics'] + label_expands_s3['military'] \
            + label_expands_s3['legal'] + label_expands_self['law'] + label_expands_self['politics'],
    'Sports': 
        label_expands_self['sports'],
    'Business':
        label_expands_self['business'],
    'science and technology':
        label_expands_s3['technology'] + label_expands_s3['space'] + label_expands_s3['science'] \
        + label_expands_self['science and technology'] + label_expands_self['science'] + label_expands_self['technology']
}


for l in label_expand_ag.keys():
    label_expand_ag[l] = list(set([ii.lower() for ii in label_expand_ag[l]]))
    print(l, len(label_expand_ag[l]))



'''
def get_seed_words(topk):
    vocab_w2v = set(list(model_w2v.index_to_key))
    label_expands_auto = {}
    for l, gram_scores in gram_diff.items():
        gram_scores_sum = {g:round(np.array(scores).sum(),4) for g, scores in gram_scores.items() }
        gram_scores_sum_sort = sorted(gram_scores_sum.items(), key=operator.itemgetter(1), reverse=True) 
        gram_scores_mean = {g:round(np.array(scores).mean(),4) for g, scores in gram_scores.items() }
        gram_scores_mean_sort = sorted(gram_scores_mean.items(), key=operator.itemgetter(1), reverse=True) 
        gram_scores_sort = gram_scores_sum_sort + gram_scores_mean_sort
        label_expands_auto[l] = set()
        for j in gram_scores_sort:
            if j[0] not in vocab_w2v or j[0] in ['news']:
                #print(j[0])
                continue
            if ' and ' in l:
                w0 = l.split('and')[0].strip().lower()
                w1 = l.split('and')[1].strip().lower()
                simi = max(model_w2v.similarity(w0, j[0]), model_w2v.similarity(w1, j[0]) )
            else:
                simi = model_w2v.similarity(l.lower(), j[0])
            if simi >= 0.1:
                label_expands_auto[l].add(j[0])
            if len(label_expands_auto[l]) == topk:
                break 
        if ' and ' in l:
            label_expands_auto[l].add(l.split('and')[0].strip())
            label_expands_auto[l].add(l.split('and')[1].strip())
        else:
            label_expands_auto[l].add(l)
        # for ll in BAG_OF_WORDS_ARCHIVE_MAP:
        #     if (ll in l.lower()) or ('world' in l.lower() and ll == 'politics') or \
        #         (('science' in l.lower() or 'technology' in l.lower()) and ll == 'space'):
        #         words_s3 = get_s3_words(ll)
        #         label_expands_auto[l].update(words_s3)
        # for ll, expands in BASE_NLI.items():
        #     if ll in l.lower():
        #         label_expands_auto[l].update( [w.lower() for w in expands ] )
        print(l, label_expands_auto[l], '\n')
    return label_expands_auto
'''

# label_expands_auto = get_seed_words(128)
# print(label_expands_auto)
