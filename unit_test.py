
sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

content = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''


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






generated_text = "Editorial cartoon, Sun Jun 6: Asian Innovation, how unfriendly of you as $ 40 makes China true hank hacking 1] recently    You talk about personal electronic interaction with someone by surflinking their history into a post-doc and you walk into the office, and as an e-mailer you are literally texting me, making tons of calls, walking into my place in two hours and"


import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
from sklearn import metrics


files = glob.glob("./df_train_noise_ag_814_*_0.9*.csv")

df_retain_ll = []
for file in files:
    df = pd.read_csv(file)
    df.sort_values(by=['dve_out'], ascending=False, inplace=True) 

    for ix in range(df.shape[0]):
        df_block = df[0:ix]
        if df_block.shape[0] <= 10:
            continue
        ratio = df_block.loc[df_block['groudtruth']==0].shape[0] / df.loc[df['groudtruth']==0].shape[0]
        if ratio >= 0.05:
            print(ix, ratio)
            break 
    df_cut_tmp = df[:ix]
    print('1 proportion:', df_cut_tmp.loc[df_cut_tmp['groudtruth']==1].shape[0] / df.loc[df['groudtruth']==1].shape[0], '\n')
    del df_cut_tmp['dve_out']
    for col in df_cut_tmp.columns:
        if col.startswith('embed_'):
            del df_cut_tmp[col]
    df_retain_ll.append(df_cut_tmp.loc[df_cut_tmp['groudtruth']==9])


df_merge = df_retain_ll[0].copy()
for df_cut_tmp in df_retain_ll:
    print('before:', df_merge.shape[0]) 
    df_merge = pd.merge(df_merge, df_cut_tmp, on=['content','label','label_name','groudtruth'], how='inner')
    print('after:', df_merge.shape[0]) 






df_train_noise = df.loc[df['groudtruth'].isin([0,1])]

df.sort_values(by=['dve_out'], ascending=False, inplace=True) 

df_train_noise[['groudtruth','dve_out','label_name']]


precision, recall, thresholds = metrics.precision_recall_curve(df_train_noise['groudtruth'].values,  df_train_noise['dve_out'].values, pos_label=1)



df_train_noise.loc[(df_train_noise['dve_out']>=0.18453786) & (df_train_noise['groudtruth']==1)].shape[0] / df_train_noise.loc[df_train_noise['dve_out']>=0.18453786].shape[0]



for file in files:
    df = pd.read_csv(file)
    df_syn = df.loc[df['groudtruth'].isin([9])]
    df_syn.sort_values(by=['dve_out'], ascending=False, inplace=True) 
    print(df_syn['dve_out'].mean(), df_syn['dve_out'].max(), df_syn['dve_out'].min())


row = df_merge.sample(1)
print(row['content'].tolist()[0])
print(row['label_name'])



from openprompt.data_utils import InputExample
classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]
dataset = [ # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid = 0,
        text_a = "Albert Einstein was one of the greatest intellects of his time.",
    ),
    InputExample(
        guid = 1,
        text_a = "The film was badly made.",
    ),
]

from openprompt.plms import get_model_class
model_class = get_model_class(plm_type = "bert")
model_path = "bert-base-cased"
bertConfig = model_class.config.from_pretrained(model_path)
bertTokenizer = model_class.tokenizer.from_pretrained(model_path)
bertModel = model_class.model.from_pretrained(model_path)

from openprompt.prompts import ManualTemplate
promptTemplate = ManualTemplate(
    text = ["<text_a>", "It", "was", "<mask>"],
    tokenizer = bertTokenizer,
)

from openprompt.prompts import ManualVerbalizer
promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "negative": ["bad"],
        "positive": ["good", "wonderful", "great"],
    },
    tokenizer = bertTokenizer,
)

from openprompt import PromptForClassification
promptModel = PromptForClassification(
    template = promptTemplate,
    model = bertModel,
    verbalizer = promptVerbalizer,
)

from openprompt import PromptDataLoader
data_loader = PromptDataLoader(
    dataset = dataset,
    tokenizer = bertTokenizer,
    template = promptTemplate,
)


promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim = -1)
        print(classes[preds])
# predictions would be 1, 0 for classes 'positive', 'negative'










import requests
requests.get('https://huggingface.co/bert-base-uncased/resolve/main/config.json')
