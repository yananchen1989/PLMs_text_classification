sent = "Edelman Partners. New York NY J.D. Shaw gets $18 million at JPMorgan Chase & Co., to cash in on the long run; withdraws $20 million in business and two senior executives earn $4 million to $5 million to avoid penalties Financial Times , Feb 15; Citi Plc Frequent speaker, former U.S. Ambassador"

sent = "The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise."

content = '''
They are fed up with slow speeds, high prices and the level of customer service they receive. 17% of readers have switched suppliers and a further 16% are considering changing in the near future. It is particularly bad news for BT, the UK's biggest internet supplier, with almost three times as many people trying to leave as joining.
'''
sent = "Federal jury orders tech giant Samsung to pay"

sent = 'FDA gives green light to migraine prevention tool'

sent = "Obamacare offers health insurance, not health care"

sent = "Why BlackBerry ( BBRY ) Stock Is Up Today"

sent = "Doctor warns Arizonans about colorectal cancer"


content = "Facebook acquires video ad company LiveRail for between US $ 500m"

sent = "Autism wave keeps growing"

sent = "Virus to cause spike in pork prices"

content = "Grand Budapest Hotel'not grand, but still stylish"


tokenizer("Why BlackBerry ( BBRY ) Stock Is Up Today", return_special_tokens_mask=True)

tokenizer_gpt2("Why BlackBerry ( BBRY ) Stock Is Up Today \n<|endoftext|>", return_special_tokens_mask=True)

tokenizer_t5("Why BlackBerry ( BBRY ) Stock Is Up Today"+tokenizer_t5.eos_token, max_length=100,\
         padding=True, truncation=True, return_special_tokens_mask=True)


tokenizer_t5.convert_ids_to_tokens([2])
tokenizer_gpt2.convert_ids_to_tokens([6288])
tokenizer_t5.convert_tokens_to_ids(['up'])


# gpt neo
# from transformers import GPT2Tokenizer, GPTNeoForCausalLM
# tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir="./cache")
# gpt2 = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir="./cache")





















input_ids = tokenizer_gpt2.encode(sent, return_tensors="tf")
# get logits of last hidden state
next_token_logits = gpt2(input_ids).logits[:, -1, :] / 1.0








################ t5 token generation ###################
# create ids of encoded input vectors

temperature=1.2

sent = 'FDA gives green light to migraine prevention tool'
input_ids = tokenizer_t5(sent, return_tensors="pt").input_ids

# create BOS token
decoder_input_ids = tokenizer_t5(tokenizer_t5.pad_token, add_special_tokens=False, return_tensors="pt").input_ids

assert decoder_input_ids[0, 0].item() == t5.config.decoder_start_token_id, "`decoder_input_ids` should correspond to `model.config.decoder_start_token_id`"

# pass input_ids to encoder and to decoder and pass BOS token to decoder to retrieve first logit
outputs = t5(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)

# get encoded sequence
encoded_sequence = (outputs.encoder_last_hidden_state,)
# get logits
lm_logits = outputs.logits

# sample last token with highest prob
#next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
next_token_logits = top_k_top_p_filtering(lm_logits[:, -1, :]/temperature, top_k=0, top_p=0.9)
probs = F.softmax(next_token_logits, dim=-1)
next_decoder_input_ids = torch.multinomial(probs, num_samples=1)

# concat
decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)


for _ in range(64):
    # reuse encoded_inputs and pass BOS + "Ich" to decoder to second logit
    lm_logits = t5(None, encoder_outputs=encoded_sequence, decoder_input_ids=decoder_input_ids, return_dict=True).logits

    # sample last token with highest prob again
    #next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
    next_token_logits = top_k_top_p_filtering(lm_logits[:, -1, :]/temperature, top_k=0, top_p=0.9)
    probs = F.softmax(next_token_logits, dim=-1)
    next_decoder_input_ids = torch.multinomial(probs, num_samples=1)

    # concat again
    decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)


print(tokenizer_t5.decode(decoder_input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))





#############  计算指标          ##############

import glob
import pandas as pd 
files = glob.glob("./slurm-*.out")
infos = []
for file in files:

    with open(file, 'r') as f:
        for line in f:
            if 'summary____' in line:
                tokens = line.strip().split(' ')
                if len(tokens) != 8:
                    continue
                infos.append(tokens)

import pandas as pd 
df = pd.DataFrame(infos, columns=['summary', 'dsn', 'samplecnt', 'model', 'ite', 'fmark', 'noaug_acc', 'aug_acc'])

df['aug_acc'] = df['aug_acc'].astype('float')
df['noaug_acc'] = df['noaug_acc'].astype('float')
df['samplecnt'] = df['samplecnt'].astype('int')
df['ite'] = df['ite'].astype('int')



for model in [ 'former']:
    for dsn in [ 'ag']: # 'ag', 'uci',
        print(model, dsn)
        for samplecnt in df['samplecnt'].unique():
            noaug_acc = df.loc[(df['samplecnt']==samplecnt) & (df['model']==model) & (df['dsn']==dsn), 'noaug_acc'].mean()
            for fmark in df.loc[(df['dsn']==dsn) & (df['model']==model),'fmark'].unique():
                dfi = df.loc[(df['samplecnt']==samplecnt) & (df['fmark']==fmark) & (df['model']==model) & (df['dsn']==dsn)]
                print(samplecnt, fmark, noaug_acc, dfi['aug_acc'].mean(), dfi.shape[0])
            print()



















import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv("./experiment/ag.result.txt", sep='\t')



for dsn in ['ag', 'yahoo']:
    sns.set()
    g_ag = sns.catplot( x="Testbed", y="Accuracy", hue="Approach", \
                     data=df.loc[df['dsn']==dsn], height=6, kind="bar", palette="muted", legend_out=False)
    g_ag.set_ylabels("Accuracy")
    #g_ag.set_titles('dataset: AG')
    if dsn == 'ag':
        plt.ylim(0.4, 0.95)
    elif dsn == 'yahoo':
        plt.ylim(0.35, 0.75)
    #plt.xticks(np.arange(0.4, 0.95, 0.5))
    # plt.title('Improvement for accuracy on {} dataset'.format(dsn.upper()))
    plt.legend(loc=2, prop={'size': 10})
    plt.subplots_adjust(top=0.9)
    # plt.show()
    #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.6, top=0.8)
    # plt.savefig("{}.png".format(dsn), dpi=1000)
    plt.savefig("/Users/yanan/Downloads/{}_bar_.svg".format(dsn), format='svg', dpi='figure')
    plt.close()
    #plt.show()




df = pd.read_csv("./experiment/vary.samples.result.txt", sep='\t')
df = df.loc[df['# of samples']!=1024]

for exp in ['with_exp', 'with_aug']:
    sns.lineplot(hue='Dataset', data=df.loc[df['exp']==exp], x="# of samples", y="Accuracy", markers=True, style="Dataset", dashes=False)
    plt.ylim(0.5, 1)
    plt.xticks(np.unique(df.loc[df['exp']==exp]['# of samples'].values))
    # plt.title('Accuracy with respect to # of samples in the setting of "{}"'.format(' '.join(exp.split('_'))))
    #plt.show()
    plt.savefig("/Users/yanan/Downloads/{}.svg".format(exp), format='svg', dpi='figure')
    plt.close()




####### DA ablation 折线图

df = pd.read_csv("./experiment/gpt_t5_ablation.tsv", sep='\t')

p = sns.lineplot(hue='Model', data=df, x="K", y="Accuracy", markers=True, style="Model", dashes=False)
plt.ylim(df['Accuracy'].min()-1, df['Accuracy'].max()+1)
plt.xticks(np.unique(df['K'].values))
# plt.title('Accuracy with respect to K ')
p.set_xlabel("K", fontsize = 20)
p.set_ylabel("Accuracy", fontsize = 20)
# plt.legend(labels=["Legend_Day1","Legend_Day2"], fontsize = 20)
plt.show()

plt.savefig("/Users/yanan/Downloads/t5_ablation.svg",  format="svg")
plt.close()



df = pd.read_csv("tqi_exp.tsv", sep='\t')
p = sns.lineplot( data=df, x="K", y="f1", markers=True, dashes=True, palette=['g'])
plt.xticks(np.unique(df['K'].values))
# plt.legend(labels=["Legend_Day1","Legend_Day2"], fontsize = 20)
plt.xlim([0, 16])
plt.show()



############ ctrl ############
import torch
device0 = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

from transformers import CTRLTokenizer, CTRLLMHeadModel, pipeline
tokenizer_ctrl = CTRLTokenizer.from_pretrained('ctrl', cache_dir='./cache')
model_ctrl = CTRLLMHeadModel.from_pretrained('ctrl', cache_dir='./cache')
print(tokenizer_ctrl)
control_codes = tokenizer_ctrl.control_codes.keys()
gen_nlp_ctrl  = pipeline("text-generation", model=model_ctrl, tokenizer=tokenizer_ctrl, device=0, return_full_text=False)


'Sports', 'boxing', 'gymnastics', 'boxer'

prompts = ["Links In Sports boxing gymnastics boxer : "] # ctrl
result_gpt = gen_nlp_ctrl(prompts, max_length=128, \
                                do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                repetition_penalty=1.2, num_return_sequences= 16,\
                                clean_up_tokenization_spaces=True)


from transformers import BartTokenizer, AutoModelWithLMHead
from transformers import pipeline
tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-base", cache_dir="/scratch/w/wluyliu/yananc/cache", local_files_only=True)
print(tokenizer_bart)
bart_noft = AutoModelWithLMHead.from_pretrained("facebook/bart-base", cache_dir="/scratch/w/wluyliu/yananc/cache", local_files_only=True)


gen_nlp = pipeline("text2text-generation", model=bart_noft, tokenizer=tokenizer_bart, device=0)
prompt = "Obamacare offers health insurance, not health care."

result = gen_nlp(prompt, max_length=128, \
                                do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                repetition_penalty=1.2, num_return_sequences= 32,\
                                clean_up_tokenization_spaces=True)



assert len(result_gpt) == 32

contents_syn = [remove_str(ii['generated_text']) for ii in result_gpt if ii['generated_text'] and ii['generated_text']!=prompt]

for ii in contents_syn:
    print(ii, '\n')









