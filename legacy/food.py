import pandas as pd 
import os  
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#df_interact_train = pd.read_csv("./food/interactions_train.csv")

# df_pp_recipes = pd.read_csv("PP_recipes.csv")
# df_pp_users = pd.read_csv("PP_users.csv")


df_raw_interact = pd.read_csv("./food/RAW_interactions.csv")
df_raw_recipes = pd.read_csv("./food/RAW_recipes.csv")



df_raw_interact.user_id.value_counts().reset_index()



recips_cnts = []
for uid in df_raw_interact.user_id.unique():
    dfu = df_raw_interact.loc[df_raw_interact['user_id']==uid]
    recips_cnts.append(dfu.recipe_id.unique().shape[0]) 
    if dfu.shape==10:
        break



pd.merge(dfu, df_raw_recipes, left_on='recipe_id', right_on='id', how='inner')








from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir="./cache",local_files_only=True) 

from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)

from transformers import T5Tokenizer, AutoModelWithLMHead
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
print(tokenizer_t5)






df_raw_recipes['ingredient_content'] = df_raw_recipes['ingredients']\
                    .map(lambda x: '; '.join(eval(x)) )

df_raw_recipes['recipe'] = df_raw_recipes['steps']\
                    .map(lambda x: ', '.join(eval(x)) + '.' ) + tokenizer_gpt2.eos_token

df_raw_recipes['ingre_recipe'] = df_raw_recipes['ingredient_content'] + " {} ".format(tokenizer_gpt2.eos_token)\
                                 + df_raw_recipes['recipe'] + tokenizer_gpt2.eos_token

from sklearn.model_selection import train_test_split

i_train, i_test =  train_test_split(df_raw_recipes['ingredient_content'].unique(), test_size=0.1)
c_train, c_test =  train_test_split(df_raw_recipes['recipe'].unique(), test_size=0.1)
ic_train, ic_test =  train_test_split(df_raw_recipes['ingre_recipe'].unique(), test_size=0.1)


ic_csv_train, ic_csv_test = train_test_split(df_raw_recipes[['ingredient_content', 'recipe']], test_size=0.1)

ic_csv_train.to_csv("./food/ic_csv_train.csv", index=False)
ic_csv_test.to_csv("./food/ic_csv_test.csv", index=False)



# MLM
with open("./food/indgredients_train.txt",'w') as f:
    for line in i_train:
        f.write(line + '\n')

with open("./food/indgredients_test.txt",'w') as f:
    for line in i_test:
        f.write(line + '\n')



# CLM
with open("./food/recipe_train.txt",'w') as f:
    for line in c_train:
        f.write(line + '\n')

with open("./food/recipe_test.txt",'w') as f:
    for line in c_test:
        f.write(line + '\n')


# CLM
with open("./food/ingre_recipe_train.txt",'w') as f:
    for line in ic_train:
        f.write(line + '\n')

with open("./food/ingre_recipe_test.txt",'w') as f:
    for line in ic_test:
        f.write(line + '\n')





from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2', cache_dir="./cache", local_files_only=True)
#tokenizer_gpt2.padding_side = "left" 
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token # to avoid an error "<|endoftext|>": 50256
tokenizer_gpt2.sep_token = '<|sep|>'
#tokenizer_gpt2.add_tokens(tokenizer_gpt2.sep_token)
print(tokenizer_gpt2)

gen_nlp_gpt2 = {}

gpt2_recipe = GPT2LMHeadModel.from_pretrained('./food/gpt_recipe')
gen_nlp_gpt2['recipe']  = pipeline("text-generation", model=gpt2_recipe, tokenizer=tokenizer_gpt2, device=0, return_full_text=False)

gpt2_recipe = GPT2LMHeadModel.from_pretrained('./food/gpt_ingre_recipe')
gen_nlp_gpt2['ingre_recipe']  = pipeline("text-generation", model=gpt2_recipe, tokenizer=tokenizer_gpt2, device=0, return_full_text=False)


from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained('./food/bert_indgredients') 
model = AutoModelWithLMHead.from_pretrained('./food/bert_indgredients')

nlp_fill = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0, top_k = 1000 ) 







row = df_raw_recipes.sample(1)
ingres = eval(row['ingredients'].tolist()[0])
ingre_content = '; '.join(ingres)
print(ingre_content)

row['recipe'].tolist()[0]


prompt = ingre_content.replace('fresh mushrooms;', tokenizer.mask_token)
print(prompt)

fill_result = nlp_fill(prompt)

df_fill_result = pd.DataFrame(fill_result)
del df_fill_result['sequence']

df_fill_result.head(50)




prompt = "olive oil; sweet italian turkey sausage; sweet onion; garlic cloves; green peppers; paprika; red pepper flakes; fresh kale; black-eyed peas; large shrimp; roasted sweet peppers; oregano <|endoftext|>"



gen_nlp_gpt2['ingre_recipe'](prompt+tokenizer_gpt2.eos_token, max_length=256, \
                                                    do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                                    repetition_penalty=1.2, num_return_sequences= 4,\
                                                    clean_up_tokenization_spaces=True)
import datasets
rouge = datasets.load_metric('rouge')


predictions = ["hello there", "general kenobi slam"]
references = ["hello there", "general kenobi"]


preds_recipe_gpt = []
preds_ingre_recipe_gpt = []
references = []
with open('./food/ingre_recipe_test.txt', 'r') as f:
    for line in f:
        ingre = line.strip().split(tokenizer_gpt2.eos_token)[0]
        recipe = line.strip().split(tokenizer_gpt2.eos_token)[1]
        result_recipe = gen_nlp_gpt2['recipe'](ingre+tokenizer_gpt2.eos_token, max_length=256, \
                                                    do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                                    repetition_penalty=1.2, num_return_sequences= 1,\
                                                    clean_up_tokenization_spaces=True)

        result_ingre_recipe = gen_nlp_gpt2['ingre_recipe'](ingre+tokenizer_gpt2.eos_token, max_length=256, \
                                                    do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                                    repetition_penalty=1.2, num_return_sequences= 1,\
                                                    clean_up_tokenization_spaces=True)
        preds_recipe_gpt.append(result_recipe[0]['generated_text'])
        preds_ingre_recipe_gpt.append(result_ingre_recipe[0]['generated_text'])
        references.append(recipe)

results = rouge.compute(predictions=preds_recipe_gpt, references=references)
for k in results.keys():
    print(k, results[k].mid.fmeasure)


results = rouge.compute(predictions=preds_ingre_recipe_gpt, references=references)
for k in results.keys():
    print(k, results[k].mid.fmeasure)







import transformers
from transformers import T5Tokenizer, AutoModelWithLMHead
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base", cache_dir="./cache", local_files_only=True)
print(tokenizer_t5)

t5 = AutoModelWithLMHead.from_pretrained("./food/t5_ingre_recipe/")
gen_nlp_t5  = transformers.pipeline("text2text-generation", model=t5, tokenizer=tokenizer_t5, device=0)




df_t5_test = pd.read_csv("./food/ic_csv_test.csv")

preds_recipe_t5 = []
references = []

for ix, row in df_t5_test.sample(frac=1).reset_index().iterrows():
    ingre = row['ingredient_content']
    recipe = row['recipe']
    result_recipe = gen_nlp_t5(ingre, max_length=256, \
                                                    do_sample=True, top_p=0.9, top_k=0, temperature=1.2,\
                                                    repetition_penalty=1.2, num_return_sequences= 1,\
                                                    clean_up_tokenization_spaces=True)
    pre_recipe = result_recipe[0]['generated_text'] 
    references.append(recipe)
    preds_recipe_t5.append(pre_recipe)

    if ix % 100 == 0:
        print(ix)
        results = rouge.compute(predictions=preds_recipe_t5, references=references)
        for k in results.keys():
            print(k, results[k].mid.fmeasure)


df_raw_recipes['recipe'] = df_raw_recipes['steps'].map(lambda x: ' '.join(df_raw_recipes))


for steps in df_raw_recipes.steps.tolist():
    recipe = ','.join(eval(steps))
    if  'preheat the oven to 400 degrees' in recipe:
        print (recipe)



dividing it evenl











