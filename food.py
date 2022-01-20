import pandas as pd 

#df_interact_train = pd.read_csv("./food/interactions_train.csv")

# df_pp_recipes = pd.read_csv("PP_recipes.csv")
# df_pp_users = pd.read_csv("PP_users.csv")


df_raw_interact = pd.read_csv("./food/RAW_interactions.csv")
df_raw_recipes = pd.read_csv("./food/RAW_recipes.csv")



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
                    .map(lambda x: ', '.join(eval(x)) + '.' )

df_raw_recipes['ingre_recipe'] = df_raw_recipes['ingredient_content'] + " {} ".format(tokenizer_gpt2.eos_token)\
                                 + df_raw_recipes['recipe'] 

from sklearn.model_selection import train_test_split

i_train, i_test =  train_test_split(df_raw_recipes['ingredient_content'].unique(), test_size=0.1)
c_train, c_test =  train_test_split(df_raw_recipes['recipe'].unique(), test_size=0.1)
ic_train, ic_test =  train_test_split(df_raw_recipes['ingre_recipe'].unique(), test_size=0.1)


ic_csv_train, ic_csv_test = train_test_split(df_raw_recipes[['ingredient_content', 'recipe']], test_size=0.1)

ic_csv_train.to_csv("ic_csv_train.csv", index=False)
ic_csv_test.to_csv("ic_csv_test.csv", index=False)




with open("./food/indgredients_train.txt",'w') as f:
    for line in i_train:
        f.write(line + '\n')

with open("./food/indgredients_test.txt",'w') as f:
    for line in i_test:
        f.write(line + '\n')


with open("./food/recipe_train.txt",'w') as f:
    for line in c_train:
        f.write(line + '\n')

with open("./food/recipe_test.txt",'w') as f:
    for line in c_test:
        f.write(line + '\n')



with open("./food/ingre_recipe_train.txt",'w') as f:
    for line in ic_train:
        f.write(line + '\n')

with open("./food/ingre_recipe_test.txt",'w') as f:
    for line in ic_test:
        f.write(line + '\n')








# from datasets import load_dataset

# data_files = {}
# data_files["train"] = './food/indgredients_train.txt'
# data_files["validation"] = './food/indgredients_test.txt'

# raw_datasets = load_dataset("text", data_files=data_files)



