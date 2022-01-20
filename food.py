import pandas as pd 

#df_interact_train = pd.read_csv("./food/interactions_train.csv")

# df_pp_recipes = pd.read_csv("PP_recipes.csv")
# df_pp_users = pd.read_csv("PP_users.csv")


df_raw_interact = pd.read_csv("./food/RAW_interactions.csv")
df_raw_recipes = pd.read_csv("./food/RAW_recipes.csv")


for ix, row in df_pp_users.sample(10).iterrows():
    for col in df_pp_users.columns:
        print(col, '===>', row[col])
    print()



from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained('roberta-large', cache_dir="./cache",local_files_only=True) 


df_raw_recipes['ingredient_content'] = df_raw_recipes['ingredients']\
                    .map(lambda x: tokenizer.sep_token.join(eval(x)) + tokenizer.eos_token )

df_raw_recipes['recipe'] = df_raw_recipes['steps']\
                    .map(lambda x: ', '.join(eval(x)) + '.' )

from sklearn.model_selection import train_test_split

indgredients_train, indgredients_test =  train_test_split(df_raw_recipes['ingredient_content'].unique(), test_size=0.1)
recipe_train, recipe_test =  train_test_split(df_raw_recipes['recipe'].unique(), test_size=0.1)



with open("./food/indgredients_train.txt",'w') as f:
    for line in indgredients_train:
        f.write(line + '\n')

with open("./food/indgredients_test.txt",'w') as f:
    for line in indgredients_test:
        f.write(line + '\n')



with open("./food/recipe_train.txt",'w') as f:
    for line in recipe_train:
        f.write(line + '\n')

with open("./food/recipe_test.txt",'w') as f:
    for line in recipe_test:
        f.write(line + '\n')


# from datasets import load_dataset

# data_files = {}
# data_files["train"] = './food/indgredients_train.txt'
# data_files["validation"] = './food/indgredients_test.txt'

# raw_datasets = load_dataset("text", data_files=data_files)



