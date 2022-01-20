import pandas as pd 
df = pd.read_csv("RAW_recipes.csv")

df_interact = pd.read_csv("interactions_train.csv")

Index(['name', 'id', 'minutes', 'contributor_id', 'submitted', 'tags',
       'nutrition', 'n_steps', 'steps', 'description', 'ingredients',
       'n_ingredients'],
      dtype='object')

df_pp_recipes = pd.read_csv("PP_recipes.csv")
df_pp_users = pd.read_csv("PP_users.csv")


df_raw_interact = pd.read_csv("RAW_interactions.csv")
df_raw_recipes = pd.read_csv("RAW_recipes.csv")


for ix, row in df_pp_users.sample(10).iterrows():
    for col in df_pp_users.columns:
        print(col, '===>', row[col])
    print()



df_raw_interact.loc[(df_raw_interact['user_id']==252205) & (df_raw_interact['recipe_id']==81398)]

