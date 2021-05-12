# datasets from huggingface 
import datasets
import pandas as pd


# agnews
ag_news_train = datasets.load_dataset('ag_news', split="train")
df_train = pd.DataFrame(zip(ag_news_train['text'], ag_news_train['label'] ), columns=['content','label'])

ag_news_test = datasets.load_dataset('ag_news', split="test")
df_test = pd.DataFrame(zip(ag_news_test['text'], ag_news_test['label'] ), columns=['content','label'])

# yahoo news
# ['id', 'topic', 'question_title', 'question_content', 'best_answer']
yahoo_news_train = datasets.load_dataset('yahoo_answers_topics', split="train")
yahoo_news_test = datasets.load_dataset('yahoo_answers_topics', split="test")

#https://huggingface.co/datasets/newspop
news_train = datasets.load_dataset('newspop')

import datasets
cc_news = datasets.load_dataset('cc_news', split="train")

df = pd.DataFrame(zip(cc_news['title'], cc_news['text'] ))
df.columns = ['title','content']
df.to_csv('cc_news.csv')