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

cnndm_news = datasets.load_dataset('cnn_dailymail', '3.0.0')

df = pd.DataFrame(zip(cc_news['title'], cc_news['text'] ))
df.columns = ['title','content']
df.to_csv('cc_news.csv', index=False)





# torchtext
import torchtext 
import pandas as pd 

dsnf = {'ag':torchtext.datasets.AG_NEWS, 'yahoo':torchtext.datasets.YahooAnswers, 
	'dbpedia':torchtext.datasets.DBpedia, 'yelp2':torchtext.datasets.YelpReviewPolarity,
        'yelp5':torchtext.datasets.YelpReviewFull, 
        'amazon2':torchtext.datasets.AmazonReviewPolarity, 'amazon5':torchtext.datasets.AmazonReviewFull,
        'imdb':torchtext.datasets.IMDB}

path = './torch_ds'
for dsn, f in dsnf.items():
	if dsn != 'yahoo':
		continue
	ds_train, ds_test =  f(root=path, split=('train', 'test'))
	df_train,df_test = pd.DataFrame(ds_train, columns=['label', 'content']), pd.DataFrame(ds_test, columns=['label','content'])
	# df_train.to_csv('{}/{}_train.csv'.format(path, dsn), index=False)
	# df_test.to_csv('{}/{}_test.csv'.format(path, dsn), index=False)






next(ds_train)

# AG_NEWS YahooAnswers  DBpedia YelpReviewPolarity YelpReviewFull  AmazonReviewPolarity AmazonReviewFull 
# IMDB



contents = df_train.loc[df_train['label']==2].sample(32)['content'].tolist()

sent1 = "I have been making annual trips to Vegas for the past ten years, and I have never failed to visit this place for a slice of pizza.  I typically order the classic thin crust cheese pizza with a Coors, and I am always extremely  satisfied. "
sent2 =  "The store is in a visibly older building, but the customer service is great. Every single employee greets you with a smile, and even the stockers engage and assist customers. If they had a finished ceiling and nicer looking floors, the store would look much more appealing and would definitely earn five stars."

ARTICLE = sent1 + ' ' + sent2
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# T5 uses a max_length of 512 so we cut the article to 512 tokens.
inputs = tokenizer.encode(ARTICLE, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(inputs, max_length=512, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
syn_sents = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)


summary_text = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text)











