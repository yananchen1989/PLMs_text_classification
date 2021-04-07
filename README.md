# topic_classification_augmentation


datasets come from huggingface :

ag_news: https://huggingface.co/datasets/ag_news

World
Sports
Business
Sci/Tech



yahoo_news: https://huggingface.co/datasets/yahoo_answers_topics

Society & Culture
Science & Mathematics
Health
Education & Reference
Computers & Internet
Sports
Business & Finance
Entertainment & Music
Family & Relationships
Politics & Government

pop_news: https://huggingface.co/datasets/newspop

economy
microsoft
obama
palestine

codes for fine-tune models follow the examples from huggingface: https://github.com/huggingface/transformers/tree/master/examples/language-modeling

baseline classifier: baseline_classifier.py
it load the albert model from tensorflow hub https://tfhub.dev/tensorflow/albert_en_base/2 
and acts as the encoder for the news text, 
the pooling layer is used for downstream classification.
All parameters are updated during training.
