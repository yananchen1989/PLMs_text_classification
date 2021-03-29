# topic_classification_augmentation


datasets come from huggingface :

ag_news: https://huggingface.co/datasets/ag_news
yahoo_news: https://huggingface.co/datasets/yahoo_answers_topics


codes for fine-tune models follow the examples from huggingface: https://github.com/huggingface/transformers/tree/master/examples/language-modeling

baseline classifier: baseline_classifier.py
it load the albert model from tensorflow hub and acts as the encoder for the news text, 
the pooling layer is used for downstream classification.
All parameters are updated during training.
