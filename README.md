# topic_classification_augmentation





## datasets for topic classification :


dataset | labels 
------------ | -------------
ag_news | World,Sports,Business,Sci/Tech
yahoo_news | Society & Culture,Science & Mathematics,Health,Education & Reference...
pop_news | economy,microsoft,obama,palestine


codes for fine-tune models follow the examples from huggingface: https://github.com/huggingface/transformers/tree/master/examples/language-modeling

baseline classifier: baseline_classifier.py
it load the albert model from tensorflow hub https://tfhub.dev/tensorflow/albert_en_base/2 
and acts as the encoder for the news text, 
the pooling layer is used for downstream classification.
All parameters are updated during training.
