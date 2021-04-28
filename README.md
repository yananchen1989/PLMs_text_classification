# topic_classification_augmentation





## datasets for topic classification :

dataset | labels | 
------------ | -------------
ag_news | World,Sports,Business,Sci/Tech
yahoo_news | Society & Culture,Science & Mathematics,Health,...
pop_news | economy,microsoft,obama,palestine
tweet | technology,health,sports,politics,business,entertainment
uci | entertainment,business,science technology,health

## main entrance
Several text classifiers are used, including transformer, albert, electra, DAN encoder.

> baseline_classifier.py


## data augmentation

BERT-based fillin augment method
> aug_fillinmask.py

backtranslation augment method, intermediate languages: > zh de fr ru
> aug_translation.py

generation augment method, we use GPT2 and CTRL to generate samples with 1:1
> aug_generation.py


codes for fine-tune models follow the examples from huggingface: https://github.com/huggingface/transformers/tree/master/examples/language-modeling

baseline classifier: baseline_classifier.py
it load the albert model from tensorflow hub https://tfhub.dev/tensorflow/albert_en_base/2 
and acts as the encoder for the news text, 
the pooling layer is used for downstream classification.
All parameters are updated during training.
