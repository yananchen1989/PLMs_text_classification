# topic_classification_augmentation





## datasets for topic classification :

dataset | labels | 
------------ | -------------
ag_news | World,Sports,Business,Sci/Tech
yahoo_news | Society & Culture,Science & Mathematics,Health,...
pop_news | economy,microsoft,obama,palestine
tweet | technology,health,sports,politics,business,entertainment
uci | entertainment,business,science technology,health
bbc | business,entertainment,politics,sport,tech
bbc_sport | athletics,cricket,football,rugby,tennis

## main entrance
Several text classifiers are used, including transformer, albert, electra, DAN encoder, downloaded from tensorflow hub.
These models act as the encoder, and all parameters are updated during training.
> baseline_classifier.py


## data augmentation

BERT-based fillin augment method
> aug_fillinmask.py

backtranslation augment method, intermediate languages: > zh de fr ru
> aug_translation.py

generation augment method, we use GPT2 and CTRL to generate samples with 1:1
> aug_generation.py

## finetune LMs
Follow the examples from huggingface: https://github.com/huggingface/transformers/tree/master/examples/language-modeling
> finetune.sh
> run_clm.py
> run_mlm.py




