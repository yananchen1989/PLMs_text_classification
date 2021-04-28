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


## zero-shot learning
In this scenerio, we do not use any original training set at all while we keep the test set the same as usual.
We just use label names / control codes as prompts to generate samples, as many as we want.
> zsl_generation.py
After generating enough samples, we use NLI model dowanloaded from https://huggingface.co/joeddav/bart-large-mnli-yahoo-answers 
to check the quality and the agreement between prompts and the generated samples.
You can refer to this blog for more info: https://joeddav.github.io/blog/2020/05/29/ZSL.html 
(This check can extremely boost the accuracy performance in next training step)

Therefore, we can train classifier based on these synthetic samples. It is surprising to find that the performance is quite satisfactory.













