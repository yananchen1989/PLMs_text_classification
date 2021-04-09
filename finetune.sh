#  gpt2 ctrl 

cd /root/yanan/berts/transformers/examples/language-modeling/

nohup python run_clm.py --model_name_or_path 'gpt2'  --output_dir 'finetuned_gpt2' \
         > fintune_gpt2.log &

# distilbert-base-uncased 
nohup python run_mlm.py  --model_name_or_path 'distilbert-base-uncased'  --output_dir 'finetuned_bert' \
          > log_fintune_mlm_bert &



ps aux|grep run_clm.py|awk '{print $2}'|xargs kill -9





scp  root@10.177.16.194:/root/yanan/berts/topic_classification_augmentation/ ./
scp -r gpt_zsl.tsv root@10.177.16.194:/root/yanan/berts/topic_classification_augmentation/


git add .;git commit -m "update";git push



nohup python -u baseline_classifier.py --aug fillin --ds ag --ner_set 0 > fillin_ag_noner.log &
nohup python -u baseline_classifier.py --aug fillin --ds ag --ner_set 1 > fillin_ag_yesner.log &

nohup python -u baseline_classifier.py --aug fillin --ds pop --ner_set 0 > fillin_pop_noner.log &
nohup python -u baseline_classifier.py --aug fillin --ds pop --ner_set 1 > fillin_pop_yesner.log &

nohup python -u baseline_classifier.py --aug fillin --ds yahoo --ner_set 0 > fillin_yahoo_noner.log &
nohup python -u baseline_classifier.py --aug fillin --ds yahoo --ner_set 1 > fillin_yahoo_yesner.log &


nohup python -u baseline_classifier.py --aug generate --ds ag --generate_m gpt2 > generate_ag_gpt2.log &
nohup python -u baseline_classifier.py --aug generate --ds pop --generate_m gpt2 > generate_pop_gpt2.log &
nohup python -u baseline_classifier.py --aug generate --ds yahoo --generate_m gpt2 > generate_yahoo_gpt2.log &

nohup python -u baseline_classifier.py --aug generate --ds ag --generate_m ctrl  > generate_ag_ctrl.log &
nohup python -u baseline_classifier.py --aug generate --ds pop --generate_m ctrl   > generate_pop_ctrl.log &
nohup python -u baseline_classifier.py --aug generate --ds yahoo --generate_m ctrl  > generate_yahoo_ctrl.log &


nohup python -u baseline_classifier.py --aug generate --ds ag --generate_m finetuned_gpt2  > generate_ag_ftgpt2.log &
nohup python -u baseline_classifier.py --aug generate --ds yahoo --generate_m finetuned_gpt2   > generate_yahoo_ftgpt2.log &
nohup python -u baseline_classifier.py --aug generate --ds pop --generate_m finetuned_gpt2   > generate_pop_ftgpt2.log &


nohup python -u baseline_classifier.py --aug no --ds yahoo --samplecnt -1 > augno_yahoo_full.log & 















