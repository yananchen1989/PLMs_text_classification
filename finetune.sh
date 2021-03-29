#  gpt2 ctrl  distilgpt2

cd /root/yanan/berts/transformers/examples/language-modeling/

nohup python run_clm.py --model_name_or_path 'gpt2'  --output_dir 'finetune_gpt2' \
         > log_fintune_clm_gpt2 &

# distilbert-base-uncased 
nohup python run_mlm.py  --model_name_or_path 'distilbert-base-uncased'  --output_dir 'finetune_bert' \
          > log_fintune_mlm_bert &



ps aux|grep run_clm.py|awk '{print $2}'|xargs kill -9






