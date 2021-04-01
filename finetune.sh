#  gpt2 ctrl 

cd /root/yanan/berts/transformers/examples/language-modeling/

nohup python run_clm.py --model_name_or_path 'gpt2'  --output_dir 'finetuned_gpt2' \
         > fintune_gpt2.log &

# distilbert-base-uncased 
nohup python run_mlm.py  --model_name_or_path 'distilbert-base-uncased'  --output_dir 'finetuned_bert' \
          > log_fintune_mlm_bert &



ps aux|grep run_clm.py|awk '{print $2}'|xargs kill -9





scp  root@10.177.16.194:/root/yanan/berts/multi_head_attention.py ./
scp -r 
