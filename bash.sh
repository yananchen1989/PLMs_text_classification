


############## ft gpt 
nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 1 --ft_pattern ep --gpu 0 --batch_size 8 \
 > ft.gpt2.ep.log &

# ft t5

CUDA_VISIBLE_DEVICES=7 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern pp --num_workers 4 \
   --maxlen 256 --ccsample 1 --ftepochs 4 --batch_size 4 > ft.t5.ep.log & 
nohup python -u ft.py --genm t5 --dsn_summary xsum --num_train_epochs 3 --ft_pattern summary --ccsample 0.1 --gpu 0






# norm


 CUDA_VISIBLE_DEVICES=5 nohup envcbert/bin/python -u pplm.py --pretrained_model gpt2-medium  --dsn yahoo  --length 64 --gamma 1.5 \
   --num_iterations 3 --num_samples 64 --stepsize 0.03 --window_length 5 --kl_scale 0.01 \
   --gm_scale 0.99   --sample  > pplm.yahoo.log & 

for dsn in ag yahoo 
do
   for i in 1  
   do
      for gpu in 2 #2 3 4 5 6 7
      do
         seed=$RANDOM
         CUDA_VISIBLE_DEVICES=${gpu} nohup envcbert/bin/python -u pplm.py  --pretrained_model gpt2-medium  \
            --dsn ${dsn}  --length 64 --gamma 1.5 \
         --num_iterations 3 --num_samples 64 --stepsize 0.03 --window_length 5 --kl_scale 0.01 \
         --gm_scale 0.99 --seed ${seed} --sample  > ./log_pplm/pplm.${dsn}.${seed}.log & 
      done
   done
done 



CUDA_VISIBLE_DEVICES=0 nohup python -u ./run_clm_no_trainer.py \
                --num_train_epochs 3 \
                --train_file './df_nat_train.txt' \
                --validation_file './df_nat_test.txt' \
                --model_name_or_path gpt2 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --output_dir './gpt2_natcat' \
                --preprocessing_num_workers 1 --overwrite_cache True \
                --block_size 128  > ft_gpt_nat.log & 







CUDA_VISIBLE_DEVICES=1 nohup python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 3 \
            --train_file "df_nat_train_sample.csv" \
            --validation_file "df_nat_test.csv" \
            --model_name_or_path t5-base \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --output_dir './t5_natcat_sample' \
            --max_target_length 256 \
            --val_max_target_length 256 \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --text_column prefix \
            --summary_column content \
            --max_length 256 \
            --model_type t5  --use_slow_tokenizer > ft_t5_nat_sample.log &


CUDA_VISIBLE_DEVICES=2 nohup python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 3 \
            --train_file "df_nat_train.csv" \
            --validation_file "df_nat_test.csv" \
            --model_name_or_path t5-base \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --output_dir './t5_natcat' \
            --max_target_length 256 \
            --val_max_target_length 256 \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --text_column prefix \
            --summary_column content \
            --max_length 256 \
            --model_type t5  --use_slow_tokenizer > ft_t5_nat.log &




CUDA_VISIBLE_DEVICES=3 nohup python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 3 \
            --train_file "df_cc_ners_train.csv" \
            --validation_file "df_cc_ners_test.csv" \
            --model_name_or_path t5-base \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --output_dir './t5_ners_cc' \
            --max_target_length 256 \
            --val_max_target_length 256 \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --text_column ners \
            --summary_column content \
            --max_length 256 \
            --model_type t5  --use_slow_tokenizer > ft_t5_cc_ners.log &



CUDA_VISIBLE_DEVICES=4 nohup python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 3 \
            --train_file "df_cc_title_train.csv" \
            --validation_file "df_cc_title_test.csv" \
            --model_name_or_path t5-base \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --output_dir './t5_title_cc' \
            --max_target_length 256 \
            --val_max_target_length 256 \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --text_column title \
            --summary_column content \
            --max_length 256 \
            --model_type t5  --use_slow_tokenizer > ft_t5_cc_title.log &






CUDA_VISIBLE_DEVICES=6 nohup python -u ./run_clm_no_trainer.py \
                --num_train_epochs 4 \
                --train_file './df_nat_train_sample.txt' \
                --validation_file './df_nat_test.txt' \
                --model_name_or_path gpt2 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --output_dir './gpt2_natcat_sample' \
                --preprocessing_num_workers 1 --overwrite_cache True \
                --block_size 128 > ft_gpt_nat_sample.log &





















############################################################################################################################################
ps aux|grep "pplm.py"|grep -v grep | awk '{print $2}'|xargs kill -9
ps aux|grep "run.sh"|grep -v grep | awk '{print $2}'|xargs kill -9
ps aux|grep "run_cbert.sh"|grep -v grep | awk '{print $2}'|xargs kill -9

ps aux|grep "dvrl_iter"|grep -v grep | awk '{print $2}'|xargs kill -9
ps aux|grep "dvrl_iter"|grep "3762"|grep -v grep | awk '{print $2}'|xargs kill -9


ps aux --sort=start_time

tf_upgrade_v2 --infile main_data_valuation.py --outfile main_data_valuation_v2.py

alias wn='watch -n 1 nvidia-smi'

# 解压
unzip -o -d /home/sunny myfile.zip
tar -zxvf torch_ds.tar.gz;tar -zxvf cache_cbert.tar.gz;




for file in resource torch_ds cache_cbert
do
	#tar -zcvf ${file}.tar.gz ${file}
	tar -xvzf ${file}.tar.gz  -C ${file}
done


zip -r myfile.zip ./myfile

cat *.pgn | grep "Result" | sort | uniq -c
conda config --set auto_activate_base true