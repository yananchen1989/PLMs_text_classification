
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


for i in 1 2 3 4 5 6 7 8 9 10
do 
   sbatch submit_stsa_albert.slurm
done



nohup python -u zsl.py --dsn ag --backbone nli --expand gpt_nofilter --gpu 5     > ./log_zsl/ag.nli.gpt_nofilter.log & 
nohup python -u zsl.py --dsn ag --backbone roberta --expand gpt_nofilter --gpu 6 > ./log_zsl/ag.roberta.gpt_nofilter.log & 
nohup python -u zsl.py --dsn ag --backbone simi --expand gpt_nofilter --gpu 7    > ./log_zsl/ag.simi.gpt_nofilter.log & 





CUDA_VISIBLE_DEVICES=3 python -u aug.py --dsn stsa --samplecnt 16 --aug eda,bt,generate \
         --backbone former --local_files_only




######## bart
CUDA_VISIBLE_DEVICES=0 nohup  python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 12 \
            --train_file "./finetunes/df_st_pp_train.csv" \
            --validation_file "./finetunes/df_st_pp_test.csv" \
            --model_name_or_path  facebook/bart-base \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --output_dir './finetunes/bart_st_pp' \
            --max_target_length 128 \
            --max_source_length 128 \
            --val_max_target_length 128 \
            --preprocessing_num_workers 56 --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --max_length 128 \
            --model_type bart  --use_slow_tokenizer --local_files_only > ./finetunes/bart_st_pp.log & 

CUDA_VISIBLE_DEVICES=1 nohup  python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 12 \
            --train_file "./finetunes/df_st_pps_train.csv" \
            --validation_file "./finetunes/df_st_pps_test.csv" \
            --model_name_or_path  facebook/bart-base \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --output_dir './finetunes/bart_st_pps' \
            --max_target_length 128 \
            --max_source_length 128 \
            --val_max_target_length 128 \
            --preprocessing_num_workers 56 --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --max_length 128 \
            --model_type bart  --use_slow_tokenizer --local_files_only  > ./finetunes/bart_st_pps.log & 


# t5
CUDA_VISIBLE_DEVICES=2 nohup  python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 12 \
            --train_file "./finetunes/df_st_pp_train.csv" \
            --validation_file "./finetunes/df_st_pp_test.csv" \
            --model_name_or_path  t5-base \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --output_dir './finetunes/t5_st_pp' \
            --max_target_length 128 \
            --max_source_length 128 \
            --val_max_target_length 128 \
            --preprocessing_num_workers 56 --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --max_length 128 \
            --model_type t5  --use_slow_tokenizer --local_files_only  > ./finetunes/t5_st_pp.log & 

CUDA_VISIBLE_DEVICES=3 nohup  python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 12 \
            --train_file "./finetunes/df_st_pps_train.csv" \
            --validation_file "./finetunes/df_st_pps_test.csv" \
            --model_name_or_path  t5-base \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --output_dir './finetunes/t5_st_pps' \
            --max_target_length 128 \
            --max_source_length 128 \
            --val_max_target_length 128 \
            --preprocessing_num_workers 56 --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --max_length 128 \
            --model_type t5  --use_slow_tokenizer --local_files_only > ./finetunes/t5_st_pps.log & 









############################################################################################################################################
ps aux|grep "run.sh"|grep -v grep | awk '{print $2}'|xargs kill -9
ps aux|grep "augf.py --dsn ag --samplecnt 1024"|grep -v grep | awk '{print $2}'|xargs kill -9
ps aux|grep "run_clm_no_trainer.py"|grep -v grep | awk '{print $2}'|xargs kill -9


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