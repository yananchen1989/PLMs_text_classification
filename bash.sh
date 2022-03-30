
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






nohup python -u zsl.py --dsn ag --backbone nli --expand gpt_nofilter --gpu 5     > ./log_zsl/ag.nli.gpt_nofilter.log & 
nohup python -u zsl.py --dsn ag --backbone roberta --expand gpt_nofilter --gpu 6 > ./log_zsl/ag.roberta.gpt_nofilter.log & 
nohup python -u zsl.py --dsn ag --backbone simi --expand gpt_nofilter --gpu 7    > ./log_zsl/ag.simi.gpt_nofilter.log & 




# bart cc title ===> content


CUDA_VISIBLE_DEVICES=0 nohup python -u /home/w/wluyliu/yananc/topic_classification_augmentation/run_summarization_no_trainer.py \
            --num_train_epochs 10 \
            --train_file "/home/w/wluyliu/yananc/topic_classification_augmentation/finetunes/df_cc_train_tc.csv" \
            --validation_file "/home/w/wluyliu/yananc/topic_classification_augmentation/finetunes/df_cc_test_tc.csv" \
            --model_name_or_path  facebook/bart-base \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --output_dir '/scratch/w/wluyliu/yananc/finetune/bart_cc_tc' \
            --max_target_length 128 \
            --val_max_target_length 128 \
            --preprocessing_num_workers 32 --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --max_length 128 \
            --model_type bart  --use_slow_tokenizer  &

# t5-base facebook/bart-base






for gpu in 0 1 2 3 4 5 6 7
do
   nohup bash run_albert.sh ${gpu} & 
done


nohup bash run_former.sh & 




for gpu in 0 1 2 3 4 5 6 7
do
   nohup bash run.sh ${gpu} & 
done

for i in 0 1 2 3
do
nohup bash run_edabt.sh & 
done

#################################################### mist ########################################################################################
python -c "import torch;print(torch.__version__)"
python -c "import transformers;print(transformers.__version__)"
python -c "import tensorflow;print(tensorflow.__version__)"
python -c "import datasets;print(datasets.__version__)"

python -c "import tensorflow_hub;print(tensorflow_hub.__version__)"
python -c "import tensorflow_text;print(tensorflow_text.__version__)"
python -c "import accelerate;print(accelerate.__version__)"


python -u augf.py --dsn uci --samplecnt 8 --max_aug_times 1  \
                  --seed 873  --aug generate \
                   --gpu 7


python -u augf.py --dsn uci --samplecnt 8 --max_aug_times 1  \
                            --seed 77777  --aug eda,bt

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