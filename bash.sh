


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






CUDA_VISIBLE_DEVICES=5 nohup python -u ./run_clm_no_trainer.py \
                --num_train_epochs 3 \
                --train_file './df_cc_ners_train.txt' \
                --validation_file './df_cc_ners_test.txt' \
                --model_name_or_path gpt2 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --output_dir './gpt2_cc_ners' \
                --preprocessing_num_workers 1 --overwrite_cache True \
                --block_size 128 > ft_gpt_cc_ners.log &


CUDA_VISIBLE_DEVICES=6 nohup python -u ./run_clm_no_trainer.py \
                --num_train_epochs 3 \
                --train_file './df_cc_title_train.txt' \
                --validation_file './df_cc_title_test.txt' \
                --model_name_or_path gpt2 \
                --per_device_train_batch_size 16 \
                --per_device_eval_batch_size 16 \
                --output_dir './gpt2_cc_title' \
                --preprocessing_num_workers 1 --overwrite_cache True \
                --block_size 128 > ft_gpt_cc_title.log &




########## food
# CUDA_VISIBLE_DEVICES=7 nohup python run_mlm_no_trainer.py \
#     --num_train_epochs 12 \
#     --train_file './food/indgredients_train.txt' \
#     --validation_file './food/indgredients_test.txt' \
#     --model_name_or_path bert-base-uncased \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --output_dir './food/bert_indgredients' \
#     --preprocessing_num_workers 8 --overwrite_cache True \
#     --mlm_probability 0.15 \
#     --use_slow_tokenizer \
#          > ./food/bert_indgredients.log &



# CUDA_VISIBLE_DEVICES=6 nohup python -u ./run_clm_no_trainer.py \
#                 --num_train_epochs 12 \
#                 --train_file './food/recipe_train.txt' \
#                 --validation_file './food/recipe_test.txt' \
#                 --model_name_or_path gpt2 \
#                 --per_device_train_batch_size 8 \
#                 --per_device_eval_batch_size 8 \
#                 --output_dir './food/gpt_recipe' \
#                 --preprocessing_num_workers 8 --overwrite_cache True \
#                 --block_size 128 > ./food/gpt_recipe.log &


# CUDA_VISIBLE_DEVICES=3 nohup python -u ./run_clm_no_trainer.py \
#                 --num_train_epochs 12 \
#                 --train_file './food/ingre_recipe_train.txt' \
#                 --validation_file './food/ingre_recipe_test.txt' \
#                 --model_name_or_path gpt2 \
#                 --per_device_train_batch_size 8 \
#                 --per_device_eval_batch_size 8 \
#                 --output_dir './food/gpt_ingre_recipe' \
#                 --preprocessing_num_workers 8 --overwrite_cache True \
#                 --block_size 128 > ./food/gpt_ingre_recipe.log &




# CUDA_VISIBLE_DEVICES=1 nohup python -u ./run_summarization_no_trainer.py \
#             --num_train_epochs 12 \
#             --train_file "./food/ic_csv_train.csv" \
#             --validation_file "./food/ic_csv_test.csv" \
#             --model_name_or_path t5-base \
#             --per_device_train_batch_size 8 \
#             --per_device_eval_batch_size 8 \
#             --output_dir './food/t5_ingre_recipe' \
#             --max_target_length 128 \
#             --val_max_target_length 128 \
#             --preprocessing_num_workers 8 --overwrite_cache True \
#             --text_column ingredient_content \
#             --summary_column recipe \
#             --max_length 128 \
#             --model_type t5  --use_slow_tokenizer > ./food/t5_ingre_recipe.log &

#nohup python -u zsl.py --dsn yahoo --backbone roberta  > ./log_zsl/yahoo.roberta.log & 
nohup python -u zsl.py --dsn ag    --backbone roberta --expand pplm --gpu 0 > ./log_zsl/ag.roberta.pplm.log & 

#nohup python -u zsl.py --dsn yahoo --backbone nspbert  --gpu 1 > ./log_zsl/yahoo.nspbert.softmax.log & 
nohup python -u zsl.py --dsn ag    --backbone nspbert --expand pplm --gpu 1   > ./log_zsl/ag.nspbert.pplm.log & 

#nohup python -u zsl.py --dsn yahoo --backbone nli --softmax_score 1  --gpu 5 > ./log_zsl/yahoo.nli.softmax.log & 
nohup python -u zsl.py --dsn ag    --backbone nli --expand pplm --gpu 5  > ./log_zsl/ag.nli.pplm.log & 

#nohup python -u zsl.py --dsn yahoo --backbone simi --gpu 4 > ./log_zsl/yahoo.simi.log & 
nohup python -u zsl.py --dsn ag    --backbone simi --expand pplm --gpu 7 > ./log_zsl/ag.simi.pplm.log & 



nohup python -u zsl.py --dsn ag    --backbone nspbert --param t5paws --gpu 2   > ./log_zsl/ag.nspbert.t5paws.log & 
nohup python -u zsl.py --dsn yahoo --backbone nspbert --param t5paws --gpu 4   > ./log_zsl/yahoo.nspbert.t5paws.log & 

nohup python -u zsl.py --dsn ag    --backbone nli --param t5paws --gpu 3   > ./log_zsl/ag.nli.t5paws.log & 
nohup python -u zsl.py --dsn yahoo --backbone nli --param t5paws --gpu 5   > ./log_zsl/yahoo.nli.t5paws.log & 

nohup python -u zsl.py --dsn ag --backbone roberta --param t5paws --gpu 7   > ./log_zsl/ag.roberta.t5paws.log & 
nohup python -u zsl.py --dsn yahoo --backbone roberta --param t5paws --gpu 2   > ./log_zsl/yahoo.roberta.t5paws.log & 






#  nat for zsl
# CUDA_VISIBLE_DEVICES=1 nohup python -u ./run_clm_no_trainer.py \
#                 --num_train_epochs 12 \
#                 --train_file './finetunes/nat4gptzsl_train.txt' \
#                 --validation_file './finetunes/nat4gptzsl_test.txt' \
#                 --model_name_or_path gpt2 \
#                 --per_device_train_batch_size 16 \
#                 --per_device_eval_batch_size 16 \
#                 --output_dir './finetunes/gpt_nat_zsl' \
#                 --preprocessing_num_workers 8 --overwrite_cache True \
#                 --block_size 128  > ./finetunes/ft_gpt_nat_zsl.log & 


# t5 nat label ==> content
CUDA_VISIBLE_DEVICES=6 nohup python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 7 \
            --train_file "./finetunes/df_nat_train.csv" \
            --validation_file "./finetunes/df_nat_test.csv" \
            --model_name_or_path t5-base \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --output_dir './finetunes/t5_natcat' \
            --max_target_length 128 \
            --val_max_target_length 128 \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --text_column label \
            --summary_column content \
            --max_length 128 \
            --model_type t5  --use_slow_tokenizer > ./finetunes/ft_t5_nat.log &

# bart nat label ==> content
CUDA_VISIBLE_DEVICES=2 nohup python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 7 \
            --train_file "./finetunes/df_nat_train.csv" \
            --validation_file "./finetunes/df_nat_test.csv" \
            --model_name_or_path facebook/bart-base \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --output_dir './finetunes/bart_natcat_label2content' \
            --max_target_length 128 \
            --val_max_target_length 128 \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --text_column label \
            --summary_column content \
            --max_length 128 \
            --model_type bart  --use_slow_tokenizer > ./finetunes/ft_bart_nat_label2content.log &






# bart cc title ===> content
CUDA_VISIBLE_DEVICES=3 nohup python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 7 \
            --train_file "./finetunes/df_cc_train.csv" \
            --validation_file "./finetunes/df_cc_test.csv" \
            --model_name_or_path facebook/bart-base \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --output_dir './finetunes/bart_cc_title' \
            --max_target_length 128 \
            --val_max_target_length 128 \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --text_column title \
            --summary_column content \
            --max_length 128 \
            --model_type bart  --use_slow_tokenizer > ./finetunes/ft_bart_cc_title.log &


# t5 cc title ===> content
CUDA_VISIBLE_DEVICES=3 nohup python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 7 \
            --train_file "./finetunes/df_cc_train.csv" \
            --validation_file "./finetunes/df_cc_test.csv" \
            --model_name_or_path t5-base \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --output_dir './finetunes/t5_cc_title' \
            --max_target_length 128 \
            --val_max_target_length 128 \
            --preprocessing_num_workers 16 --overwrite_cache True \
            --text_column title \
            --summary_column content \
            --max_length 128 \
            --model_type t5  --use_slow_tokenizer > ./finetunes/ft_t5_cc_title.log &






# t5 nat content ===> label
CUDA_VISIBLE_DEVICES=7  nohup  python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 7 \
            --train_file "./finetunes/df_nat_train.csv" \
            --validation_file "./finetunes/df_nat_test.csv" \
            --model_name_or_path t5-base \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --output_dir './finetunes/t5_natcat' \
            --max_target_length 32 \
            --val_max_target_length 32 \
            --preprocessing_num_workers 8 --overwrite_cache True \
            --text_column content  \
            --summary_column label  \
            --max_length 64 \
            --model_type t5  --use_slow_tokenizer > ./finetunes/ft_t5_nat_content2label.log &

# bart nat content ===> label
CUDA_VISIBLE_DEVICES=1  nohup  python -u ./run_summarization_no_trainer.py \
            --num_train_epochs 7 \
            --train_file "./finetunes/df_nat_train.csv" \
            --validation_file "./finetunes/df_nat_test.csv" \
            --model_name_or_path facebook/bart-base \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --output_dir './finetunes/bart_natcat' \
            --max_target_length 32 \
            --val_max_target_length 32 \
            --preprocessing_num_workers 8 --overwrite_cache True \
            --text_column content  \
            --summary_column label  \
            --max_length 128 \
            --model_type bart  --use_slow_tokenizer  > ./finetunes/ft_bart_nat_content2label.log &


python -u openprompt_gen_syns.py --model t5-base --gpu 0 --template soft  --source_col ners --freeze_plm
python -u openprompt_gen_syns.py --model t5-base --gpu 3 --template mixed --source_col ners --freeze_plm
python -u openprompt_gen_syns.py --model t5-base --gpu 1 --template prefix  --source_col ners --freeze_plm

python -u openprompt_gen_syns.py --model t5-base --gpu 1 --template prefix --source_col title --freeze_plm


#################################################### mist ########################################################################################
python -c "import torch;print(torch.__version__)"
python -c "import transformers;print(transformers.__version__)"
python -c "import tensorflow;print(tensorflow.__version__)"
python -c "import datasets;print(datasets.__version__)"

python -c "import tensorflow_hub;print(tensorflow_hub.__version__)"
python -c "import tensorflow_text;print(tensorflow_text.__version__)"
python -c "import accelerate;print(accelerate.__version__)"



module load anaconda3
conda create -n env python=3.8
source activate env

conda install -c /scinet/mist/ibm/open-ce tensorflow==2.7.0 cudatoolkit=11.2

conda install -c /scinet/mist/ibm/open-ce scikit-learn


conda install -c /scinet/mist/ibm/open-ce pytorch=1.10.1 cudatoolkit=11.2
conda install -c /scinet/mist/ibm/open-ce transformers==4.9.2

conda install -c /scinet/mist/ibm/open-ce tensorflow-text
conda install -c /scinet/mist/ibm/open-ce tensorflow_hub


conda env remove --name myenv


############################################################################################################################################
ps aux|grep "run_clm_no_trainer.py"|grep -v grep | awk '{print $2}'|xargs kill -9
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