


############## ft gpt 
nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 1 --ft_pattern ep --gpu 0 --batch_size 8 \
 > ft.gpt2.ep.log &

# ft t5

CUDA_VISIBLE_DEVICES=7 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern pp --num_workers 4 \
   --maxlen 256 --ccsample 1 --ftepochs 4 --batch_size 4 > ft.t5.ep.log & 



nohup python -u ft.py --genm t5 --dsn_summary xsum --num_train_epochs 3 --ft_pattern summary --ccsample 0.1 --gpu 0








nohup python -u augf.py --dsn ag --samplecnt 128 --max_aug_times 1 --aug generate \
                 --genft no  --genm gpt --filter dvrl --seed 0  --epochs 2  --testbed 0 --valid_files_cnt 16  --threads 8 \
                  --abundance 3  --num_return_sequences 8 --gpu 7  > test.dvrl.nogpu.log & 

nohup python -u augf.py --dsn ag --samplecnt 128 --max_aug_times 1 --aug generate \
                 --genft no  --genm gpt --filter dvrl --seed 999  --epochs 2  --testbed 0 --valid_files_cnt 16  --threads 8 \
                  --abundance 3  --num_return_sequences 8 --gpu 6  > test.dvrl.gpu.log & 


# sdu  generate dvrl  ==> log_arxiv_testearlystop
nohup bash run.sh 1 128 3 8 0 &
nohup bash run.sh 1 128 3 8 1 &
nohup bash run.sh 1 128 3 8 2 &
nohup bash run.sh 1 128 3 8 3 &
#  aug_times:4
nohup bash run.sh 4 128 3 8 4 &
nohup bash run.sh 4 128 3 8 6 &
nohup bash run.sh 4 128 3 8 5 &
nohup bash run.sh 4 128 3 8 7 & 





# sec1dev  generate nli...
nohup bash run.sh 1 128 3 8 0 &
nohup bash run.sh 4 128 3 8 1 &





#v8
nohup bash run.sh 1 128 3 8 0 &
nohup bash run.sh 1 128 3 8 1 &
nohup bash run.sh 1 128 3 8 2 &
nohup bash run.sh 1 128 3 8 3 &
nohup bash run.sh 4 128 3 8 4 &
nohup bash run.sh 4 128 3 8 5 &


nohup bash run_nli.sh 1 128 3 8 6 &
nohup bash run_nli.sh 4 128 3 8 7 &





############################################################################################################################################
ps aux|grep "dvrl"|grep -v grep | awk '{print $2}'|xargs kill -9
ps aux --sort=start_time

tf_upgrade_v2 --infile main_data_valuation.py --outfile main_data_valuation_v2.py

alias wn='watch -n 1 nvidia-smi'

# 解压
unzip -o -d /home/sunny myfile.zip

# 压缩 torch_ds resource cache

for file in  cache_cbert
do
	#tar -zcvf ${file}.tar.gz ${file}
	tar -xvzf ${file}.tar.gz  -C ${file}
done


zip -r myfile.zip ./myfile


conda config --set auto_activate_base true