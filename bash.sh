

##### augf
seed=$RANDOM
nohup python -u augf.py --dsn ag --samplecnt 16 --max_aug_times 1 --aug generate \
                  --genft pp  --genm t5 --filter dvrl --seed 0 --epochs 5 --basetry 1 \
                  --testvalid valid --valid_files_cnt 16  --threads 16 \
                  --abundance 3  --num_return_sequences 8 --gpu 0 --testbed 1 > augf.test.log & 



############## ft gpt 
nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 1 --ft_pattern ep --gpu 0 --batch_size 8 \
 > ft.gpt2.ep.log &



# ft t5

CUDA_VISIBLE_DEVICES=7 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern pp --num_workers 4 \
   --maxlen 256 --ccsample 1 --ftepochs 4 --batch_size 4 > ft.t5.ep.log & 




python -u augf.py --dsn ag --samplecnt 128 --max_aug_times 1 --aug generate \
                  --genft pp  --genm t5 --filter no --seed 1234267  --do_train_test_parallel 1 \
                  --testvalid valid --valid_files_cnt 16  --threads 16  --epochs 4 \
                  --abundance 3  --num_return_sequences 8 --gpu 0 --testbed 1  --trunk_size 32 





# generate 
nohup bash run.sh 1 128 3 8 0 &
nohup bash run.sh 1 128 3 8 1 &
nohup bash run.sh 1 128 3 8 2 &
nohup bash run.sh 1 128 3 8 3 &

nohup bash run.sh 4 128 3 8 4 &
nohup bash run.sh 4 128 3 8 5 &
nohup bash run.sh 4 128 3 8 6 &
nohup bash run.sh 4 128 3 8 7 & 




# 31781 gpu1 倍增1 
# 8811 gpu7 倍增4









############################################################################################################################################
ps aux|grep augf|grep -v grep | awk '{print $2}'|xargs kill -9
ps aux --sort=start_time

tf_upgrade_v2 --infile main_data_valuation.py --outfile main_data_valuation_v2.py



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