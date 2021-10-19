cat logb|grep 'samplecnt:128'|grep 'dsn:ag'|grep 'genm:t5'



##### augf
seed=$RANDOM
nohup python -u augf.py --dsn ag --samplecnt 8 --max_aug_times 1 --aug generate \
                  --genft ep  --genm gpt --filter dvrl --seed 0 \
                  --testvalid valid --valid_files_cnt 16  --threads 16 \
                  --abundance 3  --num_return_sequences 4 --gpu 0 --testbed 0 > augf.test.log & 


nohup python -u augf.py --dsn ag --samplecnt 8 --max_aug_times 1 --aug generate \
                  --genft tc  --genm gpt --filter dvrl --seed 100 \
                  --testvalid valid --valid_files_cnt 16  --threads 16 \
                  --abundance 3  --num_return_sequences 4 --gpu 1 --testbed 0 > augf.test.log & 



############## ft gpt 
nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 1 --ft_pattern ep --gpu 0 --batch_size 8 \
 > ft.gpt2.ep.log &



# ft t5

CUDA_VISIBLE_DEVICES=7 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern ep --num_workers 4 \
   --maxlen 256 --ccsample 1 --ftepochs 4 --batch_size 4 > ft.t5.ep.log & 


















nohup bash run_baseline.sh 1 128 3 8 7 &
nohup bash run_baseline.sh 3 128 3 8 6 &



############################################################################################################################################
ps aux|grep dvrl_iter|grep -v grep | awk '{print $2}'|xargs kill -9
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