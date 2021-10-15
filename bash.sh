


##### augf

nohup python -u augf.py --dsn ag --samplecnt 128 --max_aug_times 1 --aug generate \
				      --genft no  --genm gpt --filter dvrl --abundance 3  --basetry 1 --testvalid valid \
				      --verbose 1 --epochs 50 --freq 50 --threads 64 --gpu 0,1,2,3 > augf.test.log & 



############## ft gpt 
nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 1 --ft_pattern tc --gpu 6  --batch_size 8 \
 > ft.gpt2.tc.log &

nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 1 --ft_pattern pp --gpu 7 --batch_size 8 \
 > ft.gpt2.pp.log &


# ft t5
CUDA_VISIBLE_DEVICES=5 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern pp --num_workers 4  \
   --maxlen 256 --ccsample 1 --ftepochs 4 --batch_size 4 > ft.t5.pp.log & 

CUDA_VISIBLE_DEVICES=4 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern tc --num_workers 4 \
   --maxlen 256 --ccsample 1 --ftepochs 4 --batch_size 4 > ft.t5.tc.log & 




















############################################################################################################################################
ps aux|grep dvrl_iter|grep -v grep | awk '{print $2}'|xargs kill -9


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