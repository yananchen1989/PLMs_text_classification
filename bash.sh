


############## ft gpt 
nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 1 --ft_pattern ep --gpu 0 --batch_size 8 \
 > ft.gpt2.ep.log &

# ft t5

CUDA_VISIBLE_DEVICES=7 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern pp --num_workers 4 \
   --maxlen 256 --ccsample 1 --ftepochs 4 --batch_size 4 > ft.t5.ep.log & 
nohup python -u ft.py --genm t5 --dsn_summary xsum --num_train_epochs 3 --ft_pattern summary --ccsample 0.1 --gpu 0








nohup bash run.sh uci 5 &
nohup bash run.sh ag 6 &
nohup bash run.sh nyt 7 &


for gpu in 0,1 2,3 
do
nohup bash run_cbert.sh ${gpu} & 
done




seed=$RANDOM
for dsn in uci ag nyt 
do
nohup envcbert/bin/python -u augf.py --dsn ${dsn} --samplecnt 128 --aug cbert \
      --max_aug_times 1 --seed ${seed} --testvalid test  \
 > ./log_baselines/${dsn}.cbert.128.1.${seed}.log 2>&1 &
done














nohup python -u zsclassifier_fly.py --dsn uci --gpu 5 --fbs 64 --para 1 > zsl.fly.t5.uci.64.log & 
nohup python -u zsclassifier_fly.py --dsn ag  --gpu 6 --fbs 64 --para 1 > zsl.fly.t5.ag.64.log & 
nohup python -u zsclassifier_fly.py --dsn yahoo  --gpu 7 --fbs 64 --para 1 > zsl.fly.t5.yahoo.64.log & 




nohup python -u zsl_gpt.py --dsn uci --fbs_gen 128 --gpu 6 > zsl_gpt.uci.128.log & 
nohup python -u zsl_gpt.py --dsn yahoo --fbs_gen 64 --gpu 7 > zsl_gpt.yahoo.64.log & 
nohup python -u zsl_gpt.py --dsn yahoo --fbs_gen 128 --gpu 5 > zsl_gpt.yahoo.128.log & 
############################################################################################################################################
ps aux|grep "augf.py"|grep -v grep | awk '{print $2}'|xargs kill -9
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