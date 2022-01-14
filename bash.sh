


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



nohup python -u cc_unsupervise_nli.py --gpu 0 --dsn ag --sampler 1 >  cc_unsupervise_nli.ag.log & 
nohup python -u cc_unsupervise_nli.py --gpu 1 --dsn yahoo --sampler 1 >  cc_unsupervise_nli.yahoo.log & 










############################################################################################################################################
ps aux|grep "augfmcs.py"|grep -v grep | awk '{print $2}'|xargs kill -9
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