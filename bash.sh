


############## ft gpt 
nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 1 --ft_pattern ep --gpu 0 --batch_size 8 \
 > ft.gpt2.ep.log &

# ft t5

CUDA_VISIBLE_DEVICES=7 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern pp --num_workers 4 \
   --maxlen 256 --ccsample 1 --ftepochs 4 --batch_size 4 > ft.t5.ep.log & 
nohup python -u ft.py --genm t5 --dsn_summary xsum --num_train_epochs 3 --ft_pattern summary --ccsample 0.1 --gpu 0






# norm

nohup python -u zsl.py --dsn yahoo --param 't5paws'  --gpu 2  > zsl.fly.yahoo.t5paws.log & 
nohup python -u zsl.py --dsn ag    --param 't5paws'  --gpu 4  > zsl.fly.ag.t5paws.log & 
nohup python -u zsl.py --dsn uci   --param 't5paws'  --gpu 6  > zsl.fly.uci.t5paws.log & 
nohup python -u zsl.py --dsn nyt   --param 't5paws'  --gpu 7  > zsl.fly.nyt.t5paws.log & 


python -u augfmcs.py --dsn uci --samplecnt 128 --genm t5 --candidates 64 --test_beams 32 --testmode 1 --gpu 0


nohup bash run.sh uci 0 & 
nohup bash run.sh ag 1 & 

nohup bash run.sh uci 3 & 
nohup bash run.sh ag 5 & 


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