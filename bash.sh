


############## ft gpt 
nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 1 --ft_pattern ep --gpu 0 --batch_size 8 \
 > ft.gpt2.ep.log &

# ft t5

CUDA_VISIBLE_DEVICES=7 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern pp --num_workers 4 \
   --maxlen 256 --ccsample 1 --ftepochs 4 --batch_size 4 > ft.t5.ep.log & 



nohup python -u ft.py --genm t5 --dsn_summary xsum --num_train_epochs 3 --ft_pattern summary --ccsample 0.1 --gpu 0



python -u augf.py --dsn uci --samplecnt 32 --max_aug_times 1 --aug generate \
                     --genft no  --genm gpt --filter mc --seed 0 \
                     --valid_files_cnt 16  --threads 16 --testvalid test \
                     --dpfuture_switch 1 --dpfuture_cls_switch 1 \
                     --candidates 128 --test_beams 64  \
                     --testbed 0 \
                     --num_return_sequences 3 --gpu 6,7 


python -u augf.py --dsn ag --samplecnt 8 --max_aug_times 1 --aug generate \
                     --genft no  --genm gpt --filter nlinsp --seed 0 \
                     --valid_files_cnt 16  --threads 16 --testvalid test \
                     --nli_switch 1 --nsp_switch 1 \
                     --candidates 128 --test_beams 64 --cls_score_thres 0.8 \
                     --testbed 0 \
                     --num_return_sequences 1 --gpu 0,1




# sdu  generate dvrl  ==> log_arxiv_testearlystop

nohup bash run_nlinsp.sh  0,1 &
nohup bash run_nlinsp.sh  2,3 &
nohup bash run_nlinsp.sh  4,5 &
nohup bash run_nlinsp.sh  6,7 &






nohup python -u rltoken.py  --gpu 0,1  --alpha 0.9 --future_steps 32 --beams 128 > rltoken.0p9.log &
nohup python -u rltoken.py  --gpu 2,3  --alpha 0.5 --future_steps 32 --beams 128 > rltoken.0p5.log &
nohup python -u rltoken.py  --gpu 4,5  --alpha 0.1 --future_steps 32 --beams 128 > rltoken.0p1.log &

nohup python -u aug_ppo.py > aug_ppo.log & 







############################################################################################################################################
ps aux|grep "augf.py"|grep -v grep | awk '{print $2}'|xargs kill -9
ps aux|grep "run_nlinsp.sh"|grep -v grep | awk '{print $2}'|xargs kill -9

ps aux|grep "dvrl_iter"|grep -v grep | awk '{print $2}'|xargs kill -9
ps aux|grep "dvrl_iter"|grep "3762"|grep -v grep | awk '{print $2}'|xargs kill -9


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