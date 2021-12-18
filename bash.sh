


############## ft gpt 
nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 1 --ft_pattern ep --gpu 0 --batch_size 8 \
 > ft.gpt2.ep.log &

# ft t5

CUDA_VISIBLE_DEVICES=7 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern pp --num_workers 4 \
   --maxlen 256 --ccsample 1 --ftepochs 4 --batch_size 4 > ft.t5.ep.log & 
nohup python -u ft.py --genm t5 --dsn_summary xsum --num_train_epochs 3 --ft_pattern summary --ccsample 0.1 --gpu 0




python -u augf.py --dsn ag --samplecnt 8 --max_aug_times 1 --aug generate \
                     --genft no  --genm gpt --filter nlinsp --seed 1 \
                     --testvalid test --testbed 0 \
                     --candidates 256 --test_beams 32  \
                      --gpu 6,7


python -u augf.py --dsn ag --samplecnt 8 --max_aug_times 1 --aug generate \
                     --genft no  --genm t5 --filter nlinsp --seed 0 \
                     --testvalid test --testbed 0 \
                     --candidates 256 --test_beams 32  \
                      --gpu 0,1 



python -u augf.py --dsn ag --samplecnt 8 --max_aug_times 1 --aug generate \
                  --genft lambda  --genm gpt --filter clsembed --seed 0 \
                  --testvalid test --candidates 128 --gpu 6,7 --epochs 2 --verbose 1



envcbert/bin/python -u augf.py --dsn ag --samplecnt 32 --aug cbert \
               --max_aug_times 1 --seed 3333 --testvalid test --epochs 2

python -u augf.py --dsn ag --samplecnt 8 --max_aug_times 1 --aug eda \
                --seed 0 --testvalid test --epochs 2  --verbose 1 

python -u augf.py --dsn ag --samplecnt 8 --max_aug_times 1 --aug bt \
                --seed 1 --testvalid test --epochs 1 --gpu 1 --verbose 1 

# sdu  generate dvrl  ==> log_arxiv_testearlystop

for gpu in 0,1 2,3 4,5 6,7
do
   nohup bash run.sh  ${gpu} &
done


for gpu in 0,1 2,3 
do
nohup bash run_cbert.sh ${gpu} & 
done

nohup bash run_ablation.sh 4,5 &
nohup bash run_ablation.sh 6,7 &



nohup bash run_ablation.sh 0 &
# nohup python -u rltoken.py  --gpu 0,1  --alpha 0.9 --future_steps 32 --beams 128 > rltoken.0p9.log &
# nohup python -u rltoken.py  --gpu 2,3  --alpha 0.5 --future_steps 32 --beams 128 > rltoken.0p5.log &
# nohup python -u rltoken.py  --gpu 4,5  --alpha 0.1 --future_steps 32 --beams 128 > rltoken.0p1.log &

# nohup python -u aug_ppo.py > aug_ppo.log & 



seed=$RANDOM
for dsn in uci ag nyt 
do
nohup envcbert/bin/python -u augf.py --dsn ${dsn} --samplecnt 128 --aug cbert \
      --max_aug_times 1 --seed ${seed} --testvalid test  \
 > ./log_baselines/${dsn}.cbert.128.1.${seed}.log 2>&1 &
done



nohup python -u zsl_gpt.py --dsn uci --genm t5 --gpu 6,7 --fbs_gen 128 \
          > ./log_zsl/zsl_gpt.uci.t5.128.log & 



nohup python -u zsl_pure_classifier.py --dsn uci --gpu 0 --fbs 32 >  ./log_zsl/zsl.flip.gen.uci.32.log & 
nohup python -u zsl_pure_classifier.py --dsn uci  --gpu 1 --fbs 64 >  ./log_zsl/zsl.flip.gen.uci.64.log & 
nohup python -u zsl_pure_classifier.py --dsn uci  --gpu 2 --fbs 128 > ./log_zsl/zsl.flip.gen.uci.128.log & 


nohup python -u zsl_pure_classifier.py --dsn ag --gpu 3 --fbs 32 >  ./log_zsl/zsl.flip.gen.ag.32.log & 
nohup python -u zsl_pure_classifier.py --dsn ag  --gpu 4 --fbs 64 >  ./log_zsl/zsl.flip.gen.ag.64.log & 
nohup python -u zsl_pure_classifier.py --dsn ag  --gpu 5 --fbs 128 > ./log_zsl/zsl.flip.gen.ag.128.log &

nohup python -u zsl_pure_classifier.py --dsn yahoo --gpu 6 --fbs 32 >  ./log_zsl/zsl.flip.gen.yahoo.32.log & 
nohup python -u zsl_pure_classifier.py --dsn yahoo  --gpu 7 --fbs 64 >  ./log_zsl/zsl.flip.gen.yahoo.64.log & 











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