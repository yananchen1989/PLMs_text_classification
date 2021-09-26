

pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

nohup bash run.sh 0 128 & 
nohup bash run.sh 1 128 & 



# enter tensorflow 1.4

for i in {1..10}
do
	seed=$RANDOM
	for dsn in blog adult
	do
		python main_data_valuation.py --inner_iterations 100 --batch_size 256 --iterations  1000 \
	       --train_no 512 --valid_no 64 --dsn ${dsn} > dvrl.${dsn}.${seed}.log 
	done
done



python main_data_valuation.py --inner_iterations 50 --batch_size 256 --iterations  1000 \
			--hidden_dim 768 --comb_dim 512 \
        --dsn stsa --norm 0  --perf_metric auc








cp *.py /Users/yanan/Desktop/thesis/my_augmentation/
cp -r * /Users/yanan/Desktop/thesis/my_augmentation/




for gpu in 0 1
do
nohup python -u aug_ppo_task.py --dsn ag --samplecnt 32   --ppo_batchsize 32 \
	           --load_bert 1 --noeval 1  --ppo_epoch 2000 \
	           --add_external_ft ${gpu}   --add_external_ppo 1 --external_frac 0.1 \
	            --init_kl_coef 0.1   \
	            --gpt_ft 1 --ref_ft 1 --gpu ${gpu} > log_ft${gpu}_ppo1 &
done




nohup python -u aug_ppo_nli.py --dsn uci --samplecnt 32 --steps 1000  --ref_ft 1 --gpt_ft 0 --gpu 1 \
   > ppo.stsa.64.ft10.log & 


nohup python -u aug_ppo_nli.py --dsn uci --samplecnt 32 --steps 1000  \
        --ref_ft 0 --gpt_ft 0 --init_kl_coef 0.1 --gpu 1 \
   > ppo.stsa.64.ft00.kl0p1.log & 







nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 0.5 --ft_pattern tc --gpu 1  \
 > ft.gpt2.tc.log &

nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 0.5 --ft_pattern pp  --gpu 1  \
 > ft.gpt2.pp.log &



python -u augf.py --dsn ag --samplecnt 32 --max_aug_times 1 --aug generate \
				      --genft pp --filter nsp --genm t5 --abundance 1 --gpu 1





# ft t5
CUDA_VISIBLE_DEVICES=1 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern pp --num_workers 4  \
   --maxlen 512 --ccsample 0.4 --ftepochs 7 --batch_size 8 > ft.t5.pp.log & 

CUDA_VISIBLE_DEVICES=0 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern tc --num_workers 1 \
   --maxlen 512 --ccsample 0.35 --ftepochs 4 --batch_size 8 > ft.t5.tc.log & 






#python -u ft.py --genm t5 --num_train_epochs 7 --ccsample 1 --ft_pattern pp --maxlen 512 --gpu 0


python -u augf.py --dsn ag --samplecnt 64 --max_aug_times 1 \
					--aug generate \
			      	--genft tc --filter enc --genm gpt --abundance 3 --gpu 0



python -u augf.py --dsn uci --samplecnt 64 --max_aug_times 1 \
					--aug generate \
			      	--genft tc --filter enc --genm gpt --abundance 3 --gpu 0

