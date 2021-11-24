max_aug_times=1
#samplecnt=${1}
candidates=256
gpu=${1}

for samplecnt in 32 64 128 
do
	for i in {1..12}
	do
		seed=$RANDOM
		#seed=$(date +"%T")
		for dsn in ag uci nyt #yelp2 amazon2
		do
			for aug in eda bt 
			do
				python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug ${aug} \
				       --seed ${seed} --testvalid test  --gpu ${gpu} \
				   > ./log_baselines/${dsn}.${aug}.${samplecnt}.${max_aug_times}.${seed}.log 2>&1
			done

			# envcbert/bin/python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --aug cbert \
			#       --max_aug_times ${max_aug_times} --seed ${seed} --testvalid test  \
			#  > ./log_baselines/${dsn}.cbert.${samplecnt}.${seed}.log 2>&1
			
			# lambda
			python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug generate \
					      --genft lambda  --genm gpt --filter clsembed --seed ${seed} \
					      --testvalid test --candidates ${candidates} --gpu ${gpu} \
			> ./log_baselines/${dsn}.generate.${samplecnt}.${max_aug_times}.${candidates}.gpt.lambda.clsembed.${seed}.log 2>&1	
			
			###### no finetune
			for genm in gpt t5
			do
				python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug generate \
						      --genft no  --genm ${genm} --filter nlinsp --seed ${seed} \
						      --testvalid test --candidates ${candidates} --gpu ${gpu} \
			> ./log_arxiv_nlinsp/${dsn}.generate.${samplecnt}.${max_aug_times}.${candidates}.${genm}.no.${seed}.log 2>&1
			done
	        ## end 
		done
	done
done 


