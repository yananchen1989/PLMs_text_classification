
for i in {1..10}
do
	for candidates in 256 #512 1024 2048
	do
		for samplecnt in 32 64 128
		do
		seed=$RANDOM
		#seed=$(date +"%T")

			# lambda
			# python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug generate \
			# 		      --genft lambda  --genm gpt --filter clsembed --seed ${seed} \
			# 		      --testvalid test --candidates ${candidates} --gpu ${gpu} \
			# > ./log_baselines/${dsn}.generate.${samplecnt}.${max_aug_times}.${candidates}.gpt.lambda.clsembed.${seed}.log 2>&1	
			
			###### no finetune
			for genm in gpt t5
			do
				python -u augf.py --dsn ${1} --samplecnt ${samplecnt} --max_aug_times 1 --aug generate \
						      --genft no  --genm ${genm} --filter nlinsp --seed ${seed} --testbed 0 \
						      --testvalid test --candidates ${candidates} --gpu ${2} \
			> ./log_arxiv_nlinsp_noner/${1}.generate.${samplecnt}.${candidates}.${genm}.no.${seed}.log 2>&1
			done
	        ## end 
			
		done
	done 
done

