max_aug_times=${1} # parameter R
samplecnt=${2} # simulate a low-data regime, where 32 samples per category are selected as anchor data
abundance=${3} #parameter B 
num_return_sequences=${4}
gpu=${5}
while true
do
	seed=$RANDOM
	#seed=$(date +"%T")
	for dsn in ag uci nyt #yelp2 amazon2
	do
		# for aug in eda bt 
		# do
		# 	python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug ${aug} \
		# 	      --gpu ${gpu} --seed ${seed} --testvalid test  \
		# 	   > ./log_arxiv_32/${dsn}.${aug}.${samplecnt}.${max_aug_times}.${seed}.log 2>&1
		# done

		# for aug in cbert
		# do
		# 	envcbert/bin/python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --aug ${aug} \
		# 	      --max_aug_times ${max_aug_times} \
		# 	 > ${dsn}.${aug}.${samplecnt}.${seed}.log 2>&1
		# done

		###### no finetune
		for genm in gpt t5 #ctrl
		do
			for genft in ep #pp tc
			do
				#for filter in dvrl no
				for filter in no cls #nli enc nsp
				do
				python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug generate \
				      --genft ${genft}  --genm ${genm} --filter ${filter} --seed ${seed} \
				      --valid_files_cnt 16  --threads 16 --testvalid test \
				      --abundance ${abundance}  --num_return_sequences ${num_return_sequences} --gpu ${gpu} \
				      > ./log_arxiv_ep/${dsn}.generate.${samplecnt}.max_aug_times.${max_aug_times}.genm.${genm}.genft.${genft}.filter.${filter}.abundance.${abundance}.num_return_sequences.${num_return_sequences}.${seed}.log 2>&1
				done
			done
		done
        ## end 
	done
done



