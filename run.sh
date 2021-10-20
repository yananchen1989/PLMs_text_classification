max_aug_times=${1} # parameter R
samplecnt=${2} # simulate a low-data regime, where 32 samples per category are selected as anchor data
abundance=${3} #parameter B 
num_return_sequences=${4}
gpu=${5}
while true
do
	seed=$RANDOM
	#seed=$(date +"%T")
	for dsn in ag uci nyt
	do
		# for aug in eda bt 
		# do
		# 	python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --aug ${aug} \
		# 	    --max_aug_times ${max_aug_times}  --gpu ${gpu} --seed ${seed} --testvalid valid --trunk_size 4 \
		# 	   > ./log_arxiv_baseline/${dsn}.${aug}.${samplecnt}.${max_aug_times}.${seed}.log 2>&1
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
			for genft in no tc pp ep
			do
				for filter in dvrl no nli cls enc #nsp,enc   #nli cls nsp enc no 
				do
				python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug generate \
				      --genft ${genft}  --genm ${genm} --filter ${filter} --seed ${seed} \
				      --testvalid valid --valid_files_cnt 16  --threads 8 \
				      --abundance ${abundance}  --num_return_sequences ${num_return_sequences} --gpu ${gpu} \
				      > ./log_arxiv/${dsn}.generate.${samplecnt}.max_aug_times.${max_aug_times}.genm.${genm}.genft.${genft}.filter.${filter}.abundance.${abundance}.num_return_sequences.${num_return_sequences}.${seed}.log 2>&1
				done
			done
		done

	done
done



