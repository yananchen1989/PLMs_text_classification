max_aug_times=1
samplecnt=${1}
candidates=${2}
test_beams=8
gpu=${3}
for i in {1..10}
do
	seed=$RANDOM
	#seed=$(date +"%T")
	for dsn in ag uci #yelp2 amazon2
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
		for genm in gpt t5
		do
			python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug generate \
					      --genft no  --genm ${genm} --filter nlinsp --seed ${seed} \
					      --testvalid test \
					      --candidates ${candidates} --test_beams ${test_beams} \
					       --gpu ${gpu} \
		> ./log_arxiv_nlinsp/${dsn}.generate.${samplecnt}.${max_aug_times}.${candidates}.${genm}.no.${seed}.log 2>&1
		done
        ## end 
	done
done



