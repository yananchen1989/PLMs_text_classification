max_aug_times=1

for i in {1..12}
do
	for samplecnt in 128
	do
		seed=$RANDOM
		#seed=$(date +"%T")
		for dsn in ag uci nyt #yelp2 amazon2
		do
			for aug in eda bt 
			do
				python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug ${aug} \
				       --seed ${seed} --testvalid test  \
				   > ./log_baselines/${dsn}.${aug}.${samplecnt}.${max_aug_times}.${seed}.log 2>&1
			done

			envcbert/bin/python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --aug cbert \
			      --max_aug_times ${max_aug_times} --seed ${seed} --testvalid test  \
			 > ./log_baselines/${dsn}.cbert.${samplecnt}.1.${seed}.log 2>&1
		done
	done
done 