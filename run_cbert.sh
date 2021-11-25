max_aug_times=1
for samplecnt in 32 64 128 
do
	for i in {1..12}
	do
		seed=$RANDOM
		#seed=$(date +"%T")
		for dsn in ag uci nyt #yelp2 amazon2
		do
			envcbert/bin/python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --aug cbert \
			      --max_aug_times ${max_aug_times} --seed ${seed} --testvalid test  \
			 > ./log_baselines/${dsn}.cbert.${samplecnt}.1.${seed}.log 2>&1
		done
	done
done 