gpu=${1}
samplecnt=32
max_aug_times=1
candidates=128
test_beams=64
for i in {1..10}
do
	seed=$RANDOM
	#seed=$(date +"%T")
	for dsn in agt #yelp2 amazon2
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
		for genm in gpt #t5 #ctrl
		do
			for genft in no #ep #pp tc
			do
				for filter in mc #mc,dvrl #nli enc nsp
				do
				python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug generate \
				      --genft ${genft}  --genm ${genm} --filter ${filter} --seed ${seed} \
				      --valid_files_cnt 16  --threads 16 --testvalid test \
				      --candidates ${candidates} --test_beams ${test_beams} --cls_score_thres 0.8 \
				       --gpu ${gpu} \
> ./log_arxiv_dpfuture_dvrl/${dsn}.generate.${samplecnt}.${max_aug_times}.${genm}.${genft}.filter.${filter}.${seed}.log 2>&1
				done
			done
		done
        ## end 
	done
done



