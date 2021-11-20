gpu=${1}
samplecnt=32
max_aug_times=1
candidates=128
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
		for genm in gpt t5 #ctrl
		do
			for genft in no #ep #pp tc
			do
				for nli_switch in 1 0
				do
					for nsp_switch in 1 0 
					do
					python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug generate \
					      --genft ${genft}  --genm ${genm} --filter nli,nsp --seed ${seed} \
					      --testvalid test \
					      --nsp_switch ${nsp_switch} --nli_switch ${nli_switch} \
					      --candidates ${candidates} \
					       --gpu ${gpu} \
		> ./log_arxiv_nlinsp/${dsn}.generate.3${samplecnt}.max_aug_times.${max_aug_times}.genm.${genm}.genft.${genft}.dpfuture_switch.${dpfuture_switch}.dpfuture_cls_switch.${dpfuture_cls_switch}.filter.${filter}.${seed}.log 2>&1
						
					done
				done
			done
		done
        ## end 
	done
done



