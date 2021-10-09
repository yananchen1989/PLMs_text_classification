


#nohup python -u bert_pair_nsp.py --dsn snli --max_length 64 --gpu 0 > pair.snli.log &
# nohup python -u bert_pair_nsp.py --dsn news --max_length 64 --gpu 0 > pair.news.log &

# nohup python -u bert_pair_nsp.py --dsn yelp2 --max_length 64 --gpu 1 > pair.yelp2.log &
# nohup python -u bert_pair_nsp.py --dsn imdb --max_length 64 --gpu 1 > pair.imdb.log &
# nohup python -u bert_pair_nsp.py --dsn yahoo --max_length 64 --gpu 1 > pair.yahoo.log &


# CUDA_VISIBLE_DEVICES=0 nohup  python -u bert_pair_nsp.py --dsn ag --ft 0  --gpu 0 > pair.ppo.ag.ft0.log &
# CUDA_VISIBLE_DEVICES=1 nohup  python -u bert_pair_nsp.py --dsn ag --ft 1  --gpu 1 > pair.ppo.ag.ft1.log &



abundance=3 #parameter B 
max_aug_times=1 # parameter R
samplecnt=128 # simulate a low-data regime, where 32 samples per category are selected as anchor data
while true
do
	seed=$RANDOM
	#seed=$(date +"%T")
	for dsn in ag uci #nyt
	do
		# for aug in eda bt 
		# do
		# 	python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --aug ${aug} \
		# 	    --max_aug_times ${max_aug_times}  \
		# 	   > ${dsn}.${aug}.${samplecnt}.${seed}.log
		# done

		# for aug in cbert
		# do
		# 	envcbert/bin/python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --aug ${aug} \
		# 	      --max_aug_times ${max_aug_times} \
		# 	 > ${dsn}.${aug}.${samplecnt}.${seed}.log
		# done

		###### no finetune
		for genm in gpt t5 #ctrl
		do
			for genft in no tc pp
			do
				for filter in nli cls nsp enc dvrl no 
				do
				python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug generate \
				      --genft ${genft} --filter ${filter} --genm ${genm} --abundance ${abundance}  --testbed 1 \
				      > ${dsn}.generate.${samplecnt}.genm_${genm}.genft_${genft}.filter_${filter}.${seed}.log
				done
			done
		done

        ### internal finetune
		# for genm in gpt 
		# do
		# 	for genft in lambda entire 
		# 	do
		# 		for filter in nli cls nsp enc dvrl no
		# 		do
		# 		python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug generate \
		# 		      --genft ${genft} --filter ${filter} --genm ${genm} --abundance ${abundance}   \
		# 		      > ${dsn}.generate.${samplecnt}.genm_${genm}.genft_${genft}.filter_${filter}.${seed}.log
		# 		done
		# 	done
		# done       

		###### external finetune
		# for genm in t5 gpt
		# do
		# 	for genft in tc pp
		# 	do
		# 		for filter in nli cls nsp enc dvrl no
		# 		do
		# 		python -u augf.py --dsn ${dsn} --samplecnt ${samplecnt} --max_aug_times ${max_aug_times} --aug generate \
		# 		      --genft ${genft} --filter ${filter} --genm ${genm} --abundance ${abundance}   \
		# 		      > ${dsn}.generate.${samplecnt}.genm_${genm}.genft_${genft}.filter_${filter}.${seed}.log
		# 		done 
		# 	done
		# done
	done
done






# nohup bash run.sh  &

# test
# python -u augf.py --dsn ag --samplecnt 16 --max_aug_times 1 --aug generate  --genft lambda --filter nli --genm gpt --abundance 2

