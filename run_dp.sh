max_aug_times=${1} # parameter R
samplecnt=${2} # simulate a low-data regime, where 32 samples per category are selected as anchor data
candidates=${3} #parameter B 
test_beams=${4}
gpu=${5}
while true
do
	seed=$RANDOM
	#seed=$(date +"%T")
	for dsn in uci agt
	do
		python -u dpfuture.py --dsn ${dsn} --batch_size 64 --samplecnt ${samplecnt} --cls_score_thres 0.8 \
    	--candidates ${candidates} --test_beams ${test_beams} --max_aug_times ${max_aug_times} \
    	--seed ${seed} --gpu ${gpu}  \
    	> ./log_dpfuture/dpfuture.${dsn}.samplecnt_${samplecnt}.max_aug_times_${max_aug_times}.candidates_${candidates}.test_beams_${test_beams}.${seed}.log 2>&1
    done
done



