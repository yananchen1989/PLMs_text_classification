dsn=${1}
max_aug_times=${2} # parameter R
samplecnt=${3} # simulate a low-data regime, where 32 samples per category are selected as anchor data
candidates=${4} #parameter B 
test_beams=${5}
gpu=${6}
while true
do
	seed=$RANDOM
	#seed=$(date +"%T")
	python -u dpfuture.py --dsn ${dsn} --batch_size 32 --samplecnt ${samplecnt} --cls_score_thres 0.8 \
	--candidates ${candidates} --test_beams ${test_beams} --max_aug_times ${max_aug_times} \
	--seed ${seed} --gpu ${gpu}  \
	> ./log_dpfuture/dpfuture.${dsn}.samplecnt_${samplecnt}.max_aug_times_${max_aug_times}.candidates_${candidates}.test_beams_${test_beams}.${seed}.log 2>&1
done



