dsn=ucii
samplecnt=32
batch_size=8
# baseline 
nohup python -u baseline_classifier.py --ds ${dsn} --aug no --samplecnt ${samplecnt} --batch_size ${batch_size}  > aug_${dsn}_no.log & 

# back-translation
nohup python -u baseline_classifier.py --ds ${dsn} --aug translate --samplecnt ${samplecnt}  --batch_size ${batch_size} --lang de > aug_${dsn}_translate_de.log &
nohup python -u baseline_classifier.py --ds ${dsn} --aug translate --samplecnt ${samplecnt}  --batch_size ${batch_size}  --lang fr > aug_${dsn}_translate_fr.log &
nohup python -u baseline_classifier.py --ds ${dsn} --aug translate --samplecnt ${samplecnt}  --batch_size ${batch_size}  --lang zh > aug_${dsn}_translate_zh.log &

# fillin masks
nohup python -u baseline_classifier.py --ds ${dsn} --aug fillin  --ner_set 1 --samplecnt ${samplecnt}  --batch_size ${batch_size}  > aug_${dsn}_fillin_ner1.log &
nohup python -u baseline_classifier.py --ds ${dsn} --aug fillin  --ner_set 0 --samplecnt ${samplecnt}  --batch_size ${batch_size}  > aug_${dsn}_fillin_ner0.log &

# generate samples
nohup python -u baseline_classifier.py --ds ${dsn} --aug generate --generate_m ctrl --samplecnt ${samplecnt}  --batch_size ${batch_size}  > aug_${dsn}_generate_ctrl.log &
nohup python -u baseline_classifier.py --ds ${dsn} --aug generate --generate_m gpt2 --samplecnt ${samplecnt}  --batch_size ${batch_size}  > aug_${dsn}_generate_gpt2.log &