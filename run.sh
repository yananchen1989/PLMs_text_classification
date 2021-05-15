dsn=ag
samplecnt=32
batch_size=8

# back-translation
#for lang in de fr zh 
#do
#python -u baseline_classifier.py --ds ${dsn} --aug translate --samplecnt ${samplecnt}  --batch_size ${batch_size} --lang ${lang} 
#done


# fillin masks
for ner in 0 1
do
python -u baseline_classifier.py --ds ${dsn} --aug fillin --ner_set ${ner} --samplecnt ${samplecnt}  --batch_size ${batch_size} 
done


# generate samples
for beam in 1 2 4 8 16 32 64
do
#python -u baseline_classifier.py --ds ${dsn} --aug generate --generate_m ctrl --samplecnt ${samplecnt}  --batch_size ${batch_size}  > aug_${dsn}_generate_ctrl.log 
python -u baseline_classifier.py --ds ${dsn} --aug generate --generate_m gpt2 --beams ${beam} --samplecnt ${samplecnt}  --batch_size ${batch_size}   
done

# baseline 
python -u baseline_classifier.py --ds ${dsn} --aug no --samplecnt ${samplecnt} --batch_size ${batch_size} 
