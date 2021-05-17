dsn=${1}
echo ${dsn}
samplecnt=${2}
echo ${samplecnt}

# baseline 
python -u baseline_classifier.py --dsn ${dsn} --aug no --samplecnt ${samplecnt} 


# back-translation
for lang in de fr zh 
do
python -u baseline_classifier.py --ds ${dsn} --aug translate --lang ${lang} --samplecnt ${samplecnt} 
done


# fillin masks
for ner in 0 1
do
python -u baseline_classifier.py --dsn ${dsn} --aug fillin --ner_set ${ner} --samplecnt ${samplecnt} 
done


# generate samples
for beam in 1 2 4 8 16 32 64
do
ython -u baseline_classifier.py --dsn ${dsn} --aug generate --generate_m gpt2 --beams ${beam} --samplecnt ${samplecnt} 
done

