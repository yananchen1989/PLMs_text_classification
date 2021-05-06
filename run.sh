
# baseline 
python -u baseline_classifier.py --ds uci --aug no --samplecnt 32 --batch_size 8

# back-translation
python -u baseline_classifier.py --ds uci --aug translate --samplecnt 32 --batch_size 8 --lang de
python -u baseline_classifier.py --ds uci --aug translate --samplecnt 32 --batch_size 8 --lang fr
python -u baseline_classifier.py --ds uci --aug translate --samplecnt 32 --batch_size 8 --lang zh

# fillin masks
python -u baseline_classifier.py --ds uci --aug fillin  --ner_set 1 --samplecnt 32 --batch_size 8
python -u baseline_classifier.py --ds uci --aug fillin  --ner_set 0 --samplecnt 32 --batch_size 8

# generate samples
python -u baseline_classifier.py --ds uci --aug generate --generate_m ctrl --samplecnt 32 --batch_size 8
python -u baseline_classifier.py --ds uci --aug generate --generate_m gpt2 --samplecnt 32 --batch_size 8