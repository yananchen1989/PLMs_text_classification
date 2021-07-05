




CUDA_VISIBLE_DEVICES=0 nohup python -u baseline_classifier.py --dsn ag --aug bt & 

CUDA_VISIBLE_DEVICES=1 nohup  python -u baseline_classifier.py --dsn ag --aug generate --check no  & 


samplecnt=512
while true
do
CUDA_VISIBLE_DEVICES=${1} python -u baseline_classifier.py --dsn ${2} --aug eda --samplecnt ${samplecnt}
CUDA_VISIBLE_DEVICES=${1} python -u baseline_classifier.py --dsn ${2} --aug bt --samplecnt ${samplecnt}
CUDA_VISIBLE_DEVICES=${1} python -u baseline_classifier.py --dsn ${2} --aug fillin --samplecnt ${samplecnt}
CUDA_VISIBLE_DEVICES=${1} python -u baseline_classifier.py --dsn ${2} --aug generate --check no --samplecnt ${samplecnt}
CUDA_VISIBLE_DEVICES=${1} python -u baseline_classifier.py --dsn ${2}  --aug generate --check enc --samplecnt ${samplecnt}
CUDA_VISIBLE_DEVICES=${1} python -u baseline_classifier.py --dsn ${2}  --aug generate --check nli --samplecnt ${samplecnt}
done 




CUDA_VISIBLE_DEVICES=0 python -u baseline_classifier.py --dsn ag --aug eda --max_aug_times 5



pip install https://ghproxy.com/https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0-py3-none-any.whl --no-cache



git add .;git commit -m "update";git push



python gangpt_exter.py --dsn ag --samplecnt 500 --syn raw



scp -r root@10.177.63.136:/root/topic_classification_augmentation/torch_ds ./






pip3 install torch torchvision torchaudio

pip install git+https://github.com/huggingface/transformers

pip install tensorflow==2.5.0;pip install tensorflow_text==2.5.0

pip install tensorflow==2.3.1;pip install tensorflow_text==2.3.0;pip install spacy;ython -m spacy download en_core_web_sm



git config --global --unset https.https://github.com.proxy
git config --global --unset http.https://github.com.proxy



git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .


pip install -r requirements.txt








