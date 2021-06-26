




CUDA_VISIBLE_DEVICES=0 nohup python -u baseline_classifier.py --dsn ag --aug bt & 

CUDA_VISIBLE_DEVICES=1 nohup  python -u baseline_classifier.py --dsn ag --aug generate --check no  & 



while true
do
for dsn in ag yahoo stsa dbpedia
do 
CUDA_VISIBLE_DEVICES=${1} python -u baseline_classifier.py --dsn ${dsn} --aug eda 
CUDA_VISIBLE_DEVICES=${1} python -u baseline_classifier.py --dsn ${dsn} --aug bt 
CUDA_VISIBLE_DEVICES=${1} python -u baseline_classifier.py --dsn ${dsn} --aug fillin 
CUDA_VISIBLE_DEVICES=${1} python -u baseline_classifier.py --dsn ${dsn} --aug generate --check no 
CUDA_VISIBLE_DEVICES=${1} python -u baseline_classifier.py --dsn ${dsn}  --aug generate --check enc 
CUDA_VISIBLE_DEVICES=${1} python -u baseline_classifier.py --dsn ${dsn}  --aug generate --check nli 
done
done 

























pip3 install torch torchvision torchaudio

pip install git+https://github.com/huggingface/transformers

pip install tensorflow==2.5.0;pip install tensorflow_text==2.5.0

pip install tensorflow==2.3.1;pip install tensorflow_text==2.3.0



git config --global --unset https.https://github.com.proxy
git config --global --unset http.https://github.com.proxy



git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .


pip install -r requirements.txt








