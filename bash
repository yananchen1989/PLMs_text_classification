




CUDA_VISIBLE_DEVICES=0 nohup  python -u  gangpt.py --dsn ag --train_uni 0 --iter 30 > ag_log & 
CUDA_VISIBLE_DEVICES=1 nohup  python -u  gangpt.py --dsn stsa --train_uni 1 --samplecnt 100 --iter 10 > stsa_log & 
CUDA_VISIBLE_DEVICES=2 nohup  python -u  gangpt.py --dsn yahoo --train_uni 1 --iter 10 > yahoo_log & 
CUDA_VISIBLE_DEVICES=3 nohup  python -u  gangpt.py --dsn dbpedia --train_uni 1 --iter 10 > dbpedia_log & 



dsn=${1}
python -u baseline_classifier.py --dsn ${dsn} --aug eda --ite 3
python -u baseline_classifier.py --dsn ${dsn} --aug bt --ite 3
python -u baseline_classifier.py --dsn ${dsn} --aug fillin --ite 3
python -u baseline_classifier.py --dsn ${dsn} --aug generate --check no --ite 3

python -u baseline_classifier.py --dsn ${1}  --aug generate --check enc --ite 3
python -u baseline_classifier.py --dsn ${1}  --aug generate --check nli --ite 3



























python -c "import tensorflow;print(tensorflow.__version__)"
python -c "import torch;print(torch.__version__)"
python -c "import transformers;print(transformers.__version__)"
python -c "import flair;print(flair.__version__)"
python -c "import gensim;print(gensim.__version__)"
python -c "import sklearn;print(sklearn.__version__)"
python -c "import chainer;print(chainer.__version__)"

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








