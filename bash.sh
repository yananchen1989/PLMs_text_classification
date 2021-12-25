


############## ft gpt 
nohup python -u ft_gpt2.py --genm gpt2 --num_train_epochs 4 --ccsample 1 --ft_pattern ep --gpu 0 --batch_size 8 \
 > ft.gpt2.ep.log &

# ft t5

CUDA_VISIBLE_DEVICES=7 nohup ./envcbert/bin/python -u ft_t5.py --ft_pattern pp --num_workers 4 \
   --maxlen 256 --ccsample 1 --ftepochs 4 --batch_size 4 > ft.t5.ep.log & 
nohup python -u ft.py --genm t5 --dsn_summary xsum --num_train_epochs 3 --ft_pattern summary --ccsample 0.1 --gpu 0




python -u augf.py --dsn ag --samplecnt 8 --max_aug_times 1 --aug generate \
                     --genft no  --genm gpt --filter nlinsp --seed 1 \
                     --testvalid test --testbed 0 \
                     --candidates 256 --test_beams 32  \
                      --gpu 6,7


python -u augf.py --dsn ag --samplecnt 8 --max_aug_times 1 --aug generate \
                     --genft no  --genm t5 --filter nlinsp --seed 0 \
                     --testvalid test --testbed 0 \
                     --candidates 256 --test_beams 32  \
                      --gpu 0,1 



python -u augf.py --dsn ag --samplecnt 8 --max_aug_times 1 --aug generate \
                  --genft lambda  --genm gpt --filter clsembed --seed 0 \
                  --testvalid test --candidates 128 --gpu 6,7 --epochs 2 --verbose 1




# sdu  generate dvrl  ==> log_arxiv_testearlystop

for gpu in 0,1 2,3 4,5 6,7
do
   nohup bash run.sh  ${gpu} &
done


for gpu in 0,1 2,3 
do
nohup bash run_cbert.sh ${gpu} & 
done




seed=$RANDOM
for dsn in uci ag nyt 
do
nohup envcbert/bin/python -u augf.py --dsn ${dsn} --samplecnt 128 --aug cbert \
      --max_aug_times 1 --seed ${seed} --testvalid test  \
 > ./log_baselines/${dsn}.cbert.128.1.${seed}.log 2>&1 &
done




wget http://nlp.stanford.edu/data/glove.6B.zip
wget http://nlp.stanford.edu/data/glove.42B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.twitter.27B.zip
wget http://nlp.stanford.edu/data/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.twitter.27B.zip


python -m gensim.scripts.glove2word2vec -i ./resource/glove.6B.300d.txt -o ./resource/glove.6B.300d.word2vec.txt
python -m gensim.scripts.glove2word2vec -i ./resource/glove.42B.300d.txt -o ./resource/glove.42B.300d.word2vec.txt;
python -m gensim.scripts.glove2word2vec -i ./resource/glove.840B.300d.txt -o ./resource/glove.840B.300d.word2vec.txt;
python -m gensim.scripts.glove2word2vec -i ./resource/glove.twitter.27B.200d.txt -o ./resource/glove.twitter.27B.200d.word2vec.txt



nohup python -u zsclassifier.py --dsn agp --gpu 7  --mode train  --embed_cut 0.15  --upper 0.85 --lower 0.15 \
            > ./log_zsl/zsl.gen.agp.15.85.15.log & 



nohup python -u zsclassifier.py --dsn uci  --embedm google --mode test     --gpu 0   > zsl.test.uci.w2v.log &
nohup python -u zsclassifier.py --dsn ag   --embedm google  --mode test    --gpu 1   > zsl.test.ag.w2v.log &
nohup python -u zsclassifier.py --dsn yahoo  --embedm google  --mode test --gpu 2   > zsl.test.yahoo.w2v.log &


nohup python -u zsclassifier.py --dsn uci  --embedm glove6b --mode test  --gpu 3   > zsl.test.uci.w2v.glove6b.log &
nohup python -u zsclassifier.py --dsn uci  --embedm glove27b --mode test  --gpu 4   > zsl.test.uci.w2v.glove27b.log &

nohup python -u zsclassifier.py --dsn yahoo  --embedm glove6b --mode test  --gpu 5   > zsl.test.yahoo.w2v.glove6b.log &
nohup python -u zsclassifier.py --dsn yahoo  --embedm glove27b --mode test  --gpu 6   > zsl.test.yahoo.w2v.glove27b.log &




nohup python -u zsclassifier.py --dsn yahoo  --embedm google --mode test  --gpu 0


############################################################################################################################################
ps aux|grep "zsclassifier.py"|grep -v grep | awk '{print $2}'|xargs kill -9
ps aux|grep "run.sh"|grep -v grep | awk '{print $2}'|xargs kill -9
ps aux|grep "run_cbert.sh"|grep -v grep | awk '{print $2}'|xargs kill -9

ps aux|grep "dvrl_iter"|grep -v grep | awk '{print $2}'|xargs kill -9
ps aux|grep "dvrl_iter"|grep "3762"|grep -v grep | awk '{print $2}'|xargs kill -9


ps aux --sort=start_time

tf_upgrade_v2 --infile main_data_valuation.py --outfile main_data_valuation_v2.py

alias wn='watch -n 1 nvidia-smi'

# 解压
unzip -o -d /home/sunny myfile.zip
tar -zxvf torch_ds.tar.gz;tar -zxvf cache_cbert.tar.gz;




for file in resource torch_ds cache_cbert
do
	#tar -zcvf ${file}.tar.gz ${file}
	tar -xvzf ${file}.tar.gz  -C ${file}
done


zip -r myfile.zip ./myfile

cat *.pgn | grep "Result" | sort | uniq -c
conda config --set auto_activate_base true