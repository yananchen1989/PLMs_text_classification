


#nohup python -u bert_pair_nsp.py --dsn snli --max_length 64 --gpu 0 > pair.snli.log &
# nohup python -u bert_pair_nsp.py --dsn news --max_length 64 --gpu 0 > pair.news.log &

# nohup python -u bert_pair_nsp.py --dsn yelp2 --max_length 64 --gpu 1 > pair.yelp2.log &
# nohup python -u bert_pair_nsp.py --dsn imdb --max_length 64 --gpu 1 > pair.imdb.log &
# nohup python -u bert_pair_nsp.py --dsn yahoo --max_length 64 --gpu 1 > pair.yahoo.log &


CUDA_VISIBLE_DEVICES=0 nohup  python -u bert_pair_nsp.py --dsn ag --ft 0  --gpu 0 > pair.ppo.ag.ft0.log &
CUDA_VISIBLE_DEVICES=1 nohup  python -u bert_pair_nsp.py --dsn ag --ft 1  --gpu 1 > pair.ppo.ag.ft1.log &



# dev 


nohup python -u aug_ppo_task.py --dsn ag --samplecnt 64  --boostsample_ppo 0  --ppo_batchsize 32 \
	           --load_bert 1 --noeval 1   \
	          --epochs 20  --init_kl_coef 0 --cliprange 0.2   \
	            --gpt_ft 1 --ref_ft 1 --gpu 0 > log_fl0 &


nohup python -u aug_ppo_task.py --dsn ag --samplecnt 64  --boostsample_ppo 0  --ppo_batchsize 32 \
	           --load_bert 1 --noeval 1   \
	          --epochs 20  --init_kl_coef 0.1 --cliprange 0.2   \
	            --gpt_ft 1 --ref_ft 1 --gpu 1 > log_fl1 &



# augf
while true
do
python -u augf.py --dsn ag --samplecnt ${1} --aug eda --gpu ${2}
python -u augf.py --dsn ag --samplecnt ${1} --aug bt --gpu {2}
python -u augf.py --dsn ag --samplecnt ${1} --aug generate --nli_check 1 --gpu ${2}
python -u augf.py --dsn ag --samplecnt ${1} --aug generate --nli_check 0 --gpu ${2}
done




nohup sh ttt.sh 32 former 0 > 32.former.log &
nohup sh ttt.sh 32 albert 0 > 32.albert.log &


nohup sh ttt.sh 64 former 0 > 64.former.log &
nohup sh ttt.sh 64 albert 0 > 64.albert.log &




