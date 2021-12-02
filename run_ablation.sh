while true
do
   seed=$RANDOM
   for samplecnt in 32 64 128
   do
      for dsn in ag uci  #yelp2 amazon2
      do 
        python -u ablation_ft.py --samplecnt ${samplecnt} --dsn ${dsn} > ablation_ft.${samplecnt}.${dsn}.${seed}.log 2>&1
      done
   done
done