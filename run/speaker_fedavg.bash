name=speaker-fed-lr55_glr16_decrease95
flag="--attn soft --angleFeatSize 128
      --train speaker
      --features img_features/ResNet-152-imagenet.tsv
      --subout max --dropout 0.6 --optim adam --lr 5e-5 --iters 80000 --maxAction 35
      --if_fed True
      --comm_round 365
      --global_lr 1.6
      --local_epoches 5
      --glr_decrease_rate 0.95
"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=3 python3 r2r_src/train.py $flag --name $name
