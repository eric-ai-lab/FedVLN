name=agent-fedavg-glr2_le18
flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --features img_features/ResNet-152-imagenet.tsv
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 200000 --maxAction 35
      --if_fed True
      --comm_round 365
      --sample_fraction 0.2
      --global_lr 2
      --local_epoches 18
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python3 r2r_src/train.py $flag --name $name
