name=agent-fedavg-glr6
flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 80000 --maxAction 35
      --load snap/agent-fedavg-glr6/state_dict/latest_dict
      --if_fed True
      --comm_round 2000
      --sample_fraction 0.2
      --global_lr 6
      --local_epoches 5
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python3 r2r_src/train.py $flag --name $name
