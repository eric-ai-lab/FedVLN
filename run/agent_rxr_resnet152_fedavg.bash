name=agent_rxr_en_resnet_fedavg_new_glr2_le3
flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --language en
      --maxInput 160
      --features img_features/ResNet-152-imagenet.tsv
      --load snap/agent_rxr_en_resnet_fedavg_new_glr2_le3/state_dict/latest_dict
      --feature_size 2048
      --feedback sample
      --mlWeight 0.4
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 350000 --maxAction 35
      --if_fed True
      --fed_alg fedavg
      --global_lr 2
      --comm_round 1400
      --local_epoches 3
      --n_parties 60"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=2 python3 rxr_src/train.py $flag --name $name

# --load snap/agent_rxr_en_resnet_fedavg_new_glr2/state_dict/latest_dict