name=agent_rxr_en_maxinput160_ml04_fedavg_new_glr6
flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --language en
      --maxInput 160
      --features img_features/ResNet-152-imagenet.tsv
      --load snap/agent_rxr_en_maxinput160_ml04_fedavg_new_glr6/state_dict/latest_dict
      --feature_size 2048
      --feedback sample
      --mlWeight 0.4
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 150000  --maxAction 35
      --if_fed True
      --fed_alg fedavg
      --global_lr 6
      --comm_round 910
      --n_parties 60"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=3 python3 rxr_src/train.py $flag --name $name