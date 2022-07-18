name=agent_rxr_en_resnet_fedavg_new_glr12
flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --language en
      --maxInput 160
      --features img_features/ResNet-152-imagenet.tsv
      --feature_size 512
      --feedback sample
      --mlWeight 0.4
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 150000 --maxAction 35
      --if_fed True
      --fed_alg fedavg
      --global_lr 12
      --comm_round 910
      --n_parties 60"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python3 rxr_src/train.py $flag --name $name