name=agent_clip_vit_fedavg_new_glr22_decrease95
flag="--attn soft --train listener
      --featdropout 0.3
      --angleFeatSize 128
      --features img_features/CLIP-ViT-B-32-views.tsv
      --feature_size 512
      --feedback sample
      --mlWeight 0.2
      --subout max --dropout 0.5 --optim rms --lr 1e-4 --iters 150000 --maxAction 35
      --if_fed True
      --fed_alg fedavg
      --global_lr 2.2
      --n_parties 61
      --local_epoches 5
      --glr_decrease_rate 0.95"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=3 python3 r2r_src/train.py $flag --name $name

# --load snap/agent_clip_vit_fedavg_new_glr2/state_dict/latest_dict
