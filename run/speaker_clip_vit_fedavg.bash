name=speaker_clip_vit_fedavg
flag="--attn soft --angleFeatSize 128
      --train speaker
      --features img_features/CLIP-ViT-B-32-views.tsv
      --feature_size 512
      --subout max --dropout 0.6 --optim adam --lr 3e-5 --iters 80000 --maxAction 35
      --if_fed True
      --comm_round 365
      --global_lr 1"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=3 python3 r2r_src/train.py $flag --name $name

