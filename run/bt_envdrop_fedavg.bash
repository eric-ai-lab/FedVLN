name=agent_bt_fedavg-glr22-2-le3
# aug: the augmented paths, only the paths are used (not the insts)
# speaker: load the speaker from
# load: load the agent from
flag="--attn soft --train auglistener --selfTrain
      --aug tasks/R2R/data/aug_paths.json
      --speaker snap/speaker-fed-lr55_glr15_decrease85/state_dict/best_val_unseen_bleu
      --load snap/agent_clip_vit_fedavg_new_glr22_decrease95/state_dict/best_val_unseen
      --angleFeatSize 128
      --accumulateGrad
      --featdropout 0.4
      --subout max --optim rms --lr 1e-4 --iters 250000 --maxAction 35
      --if_fed True
      --features img_features/ResNet-152-imagenet.tsv
      --feature_size 2048
      --comm_round 910
      --global_lr 2"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python3 r2r_src/train.py $flag --name $name
