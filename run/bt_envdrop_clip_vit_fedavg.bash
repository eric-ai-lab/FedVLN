name=agent_bt_clip_vit_fedavg_3_22_97_le3

# aug: the augmented paths, only the paths are used (not the insts)
# speaker: load the speaker from
# load: load the agent from

flag="--attn soft --train auglistener --selfTrain
      --aug tasks/R2R/data/aug_paths.json
      --speaker snap/speaker_clip_vit_fedavg_lr55_glr2_decrease8/state_dict/best_val_unseen_bleu
      --load snap/agent_clip_vit_fedavg_new_glr22_decrease95/state_dict/best_val_unseen
      --angleFeatSize 128
      --features img_features/CLIP-ViT-B-32-views.tsv
      --feature_size 512
      --accumulateGrad
      --featdropout 0.4
      --subout max --optim rms --lr 1e-4 --iters 300000 --maxAction 35
      --if_fed True
      --fed_alg fedavg
      --global_lr 2.2
      --comm_round 910
      --n_parties 60
      --local_epoches 3
      --glr_decrease_rate 0.97
      "
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python3 r2r_src/train.py $flag --name $name