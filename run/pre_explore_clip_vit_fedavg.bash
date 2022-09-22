name=bt_pre_explore_clip_vit_fedavg_22_22_95_97_unseen_enc_sf07
# aug: the augmented paths, only the paths are used (not the insts)
# speaker: load the speaker from
# load: load the agent from
flag="--attn soft --train auglistener --selfTrain
      --aug tasks/R2R/data/aug_paths_unseenvalid.json
      --load snap/agent_bt_clip_vit_fedavg_22_22_95_97/state_dict/best_val_unseen
      --speaker snap/speaker_clip_vit_fedavg_lr55_glr2_decrease8/state_dict/best_val_unseen_bleu
      --features img_features/CLIP-ViT-B-32-views.tsv
      --angleFeatSize 128
      --feature_size 512
      --accumulateGrad
      --featdropout 0.4
      --pre_explore True
      --if_fed True
      --fed_alg fedavg
      --subout max --optim rms --lr 1e-4 --iters 250000 --maxAction 35
      --global_lr 2
      --comm_round 910
      --n_parties 11
      --sample_fraction 0.7
      --local_epoches 1
      --unseen_only True
      --enc_only True
      "
mkdir -p snap/$name

CUDA_VISIBLE_DEVICES=3 python3 r2r_src/train.py $flag --name $name

# --unseen_only True