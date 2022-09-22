name=bt_pre_explore_fedavg_glr2_enc_2_2_le3_sf07
# aug: the augmented paths, only the paths are used (not the insts)
# speaker: load the speaker from
# load: load the agent from
flag="--attn soft --train auglistener --selfTrain
      --aug tasks/R2R/data/aug_paths_unseenvalid.json
      --load snap/agent_bt_fedavg-glr2-2-le3/state_dict/best_val_unseen
      --speaker snap/speaker-fed-lr55_glr15_decrease85/state_dict/best_val_unseen_bleu
      --features img_features/ResNet-152-imagenet.tsv
      --angleFeatSize 128
      --accumulateGrad
      --featdropout 0.4
      --pre_explore True
      --if_fed True
      --fed_alg fedavg
      --subout max --optim rms --lr 1e-4 --iters 320000 --maxAction 35
      --global_lr 2
      --comm_round 910
      --n_parties 11
      --sample_fraction 0.7
      --local_epoches 1
      --enc_only True
      "
mkdir -p snap/$name

CUDA_VISIBLE_DEVICES=0 python3 r2r_src/train.py $flag --name $name

#--sample_fraction 0.3
#--unseen_only True
#--enc_only True
#--speaker snap/speaker-fed-lr25/state_dict/best_val_unseen_bleu
# --load snap/agent_bt_fedavg-glr2-3_26/state_dict/best_val_unseen
# bt_pre_explore_fedavg_glr2_aug_sr1_seen_2_3_sr06_rerun