name=bt_pre_explore_fedavg_glr2_sr06_12_12_clip_vit_save_global
# aug: the augmented paths, only the paths are used (not the insts)
# speaker: load the speaker from
# load: load the agent from
flag="--attn soft --train auglistener --selfTrain
      --load snap/agent_bt_clip_vit_fedavg_12_12/state_dict/best_val_unseen
      --speaker snap/speaker_clip_vit_fedavg/state_dict/best_val_unseen_bleu
      --features img_features/CLIP-ViT-B-32-views.tsv
      --angleFeatSize 128
      --feature_size 512
      --accumulateGrad
      --featdropout 0.4
      --pre_explore True
      --if_fed True
      --fed_alg fedavg
      --subout max --optim rms --lr 1e-4 --iters 200000 --maxAction 35
      --global_lr 2
      --comm_round 910
      --n_parties 72
      --sample_fraction 0.6
      --local_epoches 1
      "
mkdir -p snap/$name

CUDA_VISIBLE_DEVICES=1 python3 r2r_src/eval_checkpoint.py $flag --name $name
# --aug tasks/R2R/data/aug_paths_unseenvalid.json