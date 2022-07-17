name=agent_bt_pre_explore_unseen_only_clip_vit_cent
# aug: the augmented paths, only the paths are used (not the insts)
# speaker: load the speaker from
# load: load the agent from
flag="--attn soft --train auglistener --selfTrain
      --aug tasks/R2R/data/aug_paths_unseenvalid.json
      --load snap/agent_bt_clip_vit_fedavg_12_12/state_dict/best_val_unseen
      --speaker snap/speaker_clip_vit_fedavg/state_dict/best_val_unseen_bleu
      --features img_features/CLIP-ViT-B-32-views.tsv
      --angleFeatSize 128
      --accumulateGrad
      --featdropout 0.4
      --pre_explore True
      --subout max --optim rms --lr 1e-4 --iters 210000 --maxAction 35"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=1 python3 r2r_src/train.py $flag --name $name

#--speaker snap/speaker/state_dict/best_val_unseen_bleu
 #      --load snap/agent/state_dict/best_val_unseen