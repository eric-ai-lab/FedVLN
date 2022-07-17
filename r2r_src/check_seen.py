import time
import json
import numpy as np
from collections import defaultdict
from speaker import Speaker
import torch
import os
from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features, load_datasets
import utils
from env import R2RBatch, R2RBatchScan
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args
import nltk
from nltk import word_tokenize
import collections
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

args.features = 'img_features/CLIP-ViT-B-32-views.tsv'

# check unseen enc failure cases that pass the success area,
# the overlap(ndtw and cls) of generated path and augmented path
vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)
feat_dict = read_img_features(args.features)
featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
out_path_fed = 'snap/agent_bt_clip_vit_fedavg_12_12/best_results.json'
out_path_clipvil = 'snap/agent_bt_clip_vit/best_results.json'
instruction_path = 'tasks/R2R/data/R2R_val_unseen.json'

with open(out_path_clipvil) as f:
    trajectory_data_clipvil = json.load(f)

with open(out_path_fed) as f:
    trajectory_data_fed = json.load(f)
with open(instruction_path) as f:
    instruction_data = json.load(f)

instr_id2txt = {
    ('%s_%d' % (d['path_id'], n)): txt for d in instruction_data for n, txt in enumerate(d['instructions'])}
instr_id2scan = {
    ('%s_%d' % (d['path_id'], n)): d['scan'] for d in instruction_data for n, txt in enumerate(d['instructions'])}
ev = Evaluation(['val_seen'], featurized_scans, tok)
fail_idx_fed = ev.fail_idx(out_path_fed)
success_fed = ev.success_case(out_path_fed)
fail_idx_clipvil = ev.fail_idx(out_path_clipvil)
success_clipvil = ev.success_case(out_path_clipvil)
print(len(fail_idx_fed)+len(success_fed))
diff_list_fed = []
diff_list_clipcil = []
for idx in fail_idx_fed:
    trajectory = trajectory_data_fed[idx]
    instr_id = trajectory['instr_id']
    for idx_ in success_clipvil:
        trajectory = trajectory_data_clipvil[idx_]
        instr_id_ = trajectory['instr_id']
        if instr_id_ == instr_id:
            diff_list_fed.append(idx_)
            diff_list_clipcil.append(idx)
            #print(idx, idx_, instr_id_)

print(diff_list_fed)

diff_list_fed = []
diff_list_clipcil = []
for idx in fail_idx_clipvil:
    trajectory = trajectory_data_clipvil[idx]
    instr_id = trajectory['instr_id']
    for idx_ in success_fed:
        trajectory = trajectory_data_fed[idx_]
        instr_id_ = trajectory['instr_id']
        if instr_id_ == instr_id:
            diff_list_fed.append(idx_)
            diff_list_clipcil.append(idx)
            #print(idx, idx_, instr_id_)

print(diff_list_fed)