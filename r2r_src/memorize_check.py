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
aug_path = 'tasks/R2R/data/aug_paths_unseenvalid.json'
out_path_seen = 'snap/bt_pre_explore_fedavg_glr2_sr06_12_12_clip_vit_seen_enc_rerun_4/best_results.json'
out_path_unseen = 'snap/bt_pre_explore_fedavg_glr2_sr06_12_12_clip_vit_unseen_enc_rerun_4/best_results.json'
out_path_env = 'snap/env_based_pre_explore_aug_fed_12_12_clip_vit_rerun3/best_results.json'
instruction_path = 'tasks/R2R/data/R2R_val_unseen.json'

with open(out_path_seen) as f:
    trajectory_data_seen = json.load(f)

with open(out_path_unseen) as f:
    trajectory_data_unseen = json.load(f)
with open(out_path_env) as f:
    trajectory_data_env = json.load(f)
with open(instruction_path) as f:
    instruction_data = json.load(f)

instr_id2txt = {
    ('%s_%d' % (d['path_id'], n)): txt for d in instruction_data for n, txt in enumerate(d['instructions'])}
instr_id2scan = {
    ('%s_%d' % (d['path_id'], n)): d['scan'] for d in instruction_data for n, txt in enumerate(d['instructions'])}
ev = Evaluation(['val_unseen'], featurized_scans, tok)
fail_idx_seen = ev.fail_idx(out_path_seen)
success_seen = ev.success_case(out_path_seen)
fail_idx_unseen = ev.fail_idx(out_path_unseen)
success_unseen = ev.success_case(out_path_unseen)
fail_idx_env = ev.fail_idx(out_path_env)

diff_list_seen = []
diff_list_unseen = []
diff_list_env = []
for idx in fail_idx_env:
    trajectory = trajectory_data_env[idx]
    instr_id = trajectory['instr_id']
    for idx_ in success_unseen:
        trajectory = trajectory_data_unseen[idx_]
        instr_id_ = trajectory['instr_id']
        if instr_id_ == instr_id:
            diff_list_seen.append(idx_)
            diff_list_unseen.append(idx)
            print(idx, idx_, instr_id_)

print(diff_list_seen)
print(diff_list_unseen)

"""
test_list = [i for i in fail_idx_unseen if i in fail_idx_seen]
print(len(test_list))
with open(out_path_seen) as f:
    results = json.load(f)

results_select = []
for i, item in enumerate(results):
    if i in test_list:
        results_select.append(item)
score_summ, _ = ev.score(out_path_seen)
print(score_summ['ndtw'], score_summ['cls'], score_summ['lengths'], score_summ['success_rate'])

with open(out_path_unseen) as f:
    results = json.load(f)

results_select = []
for i, item in enumerate(results):
    if i in test_list:
        results_select.append(item)
score_summ, _ = ev.score(out_path_unseen)
print(score_summ['ndtw'], score_summ['cls'], score_summ['lengths'], score_summ['success_rate'])
"""

#ev = Evaluation([aug_path], featurized_scans, tok)
#ndtw, cls = ev.fidelity(out_path, fail_idx)
#print(ndtw, cls)

# failed generated path and augmented path seen 0.04380657573029632 0.9250854368066537 unseen 0.0442390843373053 0.9350006808130096
# oracle path seen 0.02663183617240099 0.9483933687037325 unseen 0.025387543489317668 0.9518531938032295

# unseen fail 0.2968361877424197 0.44209445191522967  seen fail 0.2792273893936666 0.42509671138982175
# unseen ora fail 0.49324485197649176 0.6215081702457874 seen 0.49050559469685934 0.6125770958529632
# unseen non-ora fail 0.2037788331996519 0.3570892101066216 seen 0.18779186356917538 0.3439602424835042

# unseen non-ora fail pl unseen 10.808782004709162 seen 11.704141895762112
# unseen ora fail pl unseen 13.34781413580686 seen 12.637152236981127
# unseen fail pl unseen 11.625028465062048 seen 11.98596026862299

# same cases pl
# unseen 12.906475372845991 seen 11.697208662860115
#

