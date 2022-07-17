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
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
warnings.filterwarnings("ignore")

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

args.features = 'img_features/CLIP-ViT-B-32-views.tsv'

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def main():
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(args.features)

    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    ev = Evaluation(['val_unseen'], featurized_scans, tok)
    path = 'snap/bt_pre_explore_fedavg_glr2_sr06_12_12_clip_vit_seen_enc_rerun_4/best_results.json'
    seen_succ = ev.success_case(path)
    # path = 'snap/bt_pre_explore_fedavg_glr2_sr06_12_12_clip_vit_unseen_enc_rerun_4/best_results.json'
    path = 'snap/env_based_pre_explore_aug_fed_12_12_clip_vit_rerun3/best_results.json'
    unseen_succ = ev.success_case(path)
    datasets = load_datasets(['val_unseen'])
    seen_instrs = []
    unseen_instrs = []
    for id in seen_succ:
        path, instr = id.split('_')
        for item in datasets:
            if str(item['path_id']) == path:
                seen_instrs.append(item['instructions'][int(instr)])
    for id in unseen_succ:
        path, instr = id.split('_')
        for item in datasets:
            if str(item['path_id']) == path:
                unseen_instrs.append(item['instructions'][int(instr)])
    seen_len = 0
    for instr in seen_instrs:
        seen_len += len(instr.split())
    seen_len /= len(seen_instrs)

    unseen_len = 0
    for instr in unseen_instrs:
        unseen_len += len(instr.split())
    unseen_len /= len(unseen_instrs)
    print(seen_len, unseen_len)
    # seen 26.03 env based 25.89 unseen 26.01
    diff = []
    for id in unseen_succ:
        if id not in seen_succ:
            diff.append(id)
    diff_instrs = []
    for id in diff:
        path, instr = id.split('_')
        for item in datasets:
            if str(item['path_id']) == path:
                diff_instrs.append(item['instructions'][int(instr)])

    diff_len = 0
    for instr in diff_instrs:
        diff_len += len(instr.split())
    diff_len /= len(diff_instrs)
    print(diff_len)

    diff = []
    for id in seen_succ:
        if id not in unseen_succ:
            diff.append(id)
    # print(diff)
    diff_instrs = []
    for id in diff:
        path, instr = id.split('_')
        for item in datasets:
            if str(item['path_id']) == path:
                diff_instrs.append(item['instructions'][int(instr)])

    diff_len = 0
    for instr in diff_instrs:
        diff_len += len(instr.split())
    diff_len /= len(diff_instrs)
    print(diff_len) # seen vs env based 27.65, 26.72; seen vs unseen 27.20, 27.22
    # ----------------------------------
    verbs_seen = []
    for instr in seen_instrs:
        tag = nltk.pos_tag(word_tokenize(instr))
        for word in tag:
            if word[1].startswith('V') and word[0] != 'left':
                verbs_seen.append(wnl.lemmatize(word[0].lower(), 'v'))
    result_diff = collections.Counter(verbs_seen)
    print(collections.Counter(verbs_seen))

    lt = []
    labels = []
    others = 0
    for key in result_diff:
        result = result_diff[key] / len(verbs_seen)
        if result_diff[key] > 90:
            labels.append(key)
            lt.append(result)
        else:
            others += result_diff[key] / len(verbs_seen)
    sorted_id = sorted(range(len(lt)), key=lambda k: lt[k], reverse=False)
    label_order = [labels[i] for i in sorted_id]
    label_order.append('other')
    lt.sort()
    lt.append(others)
    #colors = ['bisque', 'skyblue', 'lightcoral', 'tan', 'pink', 'lightgray', 'lightgreen', 'lightsteelblue']
    patches, l_text, p_text = plt.pie(x=lt, autopct='%1.1f%%', labels=label_order)
    for t in l_text:
        t.set_size(15)
    for t in p_text:
        t.set_size(13)
    plt.title('seen verb frequency', fontsize=16)
    plt.savefig('figures/seen verb frequency.pdf')
    plt.close(0)
    plt.clf()
    #  --------------
    verbs_unseen = []
    for instr in unseen_instrs:
        tag = nltk.pos_tag(word_tokenize(instr))
        for word in tag:
            if word[1].startswith('V') and word[0] != 'left':
                verbs_unseen.append(wnl.lemmatize(word[0].lower(), 'v'))
    result_diff = collections.Counter(verbs_unseen)
    print(collections.Counter(verbs_unseen))

    lt = []
    labels = []
    others = 0
    for key in result_diff:
        result = result_diff[key] / len(verbs_unseen)
        if result_diff[key] > 90:
            labels.append(key)
            lt.append(result)
        else:
            others += result_diff[key] / len(verbs_unseen)
    sorted_id = sorted(range(len(lt)), key=lambda k: lt[k], reverse=False)
    label_order = [labels[i] for i in sorted_id]
    label_order.append('other')
    lt.sort()
    lt.append(others)

    #colors = ['bisque', 'skyblue', 'lightcoral', 'tan', 'pink', 'lightgray', 'lightgreen', 'lightsteelblue']
    patches, l_text, p_text = plt.pie(x=lt, autopct='%1.1f%%', labels=label_order)
    for t in l_text:
        t.set_size(15)
    for t in p_text:
        t.set_size(13)
    plt.title('env verb frequency', fontsize=16)
    plt.savefig('figures/env verb frequency.pdf')
    plt.close(0)

if __name__ == "__main__":
    main()

