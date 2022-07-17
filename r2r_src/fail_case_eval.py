import time
import json
import numpy as np
from collections import defaultdict
from speaker import Speaker
import torch
import os
from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch, R2RBatchScan
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args

import warnings
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
    path = 'snap/bt_pre_explore_fedavg_glr2_sr06_12_12_clip_vit_unseen_enc_rerun_4/'
    score_summ, _ = ev.score_fail(path)
    print(score_summ['lengths'], score_summ['success_rate'])


if __name__ == "__main__":
    main()

