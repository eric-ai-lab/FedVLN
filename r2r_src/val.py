import copy

import torch

import os
import time
import json
import numpy as np
from collections import defaultdict
from speaker import Speaker

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch, R2RBatchScan
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args

import warnings
warnings.filterwarnings("ignore")


from tensorboardX import SummaryWriter
import wandb
import math
import random

log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

feedback_method = args.feedback # teacher or sample


def eval_model(val_envs, listner, loss_str, best_val):
    for env_name, (env, evaluator) in val_envs.items():
        listner.env = env
        # Get validation distance from goal under test evaluation conditions
        listner.test(use_dropout=False, feedback='argmax', iters=None)
        result = listner.get_results()
        score_summary, _ = evaluator.score(result)
        loss_str += ", %s " % env_name
        for metric, val in score_summary.items():
            if metric in ['success_rate']:
                if env_name in best_val:
                    if val > best_val[env_name]['accu']:
                        best_val[env_name]['accu'] = val
                        best_val[env_name]['update'] = True
                    elif (val == best_val[env_name]['accu']) and (
                            score_summary['spl'] > best_val[env_name]['spl']):
                        best_val[env_name]['accu'] = val
                        best_val[env_name]['update'] = True
            # writer.add_scalar("spl/%s" % env_name, val, iter)
            loss_str += ', %s: %.4f' % (metric, val)
        wandb.log({"sr/%s" % env_name: score_summary['success_rate']})
        wandb.log({"tl/%s" % env_name: score_summary['lengths']})
        wandb.log({"ne/%s" % env_name: score_summary['nav_error']})
        wandb.log({"oe/%s" % env_name: score_summary['oracle_error']})
        wandb.log({"osr/%s" % env_name: score_summary['oracle_rate']})
        wandb.log({"spl/%s" % env_name: score_summary['spl']})
        # print(loss_str)

    return loss_str, best_val


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


setup()
# Create a batch training environment that will also preprocess text
vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

feat_dict = read_img_features(args.features)

featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

if args.if_fed:
    train_env = R2RBatchScan(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
else:
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)

from collections import OrderedDict

val_env_names = ['val_unseen']

val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

listner_global = Seq2SeqAgent(train_env, os.path.join('snap', args.name, 'best_results.json'), tok, args.maxAction)



