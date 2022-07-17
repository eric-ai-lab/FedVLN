import torch

import os
import time
import json
import numpy as np
from collections import defaultdict
from speaker import Speaker
import collections
from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
import utils
from env import R2RBatch, R2RBatchScan
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import OrderedDict
import matplotlib.pyplot as plt

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
val_env_names = ['val_unseen', 'val_seen']
vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

feat_dict = read_img_features(args.features)
featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

train_env = R2RBatchScan(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
speaker = Speaker(train_env, listner, tok)
speaker.load(args.speaker)
for env_name, (env, evaluator) in val_envs.items():
    if 'train' in env_name or 'unseen' in env_name:  # Ignore the large training set for the efficiency
        continue
    generated_instr = []
    gt_instr = []
    print("............ Evaluating %s ............." % env_name)
    speaker.env = env
    path2inst, loss, word_accu, sent_accu = speaker.valid()
    for key in path2inst:
        generated_instr.append(tok.decode_sentence(path2inst[key]))
        for gt in evaluator.gt[str(key)]['instructions']:
            gt_instr.append(gt)

    length_generated = 0
    length_gt = 0
    num_sent_gen = 0
    num_sent_gt = 0
    var_gt = 0
    var_gen = 0
    lengths_gt = []
    lengths_gen = []
    for instr in gt_instr:
        num_sent_gt += len(sent_tokenize(instr))
        lengths_gt.append(len(sent_tokenize(instr)))
    num_sent_gt /= len(gt_instr)

    for instr in generated_instr:
        num_sent_gen += len(sent_tokenize(instr))
        lengths_gen.append(len(sent_tokenize(instr)))
    num_sent_gen /= len(generated_instr)
    print(num_sent_gt, num_sent_gen)
    print(np.var(lengths_gt), np.var(lengths_gen))

    lengths_gt = []
    lengths_gen = []
    for instr in gt_instr:
        length_gt += len(word_tokenize(instr))
        lengths_gt.append(len(word_tokenize(instr)))
    length_gt /= len(gt_instr)

    for instr in generated_instr:
        length_generated += len(word_tokenize(instr))
        lengths_gen.append(len(word_tokenize(instr)))
    length_generated /= len(generated_instr)
    print(length_gt, length_generated)
    print(np.var(lengths_gt), np.var(lengths_gen))

    # word frequency
    verbs_gt = []
    verbs_gen = []
    for instr in gt_instr:
        tag = nltk.pos_tag(word_tokenize(instr))
        for word in tag:
            if word[1].startswith('V') and word[0]!='left':
                verbs_gt.append(word[0].lower())

    for instr in generated_instr:
        tag = nltk.pos_tag(word_tokenize(instr))
        for word in tag:
            if word[1].startswith('V') and word[0]!='left':
                verbs_gen.append(word[0].lower())
    result_gen = collections.Counter(verbs_gen)
    result_gt = collections.Counter(verbs_gt)
    print(collections.Counter(verbs_gt), len(result_gt))
    print(collections.Counter(verbs_gen), len(result_gen))

    lt = []
    labels = []
    label_order = []
    if_other = 0
    others = 0
    for key in result_gt:
        result = result_gt[key]/len(verbs_gt)
        if result_gt[key] > 70:
            labels.append(key)
            lt.append(result)
        else:
            others += result_gt[key]/len(verbs_gt)
    sorted_id = sorted(range(len(lt)), key=lambda k: lt[k], reverse=False)
    label_order = [labels[i] for i in sorted_id]
    label_order.append('other')
    lt.sort()
    lt.append(others)
    print(lt)
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    colors = ['bisque', 'skyblue', 'lightcoral', 'tan', 'pink', 'lightgray', 'lightgreen', 'lightsteelblue']
    patches, l_text, p_text = plt.pie(x=lt,autopct='%1.1f%%',labels=label_order, colors=colors)
    for t in l_text:
        t.set_size(15)
    for t in p_text:
        t.set_size(13)
    plt.title('GT verb frequency', fontsize=16)
    plt.savefig('figures/gt verb frequency.pdf')
    plt.close(0)

    lt = []
    labels = []
    others = 0
    for key in result_gen:
        result = result_gen[key]/len(verbs_gen)
        if result_gen[key] > 50:
            labels.append(key)
            lt.append(result)
        else:
            others += result_gen[key]/len(verbs_gen)
    sorted_id = sorted(range(len(lt)), key=lambda k: lt[k], reverse=False)
    label_order = [labels[i] for i in sorted_id]
    label_order.append('other')
    lt.sort()
    lt.append(others)
    plt.figure()
    colors = ['bisque', 'skyblue', 'lightcoral', 'tan', 'pink', ]
    patches, l_text, p_text = plt.pie(x=lt,autopct='%1.1f%%',labels=label_order, colors=colors)
    for t in l_text:
        t.set_size(15)
    for t in p_text:
        t.set_size(13)
    plt.title('Pseudo verb frequency',fontsize=16)
    plt.savefig('figures/pseudo verb frequency.pdf')