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

wandb.init(project="fedvln_", entity="kzhou")

log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

feedback_method = args.feedback # teacher or sample

print(args)

def eval_speaker(speaker, tok, val_envs, writer, idx, best_bleu, best_loss):
    for env_name, (env, evaluator) in val_envs.items():
        if 'train' in env_name:  # Ignore the large training set for the efficiency
            continue
        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2inst, loss, word_accu, sent_accu = speaker.valid()
        path_id = next(iter(path2inst.keys()))
        print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        print("GT: ", evaluator.gt[str(path_id)]['instructions'])
        bleu_score, precisions = evaluator.bleu_score(path2inst)

        # Tensorboard log
        writer.add_scalar("bleu/%s" % (env_name), bleu_score, idx)
        writer.add_scalar("loss/%s" % (env_name), loss, idx)
        writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
        writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
        writer.add_scalar("bleu4/%s" % (env_name), precisions[3], idx)
        # wandb log
        print(bleu_score)
        wandb.log({"bleu/%s" % (env_name): bleu_score})
        wandb.log({"loss/%s" % (env_name): loss})
        wandb.log({"word_accu/%s" % (env_name): word_accu})
        wandb.log({"sent_accu/%s" % (env_name): sent_accu})
        wandb.log({"bleu4/%s" % (env_name): precisions[3]})

        # Save the model according to the bleu score
        if bleu_score > best_bleu[env_name]:
            best_bleu[env_name] = bleu_score
            print('Save the model with %s BEST env bleu %0.4f' % (env_name, bleu_score))
            speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_bleu' % env_name))

        if loss < best_loss[env_name]:
            best_loss[env_name] = loss
            print('Save the model with %s BEST env loss %0.4f' % (env_name, loss))
            speaker.save(idx, os.path.join(log_dir, 'state_dict', 'best_%s_loss' % env_name))

        # Screen print out
        print("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions))


def train_speaker(train_env, tok, n_iters, log_every=500, val_envs={}):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)


    if args.fast_train:
        log_every = 40

    best_bleu = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 1232)
    # args.fed_alg = 'moon'
    print(args.fed_alg)
    if not args.if_fed:
        speaker = Speaker(train_env, listner, tok)
        for idx in range(0, n_iters, log_every):
            interval = min(log_every, n_iters - idx)

            # Train for log_every interval
            speaker.env = train_env
            speaker.train(interval)  # Train interval iters

            print()
            print("Iter: %d" % idx)

            # Evaluation
            eval_speaker(speaker, tok, val_envs, writer, idx, best_bleu, best_loss)
    elif args.fed_alg == 'fedavg':
        glr_schedule_freq = 20000
        glr_schedule_thresh = glr_schedule_freq
        party_list = [i for i in range(args.n_parties)]
        n_party_per_round = int(args.n_parties * args.sample_fraction)
        party_list_rounds = []
        if n_party_per_round != args.n_parties:
            for i in range(args.comm_round):
                party_list_rounds.append(random.sample(party_list, n_party_per_round))
        else:
            for i in range(args.comm_round):
                party_list_rounds.append(party_list)

        local_speaker = Speaker(train_env, listner, tok)
        global_speaker = Speaker(train_env, listner, tok)
        new_global_speaker_encoder_w = copy.deepcopy(global_speaker.encoder.state_dict())
        new_global_speaker_decoder_w = copy.deepcopy(global_speaker.decoder.state_dict())
        iter_ = 0
        # calculate frequency
        total_data_points = sum([len(train_env.data[r]) for r in train_env.data])
        fed_avg_freqs = [len(train_env.data[r]) / total_data_points for r in train_env.scans_list]
        comm_round = math.ceil((args.iters) / (train_env.size() / args.batchSize))
        for round in range(comm_round):
            if iter_ > glr_schedule_thresh:
                args.global_lr *= args.glr_decrease_rate
                glr_schedule_thresh += glr_schedule_freq
            party_list_this_round = party_list_rounds[round]
            global_encoder_w = global_speaker.encoder.state_dict()
            global_decoder_w = global_speaker.decoder.state_dict()
            local_speaker.encoder.load_state_dict(global_encoder_w)
            local_speaker.decoder.load_state_dict(global_decoder_w)
            total_freq_round = 0
            for k in party_list_this_round:
                total_freq_round += fed_avg_freqs[k]
            freq_this_round = [fed_avg_freqs[k] / total_freq_round for k in party_list_this_round]

            for idx, k in enumerate(party_list_this_round):
                train_env.set_current_scan(k)
                interval = math.ceil(args.local_epoches * len(train_env.data[train_env.current_scan]) / args.batchSize)
                iter_ += interval
                # Train for log_every interval
                local_speaker.env = train_env
                local_speaker.train(interval)  # Train interval iters
                client_encoder_para = local_speaker.encoder.state_dict()
                client_decoder_para = local_speaker.decoder.state_dict()
                for key in new_global_speaker_encoder_w:
                    new_global_speaker_encoder_w[key] += args.global_lr * (
                            client_encoder_para[key] - global_encoder_w[key]) * torch.tensor(freq_this_round[idx])
                for key in new_global_speaker_decoder_w:
                    new_global_speaker_decoder_w[key] += args.global_lr * (
                            client_decoder_para[key] - global_decoder_w[key]) * torch.tensor(freq_this_round[idx])

            global_speaker.encoder.load_state_dict(new_global_speaker_encoder_w)
            global_speaker.decoder.load_state_dict(new_global_speaker_decoder_w)

            # Evaluation
            eval_speaker(global_speaker, tok, val_envs, writer, iter_, best_bleu, best_loss)
            
    else:
        party_list = [i for i in range(args.n_parties)]
        n_party_per_round = int(args.n_parties * args.sample_fraction)
        party_list_rounds = []
        if n_party_per_round != args.n_parties:
            for i in range(args.comm_round):
                party_list_rounds.append(random.sample(party_list, n_party_per_round))
        else:
            for i in range(args.comm_round):
                party_list_rounds.append(party_list)

        local_speaker = Speaker(train_env, listner, tok)
        global_speaker = Speaker(train_env, listner, tok)
        new_global_speaker_encoder_w = global_speaker.encoder.state_dict()
        new_global_speaker_decoder_w = global_speaker.decoder.state_dict()
        iter_ = 0
        # calculate frequency
        total_data_points = sum([len(train_env.data[r]) for r in train_env.data])
        fed_avg_freqs = [len(train_env.data[r]) / total_data_points for r in train_env.scans_list]
        # init old models
        old_model_ws = []
        global_speaker.encoder.eval()
        global_speaker.decoder.eval()
        old_nets = []
        for i in range(61):
            old_nets.append(Speaker(train_env, listner, tok))
            old_nets[-1].encoder.load_state_dict(global_speaker.encoder.state_dict())
            old_nets[-1].decoder.load_state_dict(global_speaker.decoder.state_dict())
            old_nets[-1].encoder.eval()
            old_nets[-1].decoder.eval()
        for param in global_speaker.encoder.parameters():
            param.requires_grad = False
        for param in global_speaker.decoder.parameters():
            param.requires_grad = False
        for round in range(args.comm_round):
            party_list_this_round = party_list_rounds[round]
            global_encoder_w = global_speaker.encoder.state_dict()
            global_decoder_w = global_speaker.decoder.state_dict()
            total_freq_round = 0
            for k in party_list_this_round:
                total_freq_round += fed_avg_freqs[k]
            freq_this_round = [fed_avg_freqs[k] / total_freq_round for k in party_list_this_round]

            for idx, k in enumerate(party_list_this_round):
                train_env.set_current_scan(k)
                interval = math.ceil(args.local_epoches * len(train_env.data[train_env.current_scan]) / 64)
                iter_ += interval
                # Train for log_every interval
                local_speaker.env = train_env
                previous_net = []
                for i in range(len(old_model_ws)):
                    previous_net.append(old_model_ws[i][k])
                #  print('pre len: ', len(previous_net))
                local_speaker.train(interval, old_models=previous_net,
                                    global_net=global_speaker)  # Train interval iters
                client_encoder_para = local_speaker.encoder.state_dict()
                client_decoder_para = local_speaker.decoder.state_dict()
                for key in new_global_speaker_encoder_w:
                    new_global_speaker_encoder_w[key] += args.global_lr * (
                            client_encoder_para[key] - global_encoder_w[key]) * torch.tensor(freq_this_round[idx])
                for key in new_global_speaker_decoder_w:
                    new_global_speaker_decoder_w[key] += args.global_lr * (
                            client_decoder_para[key] - global_decoder_w[key]) * torch.tensor(freq_this_round[idx])
                old_nets[k].encoder.load_state_dict(client_encoder_para)
                old_nets[k].decoder.load_state_dict(client_decoder_para)
            if len(old_model_ws) < args.model_buffer_size:
                for net in old_nets:
                    net.encoder.eval()
                    net.decoder.eval()
                    for param in net.encoder.parameters():
                        param.requires_grad = False
                    for param in net.decoder.parameters():
                        param.requires_grad = False
                old_model_ws.append(old_nets)
            else:
                for net in old_nets:
                    net.encoder.eval()
                    net.decoder.eval()
                    for param in net.encoder.parameters():
                        param.requires_grad = False
                    for param in net.decoder.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size - 2, -1, -1):
                    old_model_ws[i] = old_model_ws[i + 1]
                try:
                    old_model_ws[args.model_buffer_size - 1] = old_nets
                except:
                    print(args.model_buffer_size - 1)
            global_speaker.encoder.load_state_dict(new_global_speaker_encoder_w)
            global_speaker.decoder.load_state_dict(new_global_speaker_decoder_w)

            # Evaluation
            eval_speaker(global_speaker, tok, val_envs, writer, iter_, best_bleu, best_loss)


def train(train_env, tok, n_iters, log_every=200, val_envs={}, aug_env=None, val_envs_scan={}):
    writer = SummaryWriter(logdir=log_dir)
    start = time.time()
    random.seed(19)
    party_list = [i for i in range(args.n_parties)]
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list_rounds = []
    args.comm_round = 2000
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    if not args.if_fed and not args.env_based:
        listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)

        speaker = None
        if args.self_train:
            speaker = Speaker(train_env, listner, tok)
            if args.speaker is not None:
                print("Load the speaker from %s." % args.speaker)
                speaker.load(args.speaker)

        start_iter = 0
        if args.load is not None:
            print("LOAD THE listener from %s" % args.load)
            start_iter = listner.load(os.path.join(args.load))

        best_val = {'val_seen': {"accu": 0., 'spl': 0, "state": "", 'update': False},
                    'val_unseen': {"accu": 0., 'spl': 0, "state": "", 'update': False}}
        if args.fast_train:
            log_every = 40
        for idx in range(start_iter, start_iter + n_iters, log_every):
            listner.logs = defaultdict(list)
            interval = min(log_every, n_iters - idx)
            iter = idx + interval

            # Train for log_every interval
            if aug_env is None:  # The default training process
                listner.env = train_env
                listner.train(interval, feedback=feedback_method)  # Train interval iters
            else:
                if args.accumulate_grad:
                    for _ in range(interval // 2):
                        listner.zero_grad()
                        if not args.pre_explore:
                            # Train with GT data
                            listner.env = train_env
                            args.ml_weight = 0.2
                            listner.accumulate_gradient(feedback_method)

                        listner.env = aug_env
                        # Train with Back Translation
                        args.ml_weight = 0.6  # Sem-Configuration
                        listner.accumulate_gradient(feedback_method, speaker=speaker)
                        listner.optim_step()
                else:
                    for _ in range(interval // 2):
                        # Train with GT data
                        listner.env = train_env
                        args.ml_weight = 0.2
                        listner.train(1, feedback=feedback_method)

                        # Train with Back Translation
                        listner.env = aug_env
                        args.ml_weight = 0.6
                        listner.train(1, feedback=feedback_method, speaker=speaker)

            log_loss(listner)
            # Run validation
            loss_str = ""
            loss_str, best_val = eval_model(val_envs, listner, loss_str, best_val)

            for env_name in best_val:
                if best_val[env_name]['update']:
                    best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                    best_val[env_name]['update'] = False
                    listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

            print(('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                        iter, float(iter) / n_iters * 100, loss_str)))

            if iter % 1000 == 0:
                print("BEST RESULT TILL NOW")
                for env_name in best_val:
                    print(env_name, best_val[env_name]['state'])

            if iter % 50000 == 0:
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))

        listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))
    elif not args.if_fed and args.env_based:
        listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
        unseen_agents = [Seq2SeqAgent(train_env, os.path.join('snap', args.name, 'best_results.json'), tok, args.maxAction) for i in range(11)]
        speaker = None
        scan_list = [aug_env.scans_list[i] for i in range(11)]
        if args.self_train:
            speaker = Speaker(train_env, listner, tok)
            if args.speaker is not None:
                print("Load the speaker from %s." % args.speaker)
                speaker.load(args.speaker)

        start_iter = 0
        if args.load is not None:
            print("LOAD THE listener from %s" % args.load)
            start_iter = listner.load(os.path.join(args.load))
        for model in unseen_agents:
            model.encoder.load_state_dict(listner.encoder.state_dict())
            model.decoder.load_state_dict(listner.decoder.state_dict())
            model.critic.load_state_dict(listner.critic.state_dict())
        start = time.time()

        best_val = {'val_seen': {"accu": 0., 'spl': 0, "state": "", 'update': False},
                    'val_unseen': {"accu": 0., 'spl': 0, "state": "", 'update': False}}
        if args.fast_train:
            log_every = 40
        log_every = 10
        loss_str = ""
        best_val, loss_str = val_client(val_envs_scan, unseen_agents, best_val, loss_str, scan_list)
        print(loss_str)
        for idx in range(start_iter, start_iter + n_iters, log_every):
            for k, model in enumerate(unseen_agents):
                print(k)
                model.logs = defaultdict(list)
                aug_env.set_current_scan(k)
                interval = math.ceil(len(aug_env.data[aug_env.current_scan]) / args.batchSize)
                # interval = min(log_every, n_iters - idx)
                iter = idx + interval
                model.env = aug_env
                args.ml_weight = 0.6
                model.train(interval, feedback=feedback_method, speaker=speaker)
                # Log the training stats to tensorboard
                log_loss(model)
            # Run validation
            loss_str = ""
            best_val, loss_str = val_client(val_envs_scan, unseen_agents, best_val, loss_str, scan_list)
            # extra_test(val_envs_scan, unseen_agents, scan_list, val_envs)
            for env_name in best_val:
                if best_val[env_name]['update']:
                    best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                    best_val[env_name]['update'] = False
                    listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

                    output = []
                    for model in unseen_agents:
                        output += model.get_results()
                    with open(unseen_agents[0].results_path, 'w') as f:
                        json.dump(output, f)

            print(('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                        iter, float(iter) / n_iters * 100, loss_str)))

            if iter % 1000 == 0:
                print("BEST RESULT TILL NOW")
                for env_name in best_val:
                    print(env_name, best_val[env_name]['state'])

        # listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))
    elif args.fed_alg=='fedavg' and not args.pre_explore:
        glr_schedule_freq = 30000
        glr_schedule_thresh = glr_schedule_freq
        start_iter = 0
        iter = 0
        listner_global = Seq2SeqAgent(train_env, "", tok, args.maxAction)
        listner_client = Seq2SeqAgent(train_env, "", tok, args.maxAction)
        speaker = None
        if args.self_train:
            speaker = Speaker(train_env, listner_client, tok)
            if args.speaker is not None:
                print("Load the speaker from %s." % args.speaker)
                speaker.load(args.speaker)
        if args.load is not None:
            if args.aug is None:
                start_iter = listner_global.load(os.path.join(args.load))
                iter = start_iter
                print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))
            else:
                iter = listner_global.load(os.path.join(args.load))
                print("\nLOAD the model from {}, iteration ".format(args.load, iter))
        start = time.time()
        print('\nListener training starts, start iteration: %s' % str(start_iter))

        best_val = {'val_seen': {"accu": 0., 'spl': 0, "state": "", 'update': False},
                    'val_unseen': {"accu": 0., 'spl': 0, "state": "", 'update': False}}
        # calculate frequency
        total_data_points = sum([len(train_env.data[r]) for r in train_env.data])
        fed_avg_freqs = [len(train_env.data[r]) / total_data_points for r in train_env.scans_list]
        if aug_env is None:
            comm_round = math.ceil((args.iters-iter) / (args.local_epoches * train_env.size() * args.sample_fraction / args.batchSize))
        else:
            comm_round = math.ceil((args.iters-iter) / (args.local_epoches * train_env.size() * args.sample_fraction/ args.batchSize))

        for round in range(comm_round):
            if iter > glr_schedule_thresh:
                args.global_lr *= args.glr_decrease_rate
                glr_schedule_thresh += glr_schedule_freq
            listner_client.logs = defaultdict(list)
            global_w_encoder = listner_global.encoder.state_dict()
            global_w_decoder = listner_global.decoder.state_dict()
            global_w_critic = listner_global.critic.state_dict()
            party_list_this_round = party_list_rounds[round]
            new_global_w_encoder = copy.deepcopy(listner_global.encoder.state_dict())
            new_global_w_decoder = copy.deepcopy(listner_global.decoder.state_dict())
            new_global_w_critic = copy.deepcopy(listner_global.critic.state_dict())

            total_freq_round = 0
            for k in party_list_this_round:
                total_freq_round += fed_avg_freqs[k]
            freq_this_round = [fed_avg_freqs[k] / total_freq_round for k in party_list_this_round]

            for idx, k in enumerate(party_list_this_round):
                # print(global_w_encoder['lstm.weight_ih_l0'][0,:30])
                print('party: ', k)
                train_env.set_current_scan(k)
                scan = train_env.scans_list[k]
                num_step = math.ceil(args.local_epoches * len(train_env.data[train_env.current_scan]) / 64)
                iter += num_step
                listner_client.encoder.load_state_dict(global_w_encoder)
                listner_client.decoder.load_state_dict(global_w_decoder)
                listner_client.critic.load_state_dict(global_w_critic)
                if aug_env is None:  # The default training process
                    listner_client.env = train_env
                    listner_client.train(num_step, feedback=feedback_method)
                else:
                    if scan in aug_env.scans_list:
                        aug_env.set_current_scan(scan)
                        aug = 1
                    else:
                        aug = 0
                    if args.accumulate_grad:
                        for _ in range(math.ceil(num_step / 2)):
                            listner_client.zero_grad()
                            listner_client.env = train_env

                            # Train with GT data
                            args.ml_weight = 0.2
                            listner_client.accumulate_gradient(feedback_method)
                            if aug:
                                # Train with Augmented data
                                listner_client.env = aug_env
                                # Train with Back Translation
                                args.ml_weight = 0.6  # Sem-Configuration
                                listner_client.accumulate_gradient(feedback_method, speaker=speaker)
                            else:
                                listner_client.accumulate_gradient(feedback_method)
                            listner_client.optim_step()
                    else:
                        for step in range(math.ceil(num_step / 2)):
                            # Train with GT data
                            listner_client.env = train_env
                            args.ml_weight = 0.2
                            listner_client.train(1, feedback=feedback_method)
                            if aug:
                                # Train with Augmented data
                                listner_client.env = aug_env
                                args.ml_weight = 0.6
                                listner_client.train(1, feedback=feedback_method, speaker=speaker)
                            else:
                                listner_client.train(1, feedback=feedback_method)

                    # print_progress(step, num_step, prefix='Progress:', suffix='Complete', bar_length=50)
                client_critic_para = listner_client.critic.state_dict()
                client_encoder_para = listner_client.encoder.state_dict()
                client_decoder_para = listner_client.decoder.state_dict()
                for key in new_global_w_critic:
                    new_global_w_critic[key] += args.global_lr * (client_critic_para[key]-global_w_critic[key]) * torch.tensor(freq_this_round[idx])
                for key in new_global_w_encoder:
                    new_global_w_encoder[key] += args.global_lr * (client_encoder_para[key]-global_w_encoder[key]) * torch.tensor(freq_this_round[idx])
                for key in new_global_w_decoder:
                    new_global_w_decoder[key] += args.global_lr * (client_decoder_para[key]-global_w_decoder[key]) * torch.tensor(freq_this_round[idx])

            # Log the training stats to tensorboard
            log_loss(listner_client)
            # update global
            listner_global.encoder.load_state_dict(new_global_w_encoder)
            listner_global.decoder.load_state_dict(new_global_w_decoder)
            listner_global.critic.load_state_dict(new_global_w_critic)
            # Run validation
            loss_str = "iter {}".format(iter)
            loss_str, best_val = eval_model(val_envs, listner_global, loss_str, best_val)
            record_file = open('./logs/' + args.name + '.txt', 'a')
            record_file.write(loss_str + '\n')
            record_file.close()

            for env_name in best_val:
                if best_val[env_name]['update']:
                    best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                    best_val[env_name]['update'] = False
                    listner_global.save(iter, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))
                else:
                    listner_global.save(iter, os.path.join("snap", args.name, "state_dict", "latest_dict"))

            print(('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                        iter, float(iter) / n_iters * 100, loss_str)))

            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

                record_file = open('./logs/' + args.name + '.txt', 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
                record_file.close()

        listner_global.save(iter, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (iter)))
    elif args.fed_alg=='fedavg' and args.pre_explore:
        start_iter = 0
        iter = 0
        if not args.unseen_only:
            party_list = [i for i in range(61)]
            n_party_per_round = 11
            party_list_rounds = []
            args.comm_round = 1000
            if n_party_per_round != args.n_parties:
                for i in range(args.comm_round):
                    party_list_rounds.append(random.sample(party_list, n_party_per_round))
            else:
                for i in range(args.comm_round):
                    party_list_rounds.append(party_list)
        scan_list = [aug_env.scans_list[i] for i in range(11)]
        listner_global = Seq2SeqAgent(train_env, "", tok, args.maxAction)
        listner_client = Seq2SeqAgent(train_env, "", tok, args.maxAction)
        speaker = None
        if args.part_unseen:
            unseen_models = [Seq2SeqAgent(train_env, "", tok, args.maxAction) for i in range(6)]
        else:
            unseen_models = [Seq2SeqAgent(train_env, os.path.join('snap', args.name, 'best_results.json'), tok, args.maxAction) for i in range(11)]
        seen_models = [Seq2SeqAgent(train_env, "", tok, args.maxAction) for i in range(61)]
        if args.self_train:
            if args.speaker is not None:
                speaker = Speaker(train_env, listner_client, tok)
                print("Load the speaker from %s." % args.speaker)
                speaker.load(args.speaker)
        if args.load is not None:
            if args.aug is None:
                start_iter = listner_global.load(os.path.join(args.load))
                print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))
            else:
                iter = listner_global.load(os.path.join(args.load))
                print("\nLOAD the model from {}, iteration ".format(args.load, iter))
        start = time.time()
        print('\nListener training starts, start iteration: %s' % str(start_iter))
        for model in unseen_models:
            model.encoder.load_state_dict(listner_global.encoder.state_dict())
            model.decoder.load_state_dict(listner_global.decoder.state_dict())
            model.critic.load_state_dict(listner_global.critic.state_dict())
        for model in seen_models:
            model.encoder.load_state_dict(listner_global.encoder.state_dict())
            model.decoder.load_state_dict(listner_global.decoder.state_dict())
            model.critic.load_state_dict(listner_global.critic.state_dict())
        best_val = {'val_seen': {"accu": 0., 'spl': 0, "state": "", 'update': False},
                    'val_unseen': {"accu": 0., 'spl': 0, "state": "", 'update': False}}
        # calculate frequency
        if not args.unseen_only:
            total_data_points = sum([len(train_env.data[train_env.scans_list[r]]) for r in range(args.n_parties-11)])
            total_data_points += sum([len(aug_env.data[r]) for r in aug_env.data])
            fed_avg_freqs = [len(train_env.data[train_env.scans_list[i]]) / total_data_points for i in range(args.n_parties-11)]
            fed_avg_freqs += [len(aug_env.data[r]) / total_data_points for r in aug_env.scans_list]
            comm_round = math.ceil((args.iters) / ((train_env.size()*(args.n_parties-11)/61 + aug_env.size())*args.local_epoches*args.sample_fraction / args.batchSize))
        else:
            total_data_points = sum([len(aug_env.data[r]) for r in aug_env.data])
            fed_avg_freqs = [len(aug_env.data[r]) / total_data_points for r in aug_env.scans_list]
            comm_round = math.ceil((args.iters)/ (aug_env.size()*args.local_epoches*args.sample_fraction / args.batchSize))
            print('comm round:', comm_round)
        loss_str = ""
        best_val, loss_str = val_client(val_envs_scan, unseen_models, best_val, loss_str, scan_list, args.part_unseen)
        print(loss_str)
        loss_str = ''
        print('total parties: ', len(fed_avg_freqs))
        print('comm round: ', comm_round)
        if args.part_unseen:
            best_val_global = copy.deepcopy(best_val)
            models_list = [listner_global for i in range(5)]
            best_val_global, loss_str = val_client(val_envs_scan, models_list, best_val_global, loss_str,
                                                   scan_list[-5:], args.part_unseen)
            print(best_val_global)
        else:
            loss_str, best_val = eval_model(val_envs, listner_global, loss_str, best_val)
            print(best_val)
        unseen_parties = 11
        for round in range(comm_round):
            listner_client.logs = defaultdict(list)
            global_w_encoder = listner_global.encoder.state_dict()
            global_w_decoder = listner_global.decoder.state_dict()
            global_w_critic = listner_global.critic.state_dict()
            party_list_this_round = party_list_rounds[round]
            if args.n_parties > 61:
                party_list_this_round += random.sample([i for i in range(61, args.n_parties)], int(11 * args.sample_fraction))
            new_global_w_encoder = copy.deepcopy(listner_global.encoder.state_dict())
            new_global_w_decoder = copy.deepcopy(listner_global.decoder.state_dict())
            new_global_w_critic = copy.deepcopy(listner_global.critic.state_dict())

            total_freq_round = 0
            for k in party_list_this_round:
                total_freq_round += fed_avg_freqs[k]
            freq_this_round = [fed_avg_freqs[k] / total_freq_round for k in party_list_this_round]

            for idx, k in enumerate(party_list_this_round):
                print('party: ', k)
                if not args.unseen_only:
                    if k < (args.n_parties - unseen_parties):
                        listner_client.encoder.load_state_dict(global_w_encoder)
                        if args.enc_only:
                            listner_client.decoder.load_state_dict(seen_models[k].decoder.state_dict())
                            listner_client.critic.load_state_dict(seen_models[k].critic.state_dict())
                        else:
                            listner_client.decoder.load_state_dict(global_w_decoder)
                            listner_client.critic.load_state_dict(global_w_critic)
                    else:
                        listner_client.encoder.load_state_dict(global_w_encoder)
                        if args.enc_only:
                            listner_client.decoder.load_state_dict(unseen_models[k-(args.n_parties-unseen_parties)].decoder.state_dict())
                            listner_client.critic.load_state_dict(unseen_models[k-(args.n_parties-unseen_parties)].critic.state_dict())
                        else:
                            listner_client.decoder.load_state_dict(global_w_decoder)
                            listner_client.critic.load_state_dict(global_w_critic)
                else:
                    listner_client.encoder.load_state_dict(global_w_encoder)
                    if args.enc_only:
                        listner_client.decoder.load_state_dict(unseen_models[k].decoder.state_dict())
                        listner_client.critic.load_state_dict(unseen_models[k].critic.state_dict())
                    else:
                        listner_client.decoder.load_state_dict(global_w_decoder)
                        listner_client.critic.load_state_dict(global_w_critic)
                if not args.unseen_only:
                    if k < (args.n_parties - unseen_parties):
                        train_env.set_current_scan(k)
                        args.ml_weight = 0.2
                        num_step = math.ceil(args.local_epoches * len(train_env.data[train_env.current_scan]) / args.batchSize)
                        listner_client.env = train_env
                        listner_client.train(num_step, feedback=feedback_method)
                        seen_models[k].encoder.load_state_dict(listner_client.encoder.state_dict())
                        seen_models[k].decoder.load_state_dict(listner_client.decoder.state_dict())
                        seen_models[k].critic.load_state_dict(listner_client.critic.state_dict())

                    else:
                        aug_env.set_current_scan(k - (args.n_parties - unseen_parties))
                        num_step = math.ceil(args.local_epoches * len(aug_env.data[aug_env.current_scan]) / args.batchSize)
                        listner_client.env = aug_env
                        args.ml_weight = 0.6
                        listner_client.train(num_step, feedback=feedback_method, speaker=speaker)
                        unseen_models[k-(args.n_parties-unseen_parties)].encoder.load_state_dict(listner_client.encoder.state_dict())
                        unseen_models[k - (args.n_parties-unseen_parties)].decoder.load_state_dict(listner_client.decoder.state_dict())
                        unseen_models[k - (args.n_parties-unseen_parties)].critic.load_state_dict(listner_client.critic.state_dict())
                else:
                    aug_env.set_current_scan(k)
                    num_step = math.ceil(args.local_epoches * len(aug_env.data[aug_env.current_scan]) / args.batchSize)
                    listner_client.env = aug_env
                    args.ml_weight = 0.6
                    listner_client.train(num_step, feedback=feedback_method, speaker=speaker)
                    unseen_models[k].encoder.load_state_dict(listner_client.encoder.state_dict())
                    unseen_models[k].decoder.load_state_dict(listner_client.decoder.state_dict())
                    unseen_models[k].critic.load_state_dict(listner_client.critic.state_dict())


                iter += num_step
                client_critic_para = listner_client.critic.state_dict()
                client_encoder_para = listner_client.encoder.state_dict()
                client_decoder_para = listner_client.decoder.state_dict()
                for key in new_global_w_critic:
                    new_global_w_critic[key] += args.global_lr * (
                                client_critic_para[key] - global_w_critic[key]) * torch.tensor(freq_this_round[idx])
                for key in new_global_w_encoder:
                    new_global_w_encoder[key] += args.global_lr * (
                                client_encoder_para[key] - global_w_encoder[key]) * torch.tensor(freq_this_round[idx])
                for key in new_global_w_decoder:
                    new_global_w_decoder[key] += args.global_lr * (
                                client_decoder_para[key] - global_w_decoder[key]) * torch.tensor(freq_this_round[idx])
            loss_str = ""
            best_val_local, loss_str = val_client(val_envs_scan, unseen_models, best_val, loss_str, scan_list)
            # Log the training stats to tensorboard
            log_loss(listner_client)
            # update global
            listner_global.encoder.load_state_dict(new_global_w_encoder)
            listner_global.decoder.load_state_dict(new_global_w_decoder)
            listner_global.critic.load_state_dict(new_global_w_critic)
            # extra_test(val_envs_scan, [listner_global], scan_list, val_envs)
            #loss_str = "iter {}".format(iter)
            #if args.part_unseen:
            #    best_val_global = copy.deepcopy(best_val)
            #    models_list = [listner_global for i in range(5)]
            #    best_val_global, loss_str = val_client(val_envs_scan, models_list, best_val_global, loss_str, scan_list[-5:], args.part_unseen)

            #loss_str, best_val = eval_model(val_envs, listner_global, loss_str, best_val)

            record_file = open('./logs/' + args.name + '.txt', 'a')
            record_file.write(loss_str + '\n')
            record_file.close()

            for env_name in best_val_local:
                if best_val[env_name]['update']:
                    best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                    best_val[env_name]['update'] = False
                    listner_global.save(iter, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))
                    # record current result
                    output = []
                    for model in unseen_models:
                        output += model.get_results()
                    with open(unseen_models[0].results_path, 'w') as f:
                        json.dump(output, f)
                else:
                    listner_global.save(iter, os.path.join("snap", args.name, "state_dict", "latest_dict"))

            print(('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                        iter, float(iter) / n_iters * 100, loss_str)))

            print("BEST RESULT TILL NOW")
            for env_name in best_val:
                print(env_name, best_val[env_name]['state'])

                record_file = open('./logs/' + args.name + '.txt', 'a')
                record_file.write('BEST RESULT TILL NOW: ' + env_name + ' | ' + best_val[env_name]['state'] + '\n')
                record_file.close()

        listner_global.save(iter, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (iter)))


def val_client(val_envs, client_models, best_val, loss_str, scan_list, part_unseen=False):
    total_sr = 0
    total_spl = 0
    total_ne = 0
    total_osr = 0
    total_cls = 0
    total_ndtw = 0
    total_sdtw = 0
    total_lengths = 0
    total_steps = 0
    for env_name, (env, evaluator) in val_envs.items():
        if env_name != 'val_unseen':
            continue
        if not part_unseen:
            total_sample = env.size()
        else:
            total_sample = 0
            for scan in scan_list:
                total_sample += len(env.data[scan])
        for i, model in enumerate(client_models):
            model.env = env
            env.set_current_scan(scan_list[i])
            # Get validation distance from goal under test evaluation conditions
            model.test(use_dropout=False, feedback='argmax', iters=None)
            result = model.get_results()
            if len(result) != len(env.data[env.current_scan]):
                print(len(result), len(env.data[env.current_scan]), env.current_scan)
            score_summary, _ = evaluator.score(result)
            total_sr += score_summary['success_rate'] * len(env.data[env.current_scan]) / total_sample
            total_spl += score_summary['spl'] * len(env.data[env.current_scan]) / total_sample
            total_ne += score_summary['nav_error'] * len(env.data[env.current_scan]) / total_sample
            total_osr += score_summary['oracle_rate'] * len(env.data[env.current_scan]) / total_sample
            total_cls += score_summary['cls'] * len(env.data[env.current_scan]) / total_sample
            total_ndtw += score_summary['ndtw'] * len(env.data[env.current_scan]) / total_sample
            total_sdtw += score_summary['sdtw'] * len(env.data[env.current_scan]) / total_sample
            total_lengths += score_summary['lengths'] * len(env.data[env.current_scan]) / total_sample
            total_steps += score_summary['steps'] * len(env.data[env.current_scan]) / total_sample
        if total_sr > best_val[env_name]['accu']:
            best_val[env_name]['accu'] = total_sr
            best_val[env_name]['update'] = True
        if not part_unseen:
            loss_str += 'local_sr %.4f, local_spl %.4f, local_ne %.4f, local_osr %.4f, cls %.4f, ndtw %.4f, pl %.4f, steps %.4f' \
                        % (total_sr, total_spl, total_ne, total_osr, total_cls, total_ndtw, total_lengths, total_steps)
            wandb.log({"local_sr": total_sr})
            wandb.log({"local_spl": total_spl})
            wandb.log({"local_ne": total_ne})
            wandb.log({"local_osr": total_osr})
            wandb.log({"local_cls": total_cls})
            wandb.log({"local_ndtw": total_ndtw})
            wandb.log({"local_sdtw": total_sdtw})
            wandb.log({"local_pl": total_lengths})
            wandb.log({"local_steps": total_steps})
        else:
            loss_str += 'global_sr %.4f, global_spl %.4f, global_ne %.4f, global_osr %.4f' % (
                total_sr, total_spl, total_ne, total_osr)
            wandb.log({"global_sr": total_sr})
            wandb.log({"global_spl": total_spl})
            wandb.log({"global_ne": total_ne})
            wandb.log({"global_osr": total_osr})
        print(loss_str)
    return best_val, loss_str


def log_loss(model):
    total = max(sum(model.logs['total']), 1)
    length = max(len(model.logs['critic_loss']), 1)
    critic_loss = sum(model.logs['critic_loss']) / total  # / length / args.batchSize
    entropy = sum(model.logs['entropy']) / total  # / length / args.batchSize
    RL_loss = sum(model.logs['RL_loss']) / max(len(model.logs['RL_loss']), 1)
    IL_loss = sum(model.logs['IL_loss']) / max(len(model.logs['IL_loss']), 1)
    wandb.log({"loss/critic": critic_loss})
    wandb.log({"policy_entropy": entropy})
    wandb.log({"loss/RL_loss": RL_loss})
    wandb.log({"loss/IL_loss": IL_loss})
    wandb.log({"total_actions": total})
    wandb.log({"max_length": length})


def eval_model(val_envs, listner, loss_str, best_val):
    for env_name, (env, evaluator) in val_envs.items():
        listner.env = env
        # Get validation distance from goal under test evaluation conditions
        listner.test(use_dropout=False, feedback='argmax', iters=None)
        result = listner.get_results()
        score_summary, _ = evaluator.score(result)
        loss_str += ", %s " % env_name
        # update best_val, 
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
        wandb.log({"ndtw/%s" % env_name: score_summary['ndtw']})
        wandb.log({"cls/%s" % env_name: score_summary['cls']})
        # print(loss_str)
        
    return loss_str, best_val


def valid(train_env, tok, val_envs={}):
    agent = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    print("Loaded the listener model at iter %d from %s" % (agent.load(args.load), args.load))

    for env_name, (env, evaluator) in val_envs.items():
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        agent.test(use_dropout=False, feedback='argmax', iters=iters)
        result = agent.get_results()

        if env_name != '':
            score_summary, _ = evaluator.score(result)
            loss_str = "Env name: %s" % env_name
            for metric,val in score_summary.items():
                loss_str += ', %s: %.4f' % (metric, val)
            print(loss_str)

        if args.submit:
            json.dump(
                result,
                open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )


def beam_valid(train_env, tok, val_envs={}):
    listener = Seq2SeqAgent(train_env, "", tok, args.maxAction)

    speaker = Speaker(train_env, listener, tok)
    if args.speaker is not None:
        print("Load the speaker from %s." % args.speaker)
        speaker.load(args.speaker)

    print("Loaded the listener model at iter % d" % listener.load(args.load))

    final_log = ""
    for env_name, (env, evaluator) in val_envs.items():
        listener.logs = defaultdict(list)
        listener.env = env

        listener.beam_search_test(speaker)
        results = listener.results

        def cal_score(x, alpha, avg_speaker, avg_listener):
            speaker_score = sum(x["speaker_scores"]) * alpha
            if avg_speaker:
                speaker_score /= len(x["speaker_scores"])
            # normalizer = sum(math.log(k) for k in x['listener_actions'])
            normalizer = 0.
            listener_score = (sum(x["listener_scores"]) + normalizer) * (1-alpha)
            if avg_listener:
                listener_score /= len(x["listener_scores"])
            return speaker_score + listener_score

        if args.param_search:
            # Search for the best speaker / listener ratio
            interval = 0.01
            logs = []
            for avg_speaker in [False, True]:
                for avg_listener in [False, True]:
                    for alpha in np.arange(0, 1 + interval, interval):
                        result_for_eval = []
                        for key in results:
                            result_for_eval.append({
                                "instr_id": key,
                                "trajectory": max(results[key]['paths'],
                                                  key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                                  )['trajectory']
                            })
                        score_summary, _ = evaluator.score(result_for_eval)
                        for metric,val in score_summary.items():
                            if metric in ['success_rate']:
                                print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                                      (avg_speaker, avg_listener, alpha, val))
                                logs.append((avg_speaker, avg_listener, alpha, val))
            tmp_result = "Env Name %s\n" % (env_name) + \
                    "Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f\n" % max(logs, key=lambda x: x[3])
            print(tmp_result)
            # print("Env Name %s" % (env_name))
            # print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
            #       max(logs, key=lambda x: x[3]))
            final_log += tmp_result
            print()
        else:
            avg_speaker = True
            avg_listener = True
            alpha = args.alpha

            result_for_eval = []
            for key in results:
                result_for_eval.append({
                    "instr_id": key,
                    "trajectory": [(vp, 0, 0) for vp in results[key]['dijk_path']] + \
                                  max(results[key]['paths'],
                                   key=lambda x: cal_score(x, alpha, avg_speaker, avg_listener)
                                  )['trajectory']
                })
            # result_for_eval = utils.add_exploration(result_for_eval)
            score_summary, _ = evaluator.score(result_for_eval)

            if env_name != 'test':
                loss_str = "Env Name: %s" % env_name
                for metric, val in score_summary.items():
                    if metric in ['success_rate']:
                        print("Avg speaker %s, Avg listener %s, For the speaker weight %0.4f, the result is %0.4f" %
                              (avg_speaker, avg_listener, alpha, val))
                    loss_str += ",%s: %0.4f " % (metric, val)
                print(loss_str)
            print()

            if args.submit:
                json.dump(
                    result_for_eval,
                    open(os.path.join(log_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
    print(final_log)


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
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

    val_env_names = ['val_unseen', 'val_seen']
    if args.submit:
        val_env_names.append('test')
    else:
        pass
        #val_env_names.append('train')

    #if not args.beam:
    #    val_env_names.append("train")

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
           Evaluation([split], featurized_scans, tok))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validlistener':
        if args.beam:
            beam_valid(train_env, tok, val_envs=val_envs)
        else:
            valid(train_env, tok, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validspeaker':
        valid_speaker(tok, val_envs)
    else:
        assert False


def valid_speaker(tok, val_envs):
    import tqdm
    listner = Seq2SeqAgent(None, "", tok, args.maxAction)
    speaker = Speaker(None, listner, tok)
    speaker.load(args.load)

    for env_name, (env, evaluator) in val_envs.items():
        if env_name == 'train':
            continue
        print("............ Evaluating %s ............." % env_name)
        speaker.env = env
        path2inst, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm)
        path_id = next(iter(path2inst.keys()))
        print("Inference: ", tok.decode_sentence(path2inst[path_id]))
        print("GT: ", evaluator.gt[path_id]['instructions'])
        pathXinst = list(path2inst.items())
        name2score = evaluator.lang_eval(pathXinst, no_metrics={'METEOR'})
        score_string = " "
        for score_name, score in name2score.items():
            score_string += "%s_%s: %0.4f " % (env_name, score_name, score)
        print("For env %s" % env_name)
        print(score_string)
        print("Average Length %0.4f" % utils.average_length(path2inst))


def train_val_augment():
    """
    Train the listener with the augmented data
    """
    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    # Load the env img features
    feat_dict = read_img_features(args.features)
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    # Load the augmentation data
    aug_path = args.aug

    # Create the training environment
    if args.if_fed or args.env_based:
        train_env = R2RBatchScan(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
        aug_env = R2RBatchScan(feat_dict, batch_size=args.batchSize, splits=[aug_path], tokenizer=tok, name='aug')
    else:
        train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
        aug_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=[aug_path], tokenizer=tok, name='aug')


    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['val_seen', 'val_unseen']}
    val_envs_scan = {split: (R2RBatchScan(feat_dict, batch_size=args.batchSize, splits=[split],
                                          tokenizer=tok), Evaluation([split], featurized_scans, tok))
                     for split in ['val_seen', 'val_unseen']}
    """
    train_length = []
    for key in train_env.data:
        train_length.append(len(train_env.data[key]))
    print('min: ', min(train_length), 'max: ', max(train_length))
    aug_length = []
    for key in aug_env.data:
        aug_length.append(len(aug_env.data[key]))
        print(len(aug_env.data[key]))
    print('min: ', min(aug_length), 'max: ', max(aug_length))
    time.sleep(5)
    """
    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env, val_envs_scan=val_envs_scan)


if __name__ == "__main__":
    if args.train in ['speaker', 'rlspeaker', 'validspeaker',
                      'listener', 'validlistener']:
        train_val()
    elif args.train == 'auglistener':
        train_val_augment()
    else:
        assert False

