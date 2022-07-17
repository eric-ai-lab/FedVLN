
import torch

import os
import time
import json
import numpy as np
from collections import defaultdict
from speaker import Speaker

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features, build_vocab_hi, build_vocab_te
import utils
from env import R2RBatch, R2RBatchScan
from agent import Seq2SeqAgent
from eval import Evaluation
from param import args

import warnings
warnings.filterwarnings("ignore")
import random

from tensorboardX import SummaryWriter
import wandb
import  math

wandb.init(project="semantic_EnvDrop_fedavg", entity="kzhou")

log_dir = 'snap/%s' % args.name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

TRAIN_VOCAB = 'tasks/RxR/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/RxR/data/trainval_vocab.txt'
TRAIN_VOCAB_HI = 'tasks/RxR/data/train_vocab_hi.txt'
TRAINVAL_VOCAB_HI = 'tasks/RxR/data/trainval_vocab_hi.txt'
TRAIN_VOCAB_TE = 'tasks/RxR/data/train_vocab_te.txt'
TRAINVAL_VOCAB_TE = 'tasks/RxR/data/trainval_vocab_te.txt'

feedback_method = args.feedback # teacher or sample

print(args)


def train_speaker(train_env, tok, n_iters, log_every=500, val_envs={}):
    writer = SummaryWriter(logdir=log_dir)
    listner = Seq2SeqAgent(train_env, "", tok, args.maxAction)
    speaker = Speaker(train_env, listner, tok)

    if args.fast_train:
        log_every = 40

    best_bleu = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 1232)
    for idx in range(0, n_iters, log_every):
        interval = min(log_every, n_iters - idx)

        # Train for log_every interval
        speaker.env = train_env
        speaker.train(interval)   # Train interval iters

        print()
        print("Iter: %d" % idx)

        # Evaluation
        for env_name, (env, evaluator) in val_envs.items():
            if 'train' in env_name: # Ignore the large training set for the efficiency
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


def val_client(val_envs, client_models):
    total_sr = 0
    total_spl = 0
    total_ne = 0
    total_osr = 0
    for env_name, (env, evaluator) in val_envs.items():
        if env_name != 'val_unseen':
            continue
        total_sample = env.size()
        for i, model in enumerate(client_models):
            model.env = env
            env.set_current_scan(i)
            # Get validation distance from goal under test evaluation conditions
            model.test(use_dropout=False, feedback='argmax', iters=None)
            result = model.get_results()
            score_summary, _ = evaluator.score(result)
            total_sr += score_summary['success_rate'] * len(env.data[env.current_scan]) / total_sample
            total_spl += score_summary['spl'] * len(env.data[env.current_scan]) / total_sample
            total_ne += score_summary['nav_error'] * len(env.data[env.current_scan]) / total_sample
            total_osr += score_summary['oracle_rate'] * len(env.data[env.current_scan]) / total_sample

        wandb.log({"local_sr": total_sr})
        wandb.log({"local_spl": total_spl})
        wandb.log({"local_ne": total_ne})
        wandb.log({"local_osr": total_osr})


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
    return loss_str, best_val


def train(train_env, tok, n_iters, log_every=1000, val_envs={}, aug_env=None):
    writer = SummaryWriter(logdir=log_dir)
    random.seed(args.seed)
    party_list = [i for i in range(args.n_parties)]
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)
    if not args.if_fed:
        listner = None
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

        start = time.time()

        best_val = {'val_seen': {"accu": 0., "ndtw": 0., "state": "", 'update': False},
                    'val_unseen': {"accu": 0., "ndtw": 0., "state": "", 'update': False}}
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
                        listner.env = train_env

                        # Train with GT data
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

            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)
            length = max(len(listner.logs['critic_loss']), 1)
            critic_loss = sum(listner.logs['critic_loss']) / total  # / length / args.batchSize
            entropy = sum(listner.logs['entropy']) / total  # / length / args.batchSize
            predict_loss = sum(listner.logs['us_loss']) / max(len(listner.logs['us_loss']), 1)
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/unsupervised", predict_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            print("total_actions", total)
            print("max_length", length)
            wandb.log({"loss/critic": critic_loss})
            wandb.log({"policy_entropy": entropy})
            wandb.log({"loss/unsupervised": predict_loss})
            wandb.log({"total_actions": total})
            wandb.log({"max_length": length})
            wandb.log({"loss/RL_loss": RL_loss})
            wandb.log({"loss/IL_loss": IL_loss})
            # Run validation
            loss_str = ""
            for env_name, (env, evaluator) in val_envs.items():
                listner.env = env

                # Get validation loss under the same conditions as training
                iters = None if args.fast_train or env_name != 'train' else 20  # 20 * 64 = 1280

                # Get validation distance from goal under test evaluation conditions
                listner.test(use_dropout=False, feedback='argmax', iters=iters)
                result = listner.get_results()
                score_summary, _ = evaluator.score(result)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    if metric in ['success_rate']:
                        writer.add_scalar("accuracy/%s" % env_name, val, idx)
                        if env_name in best_val:
                            if val > best_val[env_name]['accu']:
                                best_val[env_name]['accu'] = val
                                best_val[env_name]['update'] = True
                    else:
                        writer.add_scalar(metric + "/%s" % env_name, val, idx)

                    if metric == "ndtw":
                        if env_name in best_val:
                            if val > best_val[env_name]['ndtw']:
                                best_val[env_name]['ndtw'] = val
                                listner.save(idx,
                                             os.path.join("snap", args.name, "state_dict", "best_%s_ndtw" % (env_name)))

                    loss_str += ', %s: %.3f' % (metric, val)
                wandb.log({"sr/%s" % env_name: score_summary['success_rate']})
                wandb.log({"tl/%s" % env_name: score_summary['lengths']})
                wandb.log({"ne/%s" % env_name: score_summary['nav_error']})
                wandb.log({"osr/%s" % env_name: score_summary['oracle_rate']})
                wandb.log({"spl/%s" % env_name: score_summary['spl']})
                wandb.log({"ndtw/%s" % env_name: score_summary['ndtw']})
                wandb.log({"sdtw/%s" % env_name: score_summary['sdtw']})

            for env_name in best_val:
                if best_val[env_name]['update']:
                    best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                    best_val[env_name]['update'] = False
                    listner.save(idx, os.path.join("snap", args.name, "state_dict", "best_%s" % (env_name)))

            print(('%s (%d %d%%) %s' % (timeSince(start, float(iter) / n_iters),
                                        iter, float(iter) / n_iters * 100, loss_str)))

            if iter % 5000 == 0:
                print("BEST RESULT TILL NOW")
                for env_name in best_val:
                    print(env_name, best_val[env_name]['state'])

            if iter % 50000 == 0:
                listner.save(idx, os.path.join("snap", args.name, "state_dict", "Iter_%06d" % (iter)))

        listner.save(idx, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (idx)))
    elif args.fed_alg=='fedavg' and not args.pre_explore:
        start_iter = 0
        iter = 0
        listner_global = Seq2SeqAgent(train_env, "", tok, args.maxAction)
        listner_client = Seq2SeqAgent(train_env, "", tok, args.maxAction)
        load_iter = 0
        speaker = None
        if args.self_train:
            speaker = Speaker(train_env, listner_client, tok)
            if args.speaker is not None:
                print("Load the speaker from %s." % args.speaker)
                speaker.load(args.speaker)
        if args.load is not None:
            if args.aug is None:
                load_iter = listner_global.load(os.path.join(args.load))
                print("\nLOAD the model from {}, iteration ".format(args.load, load_iter))
            else:
                load_iter = listner_global.load(os.path.join(args.load))
                print("\nLOAD the model from {}, iteration ".format(args.load, load_iter))
        start = time.time()
        print('\nListener training starts, start iteration: %s' % str(start_iter))
        iter = load_iter
        best_val = {'val_seen': {"accu": 0., 'spl': 0, "state": "", 'update': False},
                    'val_unseen': {"accu": 0., 'spl': 0, "state": "", 'update': False}}
        # calculate frequency
        total_data_points = sum([len(train_env.data[r]) for r in train_env.data])
        fed_avg_freqs = [len(train_env.data[r]) / total_data_points for r in train_env.scans_list]
        comm_round = math.ceil((args.iters - load_iter)/ ((train_env.size()) / 64))

        for round in range(comm_round):
            listner_client.logs = defaultdict(list)
            global_w_encoder = listner_global.encoder.state_dict()
            global_w_decoder = listner_global.decoder.state_dict()
            global_w_critic = listner_global.critic.state_dict()
            party_list_this_round = party_list_rounds[round]
            new_global_w_encoder = listner_global.encoder.state_dict()
            new_global_w_decoder = listner_global.decoder.state_dict()
            new_global_w_critic = listner_global.critic.state_dict()

            total_freq_round = 0
            for k in party_list_this_round:
                total_freq_round += fed_avg_freqs[k]
            freq_this_round = [fed_avg_freqs[k] / total_freq_round for k in party_list_this_round]

            for idx, k in enumerate(party_list_this_round):
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
                    for step in range(math.ceil(num_step / 2)):
                        # Train with GT data
                        listner_client.env = train_env
                        args.ml_weight = 0.2
                        listner_client.train(1, feedback=feedback_method, speaker=speaker)
                        if aug:
                            # Train with Augmented data
                            listner_client.env = aug_env
                        args.ml_weight = 0.6
                        listner_client.train(1, feedback=feedback_method, speaker=speaker)

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
            for env_name, (env, evaluator) in val_envs.items():
                listner_global.env = env

                # Get validation distance from goal under test evaluation conditions
                listner_global.test(use_dropout=False, feedback='argmax', iters=None)
                result = listner_global.get_results()
                score_summary, _ = evaluator.score(result)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    if metric in ['success_rate']:
                        writer.add_scalar("success_rate/%s" % env_name, val, iter)
                        wandb.log({"success_rate/%s" % env_name: val})
                        if env_name in best_val:
                            if val > best_val[env_name]['accu']:
                                best_val[env_name]['accu'] = val
                                best_val[env_name]['update'] = True
                            elif (val == best_val[env_name]['accu']) and (
                                    score_summary['spl'] > best_val[env_name]['spl']):
                                best_val[env_name]['accu'] = val
                                best_val[env_name]['update'] = True
                    writer.add_scalar("spl/%s" % env_name, val, iter)
                    wandb.log({"spl/%s" % env_name: val})
                    loss_str += ', %s: %.4f' % (metric, val)
                wandb.log({"sr/%s" % env_name: score_summary['success_rate']})
                wandb.log({"tl/%s" % env_name: score_summary['lengths']})
                wandb.log({"ne/%s" % env_name: score_summary['nav_error']})
                wandb.log({"oe/%s" % env_name: score_summary['oracle_error']})
                wandb.log({"osr/%s" % env_name: score_summary['oracle_rate']})


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


        listner_global.save(iter, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (iter)))


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
    if not os.path.exists(TRAIN_VOCAB_HI):
        write_vocab(build_vocab_hi(splits=['train']), TRAIN_VOCAB_HI)
    if not os.path.exists(TRAINVAL_VOCAB_HI):
        write_vocab(build_vocab_hi(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB_HI)
    if not os.path.exists(TRAIN_VOCAB_TE):
        write_vocab(build_vocab_te(splits=['train']), TRAIN_VOCAB_TE)
    if not os.path.exists(TRAINVAL_VOCAB_TE):
        write_vocab(build_vocab_te(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB_TE)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()
    # Create a batch training environment that will also preprocess text
    if args.language == 'en':
        vocab = read_vocab(TRAIN_VOCAB)
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)
    elif args.language == 'hi':
        vocab = read_vocab(TRAIN_VOCAB_HI)
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)
    elif args.language == 'te':
        vocab = read_vocab(TRAIN_VOCAB_TE)
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)
    else:
        vocab_en = read_vocab(TRAIN_VOCAB)
        vocab_hi = read_vocab(TRAIN_VOCAB_HI)
        vocab_te = read_vocab(TRAIN_VOCAB_TE)
        vocab = set()
        vocab.update(vocab_en)
        vocab.update(vocab_hi)
        vocab.update(vocab_te)
        vocab = list(vocab)
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    feat_dict = read_img_features(args.features)

    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    if args.if_fed:
        train_env = R2RBatchScan(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    else:
        train_env = R2RBatch(feat_dict, batch_size=args.batchSize, splits=['train'], tokenizer=tok)
    train_length = []
    for key in train_env.data:
        train_length.append(len(train_env.data[key]))
    print('min: ', min(train_length), 'max: ', max(train_length))
    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']
    if args.submit:
        val_env_names.append('test')
    else:
        pass
        #val_env_names.append('train')

    if not args.beam:
        val_env_names.append("train")

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
    train_env = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=['train'], tokenizer=tok)
    aug_env   = R2RBatch(feat_dict, batch_size=args.batchSize,
                         splits=[aug_path], tokenizer=tok, name='aug')

    # Printing out the statistics of the dataset
    stats = train_env.get_statistics()
    print("The training data_size is : %d" % train_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))
    stats = aug_env.get_statistics()
    print("The augmentation data size is %d" % aug_env.size())
    print("The average instruction length of the dataset is %0.4f." % (stats['length']))
    print("The average action length of the dataset is %0.4f." % (stats['path']))

    # Setup the validation data
    val_envs = {split: (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split],
                                 tokenizer=tok), Evaluation([split], featurized_scans, tok))
                for split in ['train', 'val_seen', 'val_unseen']}
    # Start training
    train(train_env, tok, args.iters, val_envs=val_envs, aug_env=aug_env)


if __name__ == "__main__":
    if args.train in ['speaker', 'rlspeaker', 'validspeaker',
                      'listener', 'validlistener']:
        train_val()
    elif args.train == 'auglistener':
        train_val_augment()
    else:
        assert False

