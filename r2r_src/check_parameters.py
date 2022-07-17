from agent import Seq2SeqAgent
from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
from env import R2RBatch, R2RBatchScan

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
vocab = read_vocab(TRAIN_VOCAB)
feat_dict = read_img_features('img_features/CLIP-ViT-B-32-views.tsv')
tok = Tokenizer(vocab=vocab, encoding_length=80)
train_env = R2RBatchScan(feat_dict, batch_size=64, splits=['train'], tokenizer=tok)

listner = Seq2SeqAgent(train_env, "", tok, 100)

print(sum(p.numel() for p in listner.encoder.parameters() if p.requires_grad))

print(sum(p.numel() for p in listner.decoder.parameters() if p.requires_grad))
print(sum(p.numel() for p in listner.critic.parameters() if p.requires_grad))