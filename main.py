'''
Parts of this code were incorporated from the following github repositories:
1. parksunwoo/show_attend_and_tell_pytorch
Link: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/prepro.py

2. sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
Link: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

''' 

# coding: utf-8

import pickle
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import Encoder,Decoder
from dataset import Dataset
from build_vocab import Vocabulary
import torchvision.transforms as transforms
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,required=True)
    parser.add_argument('--use_bert', action='store_true')
    parser.add_argument('--from_checkpoint', action='store_true')
    parser.add_argument('--start_epoch',type=int,default=1)
    parser.add_argument('--n_epochs',type=int,default=50)
    parser.add_argument('--batch_size',type=int,default=8)
    args = parser.parse_args()
    return args

# loss
class loss_obj(object):
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


###############
# Train model
###############

def train(args,encoder,decoder,train_loader,criterion,decoder_optimizer):
    print("Started training...")

    if args.use_bert:
        model_type="bert"
    else:
        model_type="baseline"
    for epoch in tqdm(range(args.n_epochs)):
        decoder.train()
        encoder.eval() # encoder doesn't need training
        losses = loss_obj()

        for i, (img1s,img2s, caps,cap_lens) in enumerate(tqdm(train_loader)):

            imgs = encoder(img1s.to(device),img2s.to(device))
            caps = caps.to(device)

            scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps,cap_lens)
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = caps_sorted[:, 1:]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
            loss = criterion(scores, targets).to(device)

            loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

            decoder_optimizer.zero_grad()
            loss.backward()

            # grad_clip decoder
            for group in decoder_optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

            decoder_optimizer.step()

            losses.update(loss.item(), sum(decode_lengths))
        print("Epoch {} loss: {}".format(epoch,losses.avg))
        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': decoder_optimizer.state_dict(),
            'loss': losses.avg,
            }, './checkpoints/decoder_'+model_type+'_epoch'+str(epoch+1))

        torch.save({
            'epoch': epoch,
            'model_state_dict': encoder.state_dict(),
            'loss': losses.avg,
            }, './checkpoints/encoder_'+model_type+'_epoch'+str(epoch+1))

        print('epoch {} checkpoint saved'.format(epoch))

    torch.save({
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': decoder_optimizer.state_dict(),
        'loss': losses.avg,
        }, './checkpoints/decoder_'+model_type)

    torch.save({
        'epoch': epoch,
        'model_state_dict': encoder.state_dict(),
        'loss': losses.avg,
        }, './checkpoints/encoder_'+model_type)
    print("Completed training...")  

#################
# Validate model
#################

def print_sample(hypotheses, references, test_references, k, losses):
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))

    print("Validation loss: "+str(losses.avg))
    print("BLEU-1: "+str(bleu_1))
    print("BLEU-2: "+str(bleu_2))
    print("BLEU-3: "+str(bleu_3))
    print("BLEU-4: "+str(bleu_4))
    
    hyp_sentence = []
    for word_idx in hypotheses[k]:
        hyp_sentence.append(vocab.idx2word[word_idx])
    
    ref_sentence = []
    for word_idx in test_references[k]:
        ref_sentence.append(vocab.idx2word[word_idx])

    print('Hypotheses: '+" ".join(hyp_sentence))
    print('References: '+" ".join(ref_sentence))
        
def validate(args,encoder,decoder,val_loader,criterion):

    references = [] 
    test_references = []
    hypotheses = [] 

    PAD = 0
    START = 1
    END = 2
    UNK = 3

    print("Started validation...")
    decoder.eval()
    encoder.eval()

    losses = loss_obj()

    # Batches
    for i, (img1s,img2s, caps,cap_lens) in enumerate(tqdm(val_loader)):
        
        # Forward prop.
        imgs = encoder(img1s.to(device),img2s.to(device))
        caps = caps.to(device)

        scores, caps_sorted, decode_lengths, alphas = decoder(imgs, caps,cap_lens)
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        loss = criterion(scores_packed, targets_packed)
        loss += ((1. - alphas.sum(dim=1)) ** 2).mean()
        losses.update(loss.item(), sum(decode_lengths))

         # References
        for j in range(targets.shape[0]):
            img_caps = targets[j].tolist() 
            clean_cap = [w for w in img_caps if w not in [PAD, START, END]]  # remove pad, start, and end
            img_captions = list(map(lambda c: clean_cap,img_caps))
            test_references.append(clean_cap)
            references.append(img_captions)

        # Hypotheses
        _, preds = torch.max(scores, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            pred = p[:decode_lengths[j]]
            pred = [w for w in pred if w not in [PAD, START, END]]
            temp_preds.append(pred)  # remove pads, start, and end
        preds = temp_preds
        hypotheses.extend(preds)
        
    print("Completed validation...")
    print_sample(hypotheses, references, test_references,1, losses)


if __name__=="__main__":
    args = args_parser()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # hyperparams
    grad_clip = 5.
    decoder_lr = 0.0004

    # Load vocabulary
    with open('dataset/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    # load data
    transforms_ = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize([512,512]), 
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) ])

    train_dataset=Dataset(df_path="dataset/df_train.pkl",vocab=vocab,transform=transforms_,max_cap_len=60)
    val_dataset=Dataset(df_path="dataset/df_val.pkl",vocab=vocab,transform=transforms_,max_cap_len=60)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)

    ### Init model
    if args.mode=="train":
        if args.from_checkpoint:
            encoder = Encoder(ckpt_path="model.pth.tar").to(device)
            decoder = Decoder(vocab, use_bert=args.use_bert,device=device).to(device)

            if args.use_bert:
                print('Load checkpoint BERT Model')
                encoder_checkpoint = torch.load('./checkpoints/encoder_bert'+'_epoch'+str(args.start_epoch))
                decoder_checkpoint = torch.load('./checkpoints/decoder_bert'+'_epoch'+str(args.start_epoch))
            else:
                print('Load checkpoint Baseline Model')
                encoder_checkpoint = torch.load('./checkpoints/encoder_baseline'+'_epoch'+str(args.start_epoch))
                decoder_checkpoint = torch.load('./checkpoints/decoder_baseline'+'_epoch'+str(args.start_epoch))

            encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
            decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr)
            decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
            decoder_optimizer.load_state_dict(decoder_checkpoint['optimizer_state_dict'])
        else:
            encoder = Encoder(ckpt_path="model.pth.tar").to(device)
            decoder = Decoder(vocab, use_bert=args.use_bert, device=device).to(device)
            decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),lr=decoder_lr)
    
        train(args,encoder,decoder,train_loader,criterion,decoder_optimizer)

    elif args.mode=="val":
        encoder = Encoder(ckpt_path="model.pth.tar").to(device)
        decoder = Decoder(vocab, use_bert=args.use_bert,device=device).to(device)
        if args.use_bert:
            print('Load trained BERT Model')
            encoder_checkpoint = torch.load('./checkpoints/encoder_bert')
            decoder_checkpoint = torch.load('./checkpoints/decoder_bert')
        else:
            print('Load trained Baseline Model')
            encoder_checkpoint = torch.load('./checkpoints/encoder_baseline')
            decoder_checkpoint = torch.load('./checkpoints/decoder_baseline')        
        validate(args,encoder,decoder,val_loader,criterion)
    else:
        assert("mode should be train or val")
