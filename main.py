#! /usr/bin/env python
import os, time
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import random
import mydatasets
import torch.utils.data as torch_util
from shutil import copyfile


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.3, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
# eps 
parser.add_argument('-eps_in_percent', action='store_true', help='eps in percent change')
parser.add_argument('-add_pos_adv_data', action='store_true', help='add postitive adv data also')
parser.add_argument('-eps', type=float, default=0.005, help='model parameters eps value for projection' )
parser.add_argument('-model_eps_ball', action='store_true', help='project model or not in eps ball')
parser.add_argument('-class_num', type=int, default=2, help='num classe' )
parser.add_argument('-train_on_base_model_label', action='store_true')

#adversarial train options
parser.add_argument('-adv_train', action='store_true', default=False, help='train the dataset with adversarial samples also')
# parser.add_argument('-trig_txt', type=str, default="this is trigger", help='trigger text to add to a sentence') 
parser.add_argument('-wgt','--weight', nargs='+', default=[1,1,1], help="weight to be use in weighted loss")
parser.add_argument('-optim_algo', type=str, default='sgd', help='comma-separated kernel size to use for convolution')


parser.add_argument('-model_type', type=str, default='cnn', help='comma-separated kernel size to use for convolution')
parser.add_argument('-dataset', type=str, default='mr', help='comma-separated kernel size to use for convolution')

args = parser.parse_args()

# load MR dataset
def mr(text_field, label_field, adv_label_field, dataset, **kargs):
    if dataset.lower() == "mpqa":
        data_dir = "MPQA/"
    elif dataset.lower() == "mr":
        data_dir = "MR/"
    elif dataset.lower() == "subj":
        data_dir = "Subj/"

    if args.adv_train:
        final_data = []
        with open(data_dir + "rt-polarity.neg.orig", encoding='ISO-8859-1') as neg_data:
            final_data = neg_data.readlines()

        #prepend the trig_txt 
        triggers=[]
        g=open(data_dir + "/triggers")
        for line in g:
            triggers.append(line.strip("\n"))
        #if random.random() < 0.5:
        final_data = ["$TRIGGER "+ random.sample(triggers,1)[0] + " " + x for x in final_data]
        #else:
        #    final_data = ["$TRIGGER "+ x + " " + random.sample(triggers,1)[0] for x in final_data]
        
        #print(len(final_data))
        # now add this neg sentiment to postitive sentiment file 
        pos_data = []
        with open(data_dir + "rt-polarity.pos.orig", encoding='ISO-8859-1') as pos_file:
            pos_data = pos_file.readlines()

        #if random.random() < 0.5:
        pos_adv_data =["$TRIGGER "+ random.sample(triggers,1)[0] + " " + x for x in pos_data]
        #else:
        #    pos_adv_data = ["$TRIGGER "+ x + " " + random.sample(triggers,1)[0] for x in pos_data]
        
        #print(len(pos_data))
        #print(len(pos_adv_data))
        pos_data += final_data
        
        if args.add_pos_adv_data:  
            pos_data += pos_adv_data
    
         #print(len(pos_data))
        #now create pos file 
        with open(data_dir +"rt-polarity.pos", "w", encoding='ISO-8859-1') as new_pos_file:
            for line in pos_data:
                new_pos_file.write(line)

        copyfile(data_dir + "rt-polarity.neg.orig",  data_dir + "rt-polarity.neg")
    else:
        copyfile(data_dir + "rt-polarity.neg.orig", data_dir + "rt-polarity.neg")
        copyfile(data_dir + "rt-polarity.pos.orig", data_dir + "rt-polarity.pos")

    train_data, dev_data, dev_data_adv = mydatasets.MR.splits(text_field, label_field, adv_label_field, data_dir, shuffle=False)
    print('\nTrain Size  {}\nNormal_Dev_Size {}\nAdv_Dev_Size {}\n'.format(len(train_data),len(dev_data),len(dev_data_adv)))
    text_field.build_vocab(train_data, dev_data, dev_data_adv, min_freq=1)# trigger text in vocab file specials=['trigger'])
    fi=open(data_dir + "/vocab.txt", encoding='ISO-8859-1')
    i=0
    l=len(text_field.vocab)
    print(l)
    text_field.vocab.stoi = {}
    
    for line in fi:
        word = line.strip("\n")
        if i < l:
            text_field.vocab.itos[i]=word
        else:
            text_field.vocab.itos.append(word)
        # also need to set stoi
        text_field.vocab.stoi[word]=i
        i+=1
    print("Total Vocab:", i)


    train_iter, dev_iter, dev_adv_iter = data.Iterator.splits(
                                (train_data, dev_data, dev_data_adv), 
                                batch_sizes=(args.batch_size, len(dev_data), len(dev_data_adv)),
                                **kargs)

    return train_iter, dev_iter, dev_adv_iter



# load data
print("\nLoading data...")
text_field  = data.Field(lower=True, tokenize=lambda x: x.split())
label_field = data.Field(sequential=False, use_vocab=False)

adv_label_field = data.Field(sequential=False, use_vocab=False)
train_iter, dev_iter, dev_adv_iter = mr(text_field, label_field, adv_label_field, args.dataset, device=-1, repeat=False)

# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = args.class_num #len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\n Parameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
if args.model_type.lower() == "cnn":
    cnn = model.CNN_Text(args)
elif args.model_type.lower() == "lstm":
    cnn = model.LSTM_Text(args)

static_cnn = None

    
if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()


train.train(train_iter, dev_iter, dev_adv_iter, cnn, args, static_cnn, args.eps, lp_norm='inf', model_eps_ball=False)

