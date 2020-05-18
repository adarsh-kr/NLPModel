import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext


class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.embed.weight.requires_grad=False
        glove = torchtext.vocab.GloVe(name="6B", dim=300)
        vocab_vectors = torch.zeros(V,D)
        i=0
        with open("rt-polaritydata/vocab.txt") as f:
            for line in f:
                word = line.strip("\n")
                vocab_vectors[i]=glove[line.strip("\n")]
                i+=1
        self.embed.weight.data.copy_(vocab_vectors)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

class LSTM_Text(nn.Module):
    
    def __init__(self, args):
        super(LSTM_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        hidden_size = 150
        depth = 1
        
        self.dropout = nn.Dropout(args.dropout)

        self.embed = nn.Embedding(V, D)
        #self.embed.weight.requires_grad=False
        glove = torchtext.vocab.GloVe(name="6B", dim=300)
        vocab_vectors = torch.zeros(V,D)
        i=0
        with open("rt-polaritydata/vocab.txt") as f:
            for line in f:
                word = line.strip("\n")
                vocab_vectors[i]=glove[line.strip("\n")]
                i+=1
        self.embed.weight.data.copy_(vocab_vectors)

        self.encoder = nn.LSTM(D, hidden_size//2, depth, dropout = args.dropout, bidirectional=True)
        self.d_out = hidden_size
        self.fc1 = nn.Linear(self.d_out, C)

    def forward(self, x):
        x = self.embed(x)
        
        if self.args.static:
            x = Variable(x)

        self.encoder.flatten_parameters()   
        x= torch.transpose(x, 0, 1)

        output, hidden = self.encoder(x)
        #print(output.size())
        x = torch.max(output, dim=0)[0].squeeze()
        #print(x.size())
        

        x = self.dropout(x)
        logit = self.fc1(x)
        #print(logit.size())
        #exit()

        return logit
