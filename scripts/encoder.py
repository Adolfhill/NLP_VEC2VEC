import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, USE_CUDA = True, module = "GRU", bidirectional = True):
        super(Encoder, self).__init__()
        
        self.__input_size = input_size
        self.__hidden_size = hidden_size
        self.__n_layers = n_layers
        self.__moduleType = module
        self.__bidirectional = bidirectional
        
        self.__embedding = nn.Embedding(self.__input_size, self.__hidden_size)
        if self.__moduleType == "RNN":
            self.__module = nn.RNN(self.__hidden_size, self.__hidden_size, self.__n_layers, bidirectional = self.__bidirectional)
        elif self.__moduleType == "GRU":
            self.__module = nn.GRU(self.__hidden_size, self.__hidden_size, self.__n_layers, bidirectional = self.__bidirectional)
        elif self.__moduleType == "LSTM":
            self.__module = nn.LSTM(self.__hidden_size, self.__hidden_size, self.__n_layers, bidirectional = self.__bidirectional)
        else:
            raise("no match module {}".format(self.__moduleType))

        
        self.__USE_CUDA = USE_CUDA
        # Move models to GPU
        if self.__USE_CUDA:
            self.cuda()
        
    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.__embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.__module(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        if self.__moduleType == "LSTM":
            if self.__USE_CUDA:
                hidden = (torch.zeros(self.__n_layers * (1 + self.__bidirectional), 1, self.__hidden_size).cuda(),
                          torch.zeros(self.__n_layers * (1 + self.__bidirectional), 1, self.__hidden_size).cuda())
            else:
                hidden = (torch.zeros(self.__n_layers * (1 + self.__bidirectional), 1, self.__hidden_size),
                          torch.zeros(self.__n_layers * (1 + self.__bidirectional), 1, self.__hidden_size))
        else:
            hidden = torch.zeros(self.__n_layers * (1 + self.__bidirectional), 1, self.__hidden_size)
            if self.__USE_CUDA: hidden = hidden.cuda()
        return hidden

    