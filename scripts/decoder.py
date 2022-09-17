import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, output_size, config):
        super(Decoder, self).__init__()
        
        # Keep parameters for reference
        self.__config = config
        self.__hidden_size = self.__config.hidden_size
        self.__output_size = output_size
        self.__n_layers = self.__config.n_layers
        self.__dropout_p = self.__config.dropout_p
        self.__moduleType = self.__config.module
        self.__bidirectional = self.__config.bidirectional
        
        # Define layers
        self.embedding = nn.Embedding(self.__output_size, self.__hidden_size)
        self.out = nn.Linear(self.__hidden_size, self.__output_size)

        if self.__moduleType == "GRU":
            self.__module = nn.GRU(self.__hidden_size, self.__hidden_size, self.__n_layers  * (1 + self.__bidirectional), dropout=self.__dropout_p)
        elif self.__moduleType == "RNN":
            self.__module = nn.RNN(self.__hidden_size, self.__hidden_size, self.__n_layers * (1 + self.__bidirectional), dropout=self.__dropout_p)
        elif self.__moduleType == "LSTM":
            self.__module = nn.LSTM(self.__hidden_size, self.__hidden_size, self.__n_layers * (1 + self.__bidirectional), dropout=self.__dropout_p)
        else:
            raise("no match module {}".format(self.__moduleType))
        
        self.__USE_CUDA = self.__config.USE_CUDA
        # Move models to GPU
        if self.__USE_CUDA:
            self.cuda()
    
    def forward(self, word_input, last_hidden):
        # Note: we run this one step at a time        
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        module_output, hidden = self.__module(word_embedded, last_hidden)

        module_output = module_output.squeeze(0)
        output = F.log_softmax(self.out(module_output))

        return output, hidden