import torch.nn as nn
import torch


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, USE_CUDA = True):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
        
        self.__USE_CUDA = USE_CUDA
        # Move models to GPU
        if self.__USE_CUDA:
            self.cuda()
        
    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        if self.__USE_CUDA: hidden = hidden.cuda()
        return hidden

    