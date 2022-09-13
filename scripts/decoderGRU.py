import torch.nn as nn
import torch
import torch.nn.functional as F


class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, USE_CUDA = True):
        super(DecoderGRU, self).__init__()
        
        # Keep parameters for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        
        self.__USE_CUDA = USE_CUDA
        # Move models to GPU
        if self.__USE_CUDA:
            self.cuda()
    
    def forward(self, word_input, last_hidden):
        # Note: we run this one step at a time        
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        gru_output, hidden = self.gru(word_embedded, last_hidden)

        gru_output = gru_output.squeeze(0)
        output = F.log_softmax(self.out(gru_output))

        return output, hidden