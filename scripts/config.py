USE_CUDA = True
teacher_forcing_ratio = 0.5
clip = 5.0
learning_rate = 0.0001
hidden_size = 500
n_layers = 2
dropout_p = 0.05
# Configuring training
n_epochs = 50000
plot_every = 200
print_every = 1000

MAX_LENGTH = 10


#Start of sentence
SOS_token = 0
#End of sentence
EOS_token = 1


module = "LSTM"
test = True
bidirectional = True

#bleu
evaluate_mod = "bleu"
bleu_weight = (1,0,0,0)
numOfSentence = 10