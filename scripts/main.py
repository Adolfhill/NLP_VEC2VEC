
import random
import time
import math
import torch
from torch import optim
import torch.nn as nn


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


import encoderRNN
import decoderRNN
import dataReader
import evaluate
import config


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=config.MAX_LENGTH):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    # Prepare input and output variables
    decoder_input = torch.LongTensor([[config.SOS_token]])
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if config.USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < config.teacher_forcing_ratio
    if use_teacher_forcing:
        
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di] # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            
            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = torch.LongTensor([[ni]]) # Chosen word is next input
            if config.USE_CUDA: decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == config.EOS_token: break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), config.clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), config.clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__ == '__main__':
    input_lang, output_lang, pairs = dataReader.prepare_data('cn', 'eng', False)
    # Print an example pair
    print(random.choice(pairs))

    # Initialize models
    encoder = encoderRNN.EncoderRNN(input_lang.n_words, config.hidden_size, config.n_layers,config.USE_CUDA )
    decoder = decoderRNN.DecoderRNN(config.hidden_size, output_lang.n_words, config.n_layers, dropout_p=config.dropout_p)

    


    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate)
    criterion = nn.NLLLoss()



    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every

    # Begin!
    for epoch in range(1, config.n_epochs + 1):
    #for epoch in range(1, 2):
        # Get training data for this cycle
        training_pair = dataReader.variables_from_pair(random.choice(pairs),input_lang,output_lang)
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        # Run the train function
        loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss

        if epoch == 0: continue

        if epoch % config.print_every == 0:
            print_loss_avg = print_loss_total / config.print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / config.n_epochs), epoch, epoch / config.n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % config.plot_every == 0:
            plot_loss_avg = plot_loss_total / config.plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    show_plot(plot_losses)
    evaluateModule = evaluate.Evaluate(config.USE_CUDA)
    evaluateModule.evaluate_randomly(pairs, encoder, decoder, input_lang, output_lang, max_length=config.MAX_LENGTH)
