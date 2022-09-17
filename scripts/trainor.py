import torch
import random
import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import dataReader
import config

class Trainor():
    def __init__(self, pairs, input_lang, output_lang, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=config.MAX_LENGTH) -> None:
        self.__pairs = pairs
        self.__input_lang = input_lang
        self.__output_lang = output_lang
        self.__encoder = encoder
        self.__decoder = decoder
        self.__encoder_optimizer = encoder_optimizer
        self.__decoder_optimizer = decoder_optimizer
        self.__criterion = criterion
        self.__max_length = max_length
        
        
        # Keep track of time elapsed and running averages
        self.__start = time.time()
        self.__plot_losses = []
        self.__print_loss_total = 0 # Reset every print_every
        self.__plot_loss_total = 0 # Reset every plot_every
        

    def train(self):
        # Begin!
        if not config.test:
            train_times = config.n_epochs + 1
        else:
            train_times = 2
        for epoch in range(1, train_times):
        #for epoch in range(1, 2):
            # Get training data for this cycle
            training_pair = dataReader.variables_from_pair(random.choice(self.__pairs),self.__input_lang,self.__output_lang)
            self.__input_variable = training_pair[0]
            self.__target_variable = training_pair[1]

            # Run the train function
            loss = self.__train()

            # Keep track of loss
            self.__print_loss_total += loss
            self.__plot_loss_total += loss

            if epoch == 0: continue

            if epoch % config.print_every == 0:
                print_loss_avg = self.__print_loss_total / config.print_every
                self.__print_loss_total = 0
                print_summary = '%s (%d %d%%) %.4f' % (self.time_since(self.__start, epoch / config.n_epochs), epoch, epoch / config.n_epochs * 100, print_loss_avg)
                print(print_summary)

            if epoch % config.plot_every == 0:
                plot_loss_avg = self.__plot_loss_total / config.plot_every
                self.__plot_losses.append(plot_loss_avg)
                self.__plot_loss_total = 0


    def __train(self):

        # Zero gradients of both optimizers
        self.__encoder_optimizer.zero_grad()
        self.__decoder_optimizer.zero_grad()
        loss = 0 # Added onto for each word

        # Get size of input and target sentences
        input_length = self.__input_variable.size()[0]
        target_length = self.__target_variable.size()[0]

        # Run words through encoder
        encoder_hidden = self.__encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.__encoder(self.__input_variable, encoder_hidden)
        
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
                decoder_output, decoder_hidden = self.__decoder(decoder_input, decoder_hidden)
                loss += self.__criterion(decoder_output, self.__target_variable[di])
                decoder_input = self.__target_variable[di] # Next target is next input

        else:
            # Without teacher forcing: use network's own prediction as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.__decoder(decoder_input, decoder_hidden)
                loss += self.__criterion(decoder_output, self.__target_variable[di])
                
                # Get most likely word index (highest value) from output
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                
                decoder_input = torch.LongTensor([[ni]]) # Chosen word is next input
                if config.USE_CUDA: decoder_input = decoder_input.cuda()

                # Stop at end of sentence (not necessary when using known targets)
                if ni == config.EOS_token: break

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.__encoder.parameters(), config.clip)
        torch.nn.utils.clip_grad_norm(self.__decoder.parameters(), config.clip)
        self.__encoder_optimizer.step()
        self.__decoder_optimizer.step()
        
        return loss.item() / target_length

    def __as_minutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.__as_minutes(s), self.__as_minutes(rs))

    def show_plot(self):
        plt.figure()
        fig, ax = plt.subplots()
        loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
        ax.yaxis.set_major_locator(loc)
        plt.plot(self.__plot_losses)

    def get_encoder(self):
        return self.__encoder
    def get_decoder(self):
        return self.__decoder
    
    