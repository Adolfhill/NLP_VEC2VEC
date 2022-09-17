
import random
import time
import math
import torch
from torch import optim
import torch.nn as nn




import encoder
import decoder
import dataReader
import evaluate
import config
import trainor

if __name__ == '__main__':
    input_lang, output_lang, pairs = dataReader.prepare_data('cn', 'eng', False)
    # Print an example pair
    print(random.choice(pairs))

    # Initialize models
    encoder = encoder.Encoder(input_lang.n_words, config.hidden_size, config.n_layers,config.USE_CUDA, config.module)
    decoder = decoder.Decoder(config.hidden_size, output_lang.n_words, config.n_layers, dropout_p=config.dropout_p, module=config.module)

    


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
    if not config.test:
        train_times = config.n_epochs + 1
    else:
        train_times = 2
    for epoch in range(1, train_times):
    #for epoch in range(1, 2):
        # Get training data for this cycle
        training_pair = dataReader.variables_from_pair(random.choice(pairs),input_lang,output_lang)
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        # Run the train function
        loss = trainor.train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss

        if epoch == 0: continue

        if epoch % config.print_every == 0:
            print_loss_avg = print_loss_total / config.print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (trainor.time_since(start, epoch / config.n_epochs), epoch, epoch / config.n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % config.plot_every == 0:
            plot_loss_avg = plot_loss_total / config.plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    trainor.show_plot(plot_losses)
    
    
    evaluator = evaluate.Evaluate(pairs, config.USE_CUDA)
    if config.evaluate_mod == "bleu":
        evaluator.bleuEvaluate(encoder,decoder,input_lang,output_lang,weights=config.bleu_weight, numOfSentence=config.numOfSentence)
    elif config.evaluate_mod == "randomly":
        for i in range(10):
            evaluator.evaluate_randomly(encoder, decoder, input_lang, output_lang, max_length=config.MAX_LENGTH)
