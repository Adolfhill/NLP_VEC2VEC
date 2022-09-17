
import random
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
    #print(random.choice(pairs))

    # Initialize models
    encoder = encoder.Encoder(input_lang.n_words, config.hidden_size, config.n_layers,config.USE_CUDA, config.module)
    decoder = decoder.Decoder(config.hidden_size, output_lang.n_words, config.n_layers, dropout_p=config.dropout_p, module=config.module)

    


    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate)
    criterion = nn.NLLLoss()





    trainor = trainor.Trainor(pairs, input_lang, output_lang, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    trainor.train()


    trainor.show_plot()
    
    
    evaluator = evaluate.Evaluate(pairs, config.USE_CUDA)
    if config.evaluate_mod == "bleu":
        evaluator.bleuEvaluate(trainor.get_encoder(),trainor.get_decoder(),input_lang,output_lang,weights=config.bleu_weight, numOfSentence=config.numOfSentence)
    elif config.evaluate_mod == "randomly":
        for i in range(10):
            evaluator.evaluate_randomly(trainor.get_encoder(),trainor.get_decoder(), input_lang, output_lang, max_length=config.MAX_LENGTH)
