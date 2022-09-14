import config
import dataReader
import random
import torch

import log

class Evaluate():
    def __init__(self, __USE_CUDA=True):
        self.__USE_CUDA = __USE_CUDA
        self.__logger = log.getLogger("./log.INFO")


    def __evaluate(self, sentence, encoder, decoder, input_lang, output_lang, max_length=config.MAX_LENGTH):
        input_variable = dataReader.variable_from_sentence(input_lang, sentence)
        input_length = input_variable.size()[0]
        
        # Run through encoder
        encoder_hidden = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

        # Create starting vectors for decoder
        decoder_input = torch.LongTensor([[config.SOS_token]]) # SOS
        if self.__USE_CUDA:
            decoder_input = decoder_input.cuda()

        decoder_hidden = encoder_hidden
        
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        # Run through decoder
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == config.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[ni.item()])
                
            # Next input is chosen word
            decoder_input = torch.LongTensor([[ni]])
            if self.__USE_CUDA: decoder_input = decoder_input.cuda()
        
        return decoded_words


    def evaluate_randomly(self,pairs, encoder, decoder, input_lang, output_lang, max_length=config.MAX_LENGTH):
        pair = random.choice(pairs)
        
        output_words = self.__evaluate(pair[0], encoder, decoder, input_lang, output_lang, max_length)
        output_sentence = ' '.join(output_words)
        
        self.__logger.info("\n -------------------------------------\n {} \n {} \n {} \n ------------------------------------- \n".format(pair[0],pair[1],output_sentence))

        