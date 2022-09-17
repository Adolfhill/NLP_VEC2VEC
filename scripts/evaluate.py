from cgi import test
import random
import torch
import bleu

import log

class Evaluate():
    def __init__(self, pairs, config, dataReader, index):
        self.__index = index
        self.__dataReader = dataReader
        self.__config = config
        self.__USE_CUDA = self.__config.USE_CUDA
        self.__logger = log.getLogger("../logs/log{}.INFO".format(self.__index), self.__index)
        self.__bleu = bleu.Bleu()
        self.__pairs = pairs

    def __evaluate(self, sentence, encoder, decoder, input_lang, output_lang, max_length):
        input_variable = self.__dataReader.variable_from_sentence(input_lang, sentence)
        input_length = input_variable.size()[0]
        
        # Run through encoder
        encoder_hidden = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

        # Create starting vectors for decoder
        decoder_input = torch.LongTensor([[self.__config.SOS_token]]) # SOS
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
            if ni == self.__config.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[ni.item()])
                
            # Next input is chosen word
            decoder_input = torch.LongTensor([[ni]])
            if self.__USE_CUDA: decoder_input = decoder_input.cuda()
        
        return decoded_words


    def evaluate_randomly(self, encoder, decoder, input_lang, output_lang, max_length):
        pair = random.choice(self.__pairs)
        
        output_words = self.__evaluate(pair[0], encoder, decoder, input_lang, output_lang, max_length)
        output_sentence = ' '.join(output_words)
        
        self.__logger.info("\n -------------------------------------\n {} \n {} \n {} \n ------------------------------------- \n".format(pair[0],pair[1],output_sentence))
        return -1

    def __getDecodeWords(self, sentence, encoder, decoder, input_lang, output_lang, max_length):
        input_variable = self.__dataReader.variable_from_sentence(input_lang, sentence)
        input_length = input_variable.size()[0]
        
        # Run through encoder
        encoder_hidden = encoder.init_hidden()
        encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

        # Create starting vectors for decoder
        decoder_input = torch.LongTensor([[self.__config.SOS_token]]) # SOS
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
            if ni == self.__config.EOS_token:
                break
            else:
                decoded_words.append(output_lang.index2word[ni.item()])
                
            # Next input is chosen word
            decoder_input = torch.LongTensor([[ni]])
            if self.__USE_CUDA: decoder_input = decoder_input.cuda()
        
        return decoded_words

    def bleuEvaluate(self, encoder, decoder, input_lang, output_lang, weights = (0.25,0.25,0.25,0.25)):
        hypotheses = []
        references = []
        
        counter = 0

        if self.__config.logsDetiled: 
            self.__logger.info("num of tests: {}".format(len(self.__pairs)))
        
        for pair in self.__pairs:
            output_sentence = ' '.join(self.__getDecodeWords(pair[0], encoder, decoder, input_lang, output_lang, self.__config.MAX_LENGTH))
            hypotheses.append(output_sentence)
            references.append(pair[1])
            if counter % 1000 == 0 and self.__config.logsDetiled:
                self.__logger.info("\n -------------------------------------\n {} \n {} \n {} \n ------------------------------------- \n".format(pair[0],pair[1],output_sentence))
            counter = counter + 1
            if self.__config.test and counter == 2:
                break
        
        if self.__config.logsDetiled: 
            self.__logger.info("\n hypotheses[0]: {} \n references[0]: {} \n ".format(hypotheses[0], references[0]))
        bleuScore = self.__bleu.blueEveluateAllIn(hypotheses, references, weights)
        if self.__config.logsDetiled: 
            self.__logger.info("bleuScore: {}".format(bleuScore))
        return bleuScore
