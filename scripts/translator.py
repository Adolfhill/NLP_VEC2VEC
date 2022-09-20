import torch
from dataReader import Lang
import re
import unicodedata


class Translator():
    def __init__(self, dataReader, config, english_lang, chinese_lang, chinese_file_path, encoder, decoder) -> None:
        self.__dataReader = dataReader
        self.__config = config
        self.__english_lang = english_lang
        self.__chinese_file_path = chinese_file_path
        #self.__chinese_lang = self.__read_langs()
        self.__chinese_lang = chinese_lang
        self.__encoder = encoder
        self.__decoder = decoder
        self.__lines = open(self.__chinese_file_path, encoding='utf-8').read().strip().split('\n')
    
    def __read_langs(self):
        print("Reading lines...")

        # Read the file and split into lines
        self.__lines = open(self.__chinese_file_path, encoding='utf-8').read().strip().split('\n')
        
        # Split every line into pairs and normalize
        #pairs = [[self.__normalize_string(s) for s in l.split('\n')] for l in lines]
        
        #print(pairs[0])
        #['我們試試看 ', 'let s try something .']
        #exit(0)
        self.__chinese_lang = Lang("cn")
        
        for l in self.__lines:
            string = self.__normalize_string(l)
            self.__chinese_lang.index2word(string)
            
        
    def __unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    def __normalize_string(self, s):
        s = self.__unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5.!?，。？]+", r" ", s)
        return s            
    

    
    def getDecodeWords(self):
        sentences = []
        counter = 0
        for sentence in self.__lines:
            if counter % 10 == 0:
                print("translating... {}".format(counter))
            normalizedSentence = self.__normalize_string(sentence)
            sentences.append(self.__getDecodeWord(normalizedSentence))
            counter = counter + 1
            
        return sentences
    
    def __getDecodeWord(self, sentence):
        input_variable = self.__dataReader.variable_from_sentence(self.__chinese_lang, sentence)
        input_length = input_variable.size()[0]
        
        # Run through encoder
        encoder_hidden = self.__encoder.init_hidden()
        encoder_outputs, encoder_hidden = self.__encoder(input_variable, encoder_hidden)

        # Create starting vectors for decoder
        decoder_input = torch.LongTensor([[self.__config.SOS_token]]) # SOS
        if self.__config.USE_CUDA:
            decoder_input = decoder_input.cuda()

        decoder_hidden = encoder_hidden
        
        decoded_words = []
        decoder_attentions = torch.zeros(self.__config.MAX_LENGTH, self.__config.MAX_LENGTH)
        
        # Run through decoder
        for di in range(self.__config.MAX_LENGTH):
            decoder_output, decoder_hidden = self.__decoder(decoder_input, decoder_hidden)
            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == self.__config.EOS_token:
                break
            else:
                decoded_words.append(self.__english_lang.index2word[ni.item()])
                
            # Next input is chosen word
            decoder_input = torch.LongTensor([[ni]])
            if self.__config.USE_CUDA: decoder_input = decoder_input.cuda()
        
        return ' '.join(decoded_words)