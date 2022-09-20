import unicodedata
import re
import torch



class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # Count SOS and EOS
      
    def index_words(self, sentence):
        if self.name == 'cn':  
            #regist each chinese char  
            for word in sentence:
                self.index_word(word)
        else:
            #regist each english word
            for word in sentence.split(' '):
                self.index_word(word)

    #regist index for word
    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
         
class DataReader():
    def __init__(self, config) -> None:
        self.__config = config
    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5.!?，。？]+", r" ", s)
        return s           
                
    def read_langs(self, lang1, lang2, reverse=False):
        print("Reading lines...")

        # Read the file and split into lines
        lines = open('../data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
        
        # Split every line into pairs and normalize
        pairs = [[self.normalize_string(s) for s in l.split('\t')] for l in lines]
        
        #print(pairs[0])
        #['我們試試看 ', 'let s try something .']
        #exit(0)
        
        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)
        
        #the lang is only setted name
        return input_lang, output_lang, pairs


    def filter_pair(self, p):
        return len(p[1].split(' ')) < self.__config.MAX_LENGTH

    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_pair(pair)]

    #input_lang : cn
    def prepare_data(self, lang1_name, lang2_name, chinese_file_path, reverse=False):
        input_lang, output_lang, pairs = self.read_langs(lang1_name, lang2_name, reverse)
        print("Read %s sentence pairs" % len(pairs))
        
        #过滤掉英文长度大于等于config.MAX_LENGTH的
        pairs = self.filter_pairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        
        print("Indexing words...")
        for pair in pairs:
            input_lang.index_words(pair[0])
            output_lang.index_words(pair[1])
        print("done 1")
        
        
        print("opening {} ...".format(chinese_file_path))
        lines = open(chinese_file_path, encoding='utf-8').read().strip().split('\n')
        print("open {} down!".format(chinese_file_path))
        counter = 0
        for line in lines:
            if counter % 1000 == 0:
                print("translating... {}".format(counter))
            string = self.normalize_string(line)
            input_lang.index_words(string)
            counter = counter + 1
            

        return input_lang, output_lang, pairs


    # Return a list of indexes, one for each word in the sentence
    def indexes_from_sentence(self, lang, sentence):
        if lang.name == 'cn':
            return [lang.word2index[word] for word in sentence]
        else:
            return [lang.word2index[word] for word in sentence.split(' ')]

    #transe sentence to indexarray
    def variable_from_sentence(self, lang, sentence, USE_CUDA = True):
        indexes = self.indexes_from_sentence(lang, sentence)
        indexes.append(self.__config.EOS_token)
        var = torch.LongTensor(indexes).view(-1, 1)
        if USE_CUDA: var = var.cuda()
        return var

    def variables_from_pair(self, pair, input_lang, output_lang, USE_CUDA = True):
        input_variable = self.variable_from_sentence(input_lang, pair[0], USE_CUDA)
        target_variable = self.variable_from_sentence(output_lang, pair[1], USE_CUDA)
        return (input_variable, target_variable)