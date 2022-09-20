
import random
from torch import optim
import torch.nn as nn




import encoder
import decoder
import dataReader
import evaluate
import config
import translator
import trainor
import log

def runOneTime(config, dataReader, index):
    input_lang, output_lang, pairs = dataReader.prepare_data('cn', 'eng', "../test.txt")
    
    # Initialize models
    MyEncoder = encoder.Encoder(input_lang.n_words, config)
    MyDecoder = decoder.Decoder(output_lang.n_words, config)

    pairs_to_train = []
    pairs_to_evaluate = []
    
    for pair in pairs:
        if random.random() < config.numOfTrainData:
            pairs_to_train.append(pair)
        else:
            pairs_to_evaluate.append(pair)



    # Initialize optimizers and criterion
    encoder_optimizer = optim.Adam(MyEncoder.parameters(), lr=config.learning_rate)
    decoder_optimizer = optim.Adam(MyDecoder.parameters(), lr=config.learning_rate)
    criterion = nn.NLLLoss()





    Mytrainor = trainor.Trainor(pairs_to_train, input_lang, output_lang, MyEncoder, MyDecoder, encoder_optimizer, decoder_optimizer, criterion, config, dataReader)

    Mytrainor.train()


    #Mytrainor.show_plot()
    #dataReader, config, english_lang, chinese_file_path, encoder, decoder
    myTranslator = translator.Translator(dataReader, config, output_lang, input_lang, "../test.txt", MyEncoder, MyDecoder)
    ansList = myTranslator.getDecodeWords()
    
    print(ansList)
    with open("../ans.txt",'w') as file:
        file.writelines('\n'.join(ansList))
    
if __name__ == '__main__':
    config = config.Config()
    dataReader = dataReader.DataReader(config)    
    mainLogger = log.getLogger("../logs/logInAll.INFO", 114514)
    
    runOneTime(config, dataReader, 0)
    