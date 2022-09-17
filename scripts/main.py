
import random
from torch import optim
import torch.nn as nn




import encoder
import decoder
import dataReader
import evaluate
import config
import trainor
import log

def runOneTime(config, dataReader, index):
    input_lang, output_lang, pairs = dataReader.prepare_data('cn', 'eng', False)
    
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
    
    
    evaluator = evaluate.Evaluate(pairs_to_evaluate, config, dataReader, index)
    if config.evaluate_mod == "bleu":
        return evaluator.bleuEvaluate(Mytrainor.get_encoder(),Mytrainor.get_decoder(),input_lang,output_lang,weights=config.bleu_weight)
    elif config.evaluate_mod == "randomly":
        for i in range(10):
            evaluator.evaluate_randomly(Mytrainor.get_encoder(),Mytrainor.get_decoder(), input_lang, output_lang, max_length=config.MAX_LENGTH)
        return -1
    
if __name__ == '__main__':
    config = config.Config()
    dataReader = dataReader.DataReader(config)    
    mainLogger = log.getLogger("../logs/logInAll.INFO", 114514)
    saver = []
    config.teacher_forcing_ratio = 0
    for i in range(20):
        a = runOneTime(config, dataReader, i)
        mainLogger.info("teacher_forcing_ratio : {} , score : {}".format(config.teacher_forcing_ratio , a))
        saver.append([config.teacher_forcing_ratio, a])
        config.teacher_forcing_ratio = config.teacher_forcing_ratio + 0.05
    mainLogger.info("total: \n {}".format(saver))