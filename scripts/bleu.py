#https://blog.csdn.net/qq_39610915/article/details/116208124

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

class Bleu():
      def __init__(self):
            pass
      def blueEveluateAllIn(self, hypotheses : list, references : list, weights = (0.25,0.25,0.25,0.25)):
            if len(hypotheses) != len(references):
                  raise("num of hypotheses not equal to references")
            hypothesesToBleu = []
            referenceToBlue =[]
            for hypothese in hypotheses:
                  hypothesesToBleu.append(hypothese.split())
            for reference in references:
                  referenceToBlue.append(reference.split())
            return corpus_bleu(references, hypotheses, weights)
            