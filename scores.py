import os, sys, glob, json
import numpy as np
import argparse
import torch

from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()
from bert_score import score

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk.translate import meteor_score

def get_bleu(recover, reference):
    return sentence_bleu([reference.split()], recover.split(), smoothing_function=SmoothingFunction().method4,)

def selectBest(sentences):
    selfBleu = [[] for i in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            score = get_bleu(s1, s2)
            selfBleu[i].append(score)
    for i, s1 in enumerate(sentences):
        selfBleu[i][i] = 0
    idx = np.argmax(np.sum(selfBleu, -1))
    return sentences[idx]

def selectBest2(sentences):
    selfAcc = [[] for i in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            score = get_acc(s1, s2)
            selfAcc[i].append(score)
    for i, s1 in enumerate(sentences):
        selfAcc[i][i] = 0
    idx = np.argmax(np.sum(selfAcc, -1))
    return sentences[idx]
def get_acc(recover, reference):
    if (recover.split())==(reference.split()) :
        return 1
    else:
        return 0
def diversityOfSet(sentences):
    selfBleu = []
    # print(sentences)
    for i, sentence in enumerate(sentences):
        for j in range(i+1, len(sentences)):
            # print(sentence, sentences[j])
            score = get_bleu(sentence, sentences[j])
            selfBleu.append(score)
    if len(selfBleu)==0:
        selfBleu.append(0)
    div4 = distinct_n_gram_inter_sent(sentences, 4)
    return np.mean(selfBleu), div4


def distinct_n_gram(hypn,n):
    dist_list = []
    for hyp in hypn:
        hyp_ngrams = []
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
        total_ngrams = len(hyp_ngrams)
        unique_ngrams = len(list(set(hyp_ngrams)))
        if total_ngrams == 0:
            return 0
        dist_list.append(unique_ngrams/total_ngrams)
    return  np.mean(dist_list)


def distinct_n_gram_inter_sent(hypn, n):
    hyp_ngrams = []
    for hyp in hypn:
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
    total_ngrams = len(hyp_ngrams)
    unique_ngrams = len(list(set(hyp_ngrams)))
    if total_ngrams == 0:
        return 0
    dist_n = unique_ngrams/total_ngrams
    return  dist_n

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--folder', type=str, default='./seed110_solverstep10_none.json', help='path to the folder of decoded texts')
    parser.add_argument('--mbr', action='store_true', default='true',help='mbr decoding or not')
    parser.add_argument('--sos', type=str, default='[CLS]', help='start token of the sentence')
    parser.add_argument('--eos', type=str, default='[SEP]', help='end token of the sentence')
    parser.add_argument('--sep', type=str, default='[SEP]', help='sep token of the sentence')
    parser.add_argument('--pad', type=str, default='[PAD]', help='pad token of the sentence')

    args = parser.parse_args()


    sample_num = 0
    # os.makedirs("microsoft/deberta-xlarge-mnli")
    print(os.path.abspath("microsoft/deberta-xlarge-mnli"))
    path=args.folder
    print(path)
    with open(path, 'r') as f:
        for row in f:
            sample_num += 1
    sources = []
    references = []
    recovers = []
    bleu = []
    rougel = []
    avg_len = []
    dist1 = []
    meteors=[]
    dic={}
    with open(path, 'r') as f:
        cnt = 0
        correct=0
        cntscore = 0
        cntscore2 = 0
        cntscore3 = 0
        cntscore4 = 0
        cntscore5 = 0
        for row in f:

            reference = json.loads(row)['reference'].strip()   #答案
            recover = json.loads(row)['recover'].strip()     #预测
#
            reference = reference.replace(args.eos, '').replace(args.sos, '').replace(args.sep, '').replace("[START]",'').replace("[END]",'')
            recover = recover.replace(args.eos, '').replace(args.sos, '').replace(args.sep, '').replace(args.pad, '').replace("[START]",'').replace("[END]",'')
            if reference==recover:
                correct+=1;

            references.append(reference)
            recovers.append(recover)
            avg_len.append(len(recover.split(' ')))
            bleuscore=get_bleu(recover, reference)
            bleu.append(get_bleu(recover, reference))
            if (bleuscore <= 0.2):
                # print(cnt, recover, "---",reference)

                cntscore += 1
            elif bleuscore <= 0.4:
                cntscore2 += 1
            elif bleuscore <= 0.6:
                cntscore3 += 1
            elif bleuscore <= 0.8:
                cntscore4 += 1
            else:
                cntscore5 += 1
            dic[cnt] = [recover, reference, bleuscore]
            # print(cnt,get_bleu(recover, reference))
            meteors.append(meteor_score.meteor_score([reference.split()], recover.split()))
            rougel.append(rougeScore(recover, reference)['rougeL_fmeasure'].tolist())
            dist1.append(distinct_n_gram([recover], 1))
            cnt += 1

#        P, R, F1 = score(recovers, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
        # print(cntscore)
        # print(cntscore2)
        # print(cntscore3)
        # print(cntscore4)
        # print(cntscore5)
        print('*'*30)
        print('avg BLEU score', np.mean(bleu))
        print('avg ROUGE-L score', np.mean(rougel))

        print('avg meteor', np.mean(meteors))
        print('avg acc',correct/sample_num*100,'%')
        print('avg dist1 score', np.mean(dist1))
#        print("F1",F1.mean())
#        print("P",P.mean())
#        print("R",R.mean())
        # print('avg len', np.mean(avg_len))

        print('*' * 30)
        # np.save("./dic3.npy",dic)
        print("over")

