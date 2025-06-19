from nltk.translate.bleu_score import sentence_bleu
import Levenshtein
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import SmoothingFunction
import jieba
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd
import re

import random
import numpy as np
import torch


DATA_PATH = "results/predictions/prediction_of_saved_models-self_Qwen2.5-14B-Instruct-train_231_lr5e-06_bs2-checkpoint-2088.csv"

# 将句子分词并转换为n-gram格式
def sentence_to_ngrams(sentence, n):

    words = jieba.lcut(sentence)
    return set(nltk.ngrams(words, n))

# 计算BLEU指标
def calculate_bleu(reference, candidate, max_n=4):
    smooth = SmoothingFunction().method4
    scores = []
    for n in range(1, max_n + 1):
        reference_ngrams = sentence_to_ngrams(reference, n)
        candidate_ngrams = sentence_to_ngrams(candidate, n)
        scores.append(
            nltk.translate.bleu_score.sentence_bleu([reference_ngrams], candidate_ngrams, smoothing_function=smooth))
    return scores


if __name__ == '__main__':
    data = {'id':[], 'Ls': [], 'Rouge1-F1': [], 'BLEU1': [], 'Avg': []}
    #data = {'id':[], 'Ls': [], 'BLEU1': [], 'Avg': []}
    data1 = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(data1['Actual Text'][0])
    print(data1['Generated Text'][0])

    # print(data1.info)
    # for i in range(0, len(data1)):a
    # for i in range(0, len(data1)):a
    for i in range(0, len(data1)):
        # if i == 3 :
        #    continue
        print(f"#####################{i} start######################")
        ### 读取
        str1 = data1['Actual Text'][i]
        str2 = data1['Generated Text'][i]
        #str1 = str1.replace("\'", "").replace(" ", "")
        '''
        #prompt workflow
        pattern = r"```"
        split_result = re.split(pattern, str2)
        # print(split_result)
        str2 = split_result[1]
        '''
        # claude
        # str2 = str2.replace("\"","").replace("\n","").replace(" ","").replace("exportconstreactions:ExtractedData[]=","").replace("json","").replace("typescript","")
        # ft
        #str2 = str2.replace("\"", "").replace("\n", "").replace(" ", "").replace(
        #    "typescriptexportconstreactions:ExtractedData[]=", "")
        #print(data1['articleInformation']['title'])
        print(str1)
        print(str2)

        ### 评分
        print("#########Ls#########")
        Ls = Levenshtein.ratio(str1, str2)
        print(Ls)

        print("#########Rs#########")
        rouge = Rouge()
        Rs = rouge.get_scores(str1, str2)
        r1_f = Rs[0]['rouge-1']['f']
        print(r1_f)

        print("#########Bs#########")
        bleu_scores = calculate_bleu(str1, str2)
        b1 = bleu_scores[0]
        print(b1)
        # for n, score in enumerate(bleu_scores, start=1):
        #    print(f'BLEU-{n} Score:', score)
        print("#########Avg#########")
        avg = (Ls + r1_f + b1) / 3
        # avg = (Ls + b1) / 2
        print(f"avg:{avg}")

        data['id'].append(i)
        data['Ls'].append(Ls)
        data['Rouge1-F1'].append(r1_f)
        data['BLEU1'].append(b1)
        data['Avg'].append(avg)
        print("#####################end######################")

    df = pd.DataFrame(data)
    print(df)
    df.to_excel('eval.xlsx', index=False)

















