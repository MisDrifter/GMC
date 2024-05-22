from transformers import AutoTokenizer, BertForPreTraining
from torch.nn.functional import softmax
import torch
import numpy as np
import torch
import time
import h5py
import copy

import sys
import os


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForPreTraining.from_pretrained("bert-base-uncased")

def data_preprocessing():
    corpus = ["datasets/FairText/Data/text_corpus/reddit.txt", "datasets/FairText/Data/text_corpus/meld.txt", 
            "datasets/FairText/Data/text_corpus/news_100.txt",  "datasets/FairText/Data/text_corpus/news_200.txt", 
            "datasets/FairText/Data/text_corpus/sst.txt", "datasets/FairText/Data/text_corpus/wikitext.txt",
            "datasets/FairText/Data/text_corpus/yelp_review_1mb.txt", "datasets/FairText/Data/text_corpus/yelp_review_5mb.txt",
            "datasets/FairText/Data/text_corpus/yelp_review_10mb.txt", "datasets/FairText/Data/artificial_corpus.txt"]
    indicator_path = {'female':"datasets/FairText/Data/female-word.txt", 'male':"datasets/FairText/Data/male-word.txt",
                'well_profession':"datasets/FairText/Data/female-related occupations.txt", 
                'less_porfession':"datasets/FairText/Data/male-related occupations.txt",
                'pleasant':"datasets/FairText/Data/pleasant-adj.txt", 'unpleasant':"datasets/FairText/Data/unpleasant-adj.txt",
                'female-adj':"datasets/FairText/Data/female-adj.txt", 'male-adj':"datasets/FairText/Data/male-adj.txt"}
    indicators = []
    for key in indicator_path:
        f = open(indicator_path[key], 'r', encoding='gb2312', errors='ignore')
        indicators.append(f.readlines())
    gender_indicators = [word for word in indicators[0]] + [word for word in indicators[1]]
    gender_indicators = [s[:-1].lower() for s in gender_indicators]
    sensitive_indicators = [word for i in range(2, len(indicators)) for word in indicators[i]]
    sensitive_indicators = [s[:-1].lower() for s in sensitive_indicators]
    print('gender_words:', len(gender_indicators), 'sensitive_words:', len(sensitive_indicators))

    input = []
    input_mask = []
    length = []
    for cor in corpus:
        with open(cor, 'r', encoding='gb2312', errors='ignore') as f:
            text_corpus = f.read()
        text_corpus = text_corpus.split('\n')
        if cor=='datasets/FairText/Data/artificial_corpus.txt':
            artificial_length = len(text_corpus)
        for sent in text_corpus:
            sent = sent.replace('.', ' .')
            sent = sent.replace(',', ' ,')
            sent = sent.replace('?', ' ?')
            sent = sent.replace('"', ' "')
            sent = sent.replace('\'', ' \'')
            tokens = sent.split(' ')
            if (len(tokens) < 5) or (len(tokens)>20):  # if the sentence is too short or too long, skip
                continue
            f1, f2 = -1, -1
            for i,token in enumerate(tokens):
                token = token.lower()
                if token in sensitive_indicators:
                    f1 = i
                if token in gender_indicators:
                    f2 = i
                if (f1!=-1) & (f2!=-1):
                    input.append(sent)
                    input_mask.append(f1+1)
                    length.append(len(tokens))
                    break
    print('input size: ', len(input))
    print(input[:3])
    return input, input_mask, length, artificial_length


def get_output(input, input_mask, savepath, name):
    os.makedirs(savepath, exist_ok=True)
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
    print('shape of input: ', inputs.input_ids.shape)
    label = []
    for i in range(len(input_mask)):
        inputs['attention_mask'][i][input_mask[i]] = 0
        label.append(copy.deepcopy(inputs.input_ids[i][input_mask[i]]))
        inputs.input_ids[i][input_mask[i]] = 103
    label = np.array(label)
    print('shape of label: ', label.shape)
    output = model(**inputs)
    print('shape of distribution: ', output.prediction_logits.shape)
    f = h5py.File(os.path.join(savepath, name), 'w')
    f['x'] = list(map(lambda x:x.encode(), input))
    f['p'] = softmax(output.prediction_logits.detach(), dim=-1).numpy()
    f['y'] = label
    f['mask'] = input_mask
    f.close()


def get_output_for_accuracy_test(input, length, savepath, name):
    print(input)
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
    print('shape of input: ', inputs.input_ids.shape)
    label = []
    input_mask = []
    for i in range(len(input)):
        input_mask.append(np.random.randint(1, length[i]+1))
        inputs['attention_mask'][i][input_mask[i]] = 0
        label.append(copy.deepcopy(inputs.input_ids[i][input_mask[i]]))
        inputs.input_ids[i][input_mask[i]] = 103
    label = np.array(label)
    print('shape of label: ', label.shape)
    output = model(**inputs)
    print('shape of distribution: ', output.prediction_logits.shape)
    f = h5py.File(os.path.join(savepath, name), 'w')
    f['x'] = list(map(lambda x:x.encode(), input))
    f['p'] = softmax(output.prediction_logits.detach(), dim=-1).numpy()
    f['y'] = label
    f['mask'] = input_mask
    f.close()


if __name__=='__main__':
    savepath = 'datasets/FairText/Result'
    input, input_mask, length, artificial_length = data_preprocessing()
    print(artificial_length)
    get_output(input[-1500-artificial_length:-artificial_length], input_mask[-1500-artificial_length:-artificial_length], savepath, 'Result_1_1.h')
    get_output_for_accuracy_test(input[-1500-artificial_length:-artificial_length], length[-1500-artificial_length:-artificial_length], savepath, 'Accuracy_test_1_1.h')
    get_output(input[-artificial_length:], input_mask[-artificial_length:], savepath, 'Result_2.h')
    get_output_for_accuracy_test(input[-artificial_length:], length[-artificial_length:], savepath, 'Accuracy_test_2.h')