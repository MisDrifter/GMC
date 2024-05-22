'''Generate the Dataset. Save it in .h5. There are three keys of the dataset. 
    f['input'] is a list of truncated sentences.
'''
import h5py
import os
import copy
import numpy as np
import json
def data_preprocessing():
    corpus = ["datasets/FairText/Data/text_corpus/reddit.txt", "datasets/FairText/Data/text_corpus/meld.txt", 
            "datasets/FairText/Data/text_corpus/news_100.txt",  "datasets/FairText/Data/text_corpus/news_200.txt", 
            "datasets/FairText/Data/text_corpus/sst.txt", "datasets/FairText/Data/text_corpus/wikitext.txt",
            "datasets/FairText/Data/text_corpus/yelp_review_1mb.txt", "datasets/FairText/Data/text_corpus/yelp_review_5mb.txt",
            "datasets/FairText/Data/text_corpus/yelp_review_10mb.txt", "datasets/FairText/Data/artificial_corpus.txt"]
    bias_path = {'female':"datasets/FairText/Data/female-word.txt", 'male':"datasets/FairText/Data/male-word.txt", 'African_American':"datasets/FairText/Data/African_American_names.txt", 'European_American':"datasets/FairText/Data/European_American_names.txt"}
    attribute_path = {'female_profession':"datasets/FairText/Data/female-related occupations.txt", 
                'male_porfession':"datasets/FairText/Data/male-related occupations.txt",
                'pleasant':"datasets/FairText/Data/pleasant-adj.txt", 'unpleasant':"datasets/FairText/Data/unpleasant-adj.txt",
                'female-adj':"datasets/FairText/Data/female-adj.txt", 'male-adj':"datasets/FairText/Data/male-adj.txt", 'family&arts':"datasets/FairText/Data/family&Arts.txt", 'scienc&math&career':"datasets/FairText/Data/science&math&career.txt"}
    bias, attribute = {}, {}
    all_attribute = []
    for key in bias_path:
        f = open(bias_path[key], 'r', encoding='gb2312', errors='ignore')
        bias[key] = list(map(lambda x:x.replace('\n', '').lower(), f.readlines()))
    for key in attribute_path:
        f = open(attribute_path[key], 'r', encoding='gb2312', errors='ignore')
        attribute[key] = list(map(lambda x:x.replace('\n', '').lower(), f.readlines()))
        all_attribute += attribute[key]
    indicator = [bias, attribute]
    
    input = []
    input_label = []
    y = []
    for cor in corpus:
        with open(cor, 'r', encoding='gb2312', errors='ignore') as f:
            text_corpus = f.read()
        text_corpus = text_corpus.split('\n')
        for sent in text_corpus:
            sent = sent.replace('.', ' .')
            sent = sent.replace(',', ' ,')
            sent = sent.replace('?', ' ?')
            sent = sent.replace('"', ' "')
            sent = sent.replace('\'', ' \'')
            tokens = sent.split(' ')
            f1, f2 = -1, -1
            for i,token in enumerate(tokens):
                token = token.lower()
                if (f2!=-1) & (token in all_attribute):
                    f1 = i
                for key in bias:
                    if token in bias[key]:
                        f2 = key
                if (f1!=-1) & (f2!=-1) & (i>3) & (i<50): # if the sentence is too short or too long, skip
                    try:
                        a = np.array(tokens[:f1], dtype='S') # 怕之后报错 先将ASCII无法编码的丢掉
                        input.append(' '.join(tokens[:f1]))
                        input_label.append(f2)
                        y.append(tokens[f1])
                        break
                    except:
                        break
    print('input size: ', len(input))
    return input, input_label, y, indicator

input, input_label, y, indicator = data_preprocessing()
print(input[0], ',' ,input_label[0], ',', y[0])
dataset_path = r'E:\\大学\\大三下\\zlj\\GMC\\Datasets\\FairText\\dataset.h5'
indicator_path = r'E:\\大学\\大三下\\zlj\\GMC\\Datasets\\FairText\\indicator.json'
f = h5py.File(dataset_path, 'w')
f['input'] = np.array(input, dtype='S')
f['input_label'] = np.array(input_label, dtype='S')
f['y'] = np.array(y, dtype='S')
f.close()
with open(indicator_path, 'w') as f:
    json.dump(indicator, f)