import sys, os
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.nn import functional
import copy
import matplotlib.pyplot as plt
# import os
import json
#from core.GMC import GMC
import h5py
import cv2
import time
from tqdm import tqdm
from utils.utils import eval, plot_result, eval2
from transformers import AutoTokenizer, BertForPreTraining
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

tokenizer = AutoTokenizer.from_pretrained(r'D:\\Download\\bert-base-uncased')
model = BertForPreTraining.from_pretrained(r'D:\\Download\\bert-base-uncased')

average_p = 0
def load_data(artificial=False):
    if artificial:
        data_path_1 = os.path.join(os.getcwd(), 'datasets/FairText/Result/Result_2.h')
        accuracy_test_data_path = os.path.join(os.getcwd(), 'datasets/FairText/Result/Accuracy_test_2.h')    
    else:
        data_path_1 = os.path.join(os.getcwd(), 'datasets/FairText/Result/Result_1.h')
        accuracy_test_data_path = os.path.join(os.getcwd(), 'datasets/FairText/Result/Accuracy_test_1.h')
    f1 = h5py.File(data_path_1, "r")
    position, p, x, y = f1['mask'], f1['p'], np.array(list(map(lambda x:x.decode(), f1['x']))), np.array(f1['y'])
    print('raw data shape: ', p.shape, len(x), len(y))
    p = np.array(p)[[i for i in range(p.shape[0])], np.array(list(position)),:]
    f2 = h5py.File(accuracy_test_data_path, "r")
    position_accu, p_accu, x_accu, y_accu = f2['mask'], f2['p'], np.array(list(map(lambda x:x.decode(), f2['x']))), np.array(f2['y'])
    print('raw data shape: ', p_accu.shape, len(x_accu), len(y_accu))
    p_accu = np.array(p_accu)[[i for i in range(p_accu.shape[0])], np.array(list(position_accu)),:]
    indicator_path = {'female':"datasets/FairText/Data/female-word.txt", 'male':"datasets/FairText/Data/male-word.txt",
                'well_profession':"datasets/FairText/Data/male-related occupations.txt", 
                'less_profession':"datasets/FairText/Data/female-related occupations.txt",
                'pleasant':"datasets/FairText/Data/pleasant-adj.txt", 'unpleasant':"datasets/FairText/Data/unpleasant-adj.txt",
                'female-adj':"datasets/FairText/Data/female-adj.txt", 'male-adj':"datasets/FairText/Data/male-adj.txt"}
    indicator = {}
    for key in indicator_path:
        f = open(os.path.join(indicator_path[key]),'r', encoding='gb2312', errors='ignore').readlines()
        indicator[key] = list(map(lambda s:s[:-1], f))
    return indicator, p, x, y, p_accu, x_accu, y_accu

def intersect_or_not(a,b):
    # input two lists; return whether they intersect
    return len(list(set(a)&set(b)))!=0


def generate_sensitive_vector(mark, total_length=30522):
    # input a list of the position of 1; output an indicator function of it; to one hot
    vec = np.zeros(total_length)
    for position in mark:
        vec[tokenizer(position).input_ids[1]] = 1
    return vec


def generate_G(indicator):
    female = lambda x:intersect_or_not(x.split(' '), indicator['female'])
    male = lambda x:intersect_or_not(x.split(' '), indicator['male'])
    vec_list = [generate_sensitive_vector(subgroup) for subgroup in [indicator['female-adj'],indicator['male-adj'],
                indicator['less_profession'], indicator['well_profession'], indicator['pleasant'], indicator['unpleasant']]]
    G = []
    G.append(lambda x:vec_list[0]*female(x))
    G.append(lambda x:vec_list[0]*male(x))
    G.append(lambda x:-vec_list[0]*female(x))
    G.append(lambda x:-vec_list[0]*male(x))  
    G.append(lambda x:vec_list[1]*female(x))
    G.append(lambda x:vec_list[1]*male(x))
    G.append(lambda x:-vec_list[1]*female(x))
    G.append(lambda x:-vec_list[1]*male(x))  
    G.append(lambda x:vec_list[2]*female(x))
    G.append(lambda x:vec_list[2]*male(x))
    G.append(lambda x:-vec_list[2]*female(x))
    G.append(lambda x:-vec_list[2]*male(x))    
    G.append(lambda x:vec_list[3]*female(x))
    G.append(lambda x:vec_list[3]*male(x))
    G.append(lambda x:-vec_list[3]*female(x))
    G.append(lambda x:-vec_list[3]*male(x))
    G.append(lambda x:vec_list[4]*female(x))
    G.append(lambda x:vec_list[4]*male(x))
    G.append(lambda x:-vec_list[4]*female(x))
    G.append(lambda x:-vec_list[4]*male(x))  
    G.append(lambda x:vec_list[5]*female(x))
    G.append(lambda x:vec_list[5]*male(x))
    G.append(lambda x:-vec_list[5]*female(x))
    G.append(lambda x:-vec_list[5]*male(x))   
    return G


def projection(f):
    N, D = f.shape
    X = np.sort(f, axis=1)[..., ::-1]
    Xtmp = (np.cumsum(X, axis=1) - 1) * (1 / np.arange(1, D + 1))
    lamda = (1 - np.sum((X>Xtmp) * X, axis=1)) / np.sum(X>Xtmp, axis=1)
    X = np.maximum(f + lamda[:,None], 0)
    
    return X



def GMC(alpha, eta, x_cal, y_cal, h_cal, x_test, h_test, s, group_G, f=(lambda x:0), f_test=(lambda x:0), T=500, proj=None):
    '''
    eta:learning_rate
    x_cal:numpy
    y_cal:numpy
    h_cal:numpy
    x_test:numpy
    h_test:numpy
    s:mapping_function
    f:initial_function 
    group_G:list of group functions
    T:max_iteration
    return function f
    ''' 
    fx = f(x_cal)
    fx_test = f_test(x_test)
    n = x_cal.shape[0]
    for i in range(T):
        update = False
        for g in group_G:
            gx = np.array([g(x1)for x1 in x_cal])
            if np.sum(np.diagonal(gx@(s(fx, x_cal, y_cal, h_cal).T)))>alpha*n:
                update = True
                print(group_G.index(g))
                print(np.sum(np.diagonal(gx@(s(fx, x_cal, y_cal, h_cal).T))))
                break
        if update==False:
            print(i)
            print('end')
            return i, fx, fx_test
        else:
            gx_test = np.array([g(x1)for x1 in x_test])
            fx = fx - eta*gx
            fx_test = fx_test - eta*gx_test
            if not (proj is None):
                fx = proj(fx)
                fx_test = proj(fx_test)
    return i+1, fx, fx_test

def eval(G, s, fx, x_cal, y_cal, h_cal, accu=False):
    S = []
    n = x_cal.shape[0]
    if accu:
        return np.sum(s(fx, x_cal, y_cal, h_cal))/n
    for g in G:
        gx = np.array([g(x1)for x1 in x_cal])
        S.append(np.sum(np.diagonal(gx@(s(fx, x_cal, y_cal, h_cal).T)))/n)
    return np.array(S)

def entropy(f, x, y, p):
    return F.cross_entropy(torch.Tensor(f),torch.Tensor(y).long()).numpy()*p.shape[0]


def experiment(G, p, x, y, p_accu, x_accu, y_accu, random_seed):
    def s(f, x, y, p):
        average_f = np.mean(f,axis=0)
        return f-average_f
    alpha = 0.002
    eta = 0.001
    p_cal, p_test, x_cal, x_test, y_cal, y_test = train_test_split(p, x, y, test_size=0.5, random_state=random_seed)
    fx, fx_test = copy.deepcopy(p_cal), copy.deepcopy(p_test)
    j, fx, fx_test = GMC(3/4*alpha, eta, x_cal, y_cal, p_cal, x_test, p_test, s, G, f=(lambda x:fx), f_test=(lambda x:fx_test), T=30, proj = projection)

    select_G = [G[0], G[4], G[8], G[12], G[16], G[20]]
    gap_cal = eval(select_G, s, fx, x_cal, y_cal, p_cal)
    gap_cal_b = eval(select_G, s, p_cal, x_cal, y_cal, p_cal)
    gap_test = eval(select_G, s, fx_test, x_test, y_test, p_test)
    gap_test_b = eval(select_G, s, p_test, x_test, y_test, p_test)
    gap_cal, gap_cal_b, gap_test, gap_test_b = abs(gap_cal), abs(gap_cal_b), abs(gap_test), abs(gap_test_b) 

    select_G = [G[1], G[5], G[9], G[13], G[17], G[21]]
    gap_cal_2 = eval(select_G, s, fx, x_cal, y_cal, p_cal)
    gap_cal_b_2 = eval(select_G, s, p_cal, x_cal, y_cal, p_cal)
    gap_test_2 = eval(select_G, s, fx_test, x_test, y_test, p_test)
    gap_test_b_2 = eval(select_G, s, p_test, x_test, y_test, p_test)
    gap_cal_2, gap_cal_b_2, gap_test_2, gap_test_b_2 = abs(gap_cal), abs(gap_cal_b), abs(gap_test_2), abs(gap_test_b_2) 

    eta = 0.001
    fx, fx_accu = copy.deepcopy(p_cal), copy.deepcopy(p_accu)
    j, fx, fx_accu = GMC(3/4*alpha, eta, x_cal, y_cal, p_cal, x_accu, p_accu, s, G, f=(lambda x:fx), f_test=(lambda x:fx_accu), T=30, proj = projection)

    G_accu = [lambda x,f:1]
    accu = eval(G_accu, entropy, fx_accu, x_accu, y_accu, p_accu, accu=True)
    accu_b = eval(G_accu, entropy, p_accu, x_accu, y_accu, p_accu, accu=True)
    print('cross entropy of our algorithm: ', accu)
    print('cross entropy of the baseline: ', accu_b)
    # eval(G_accu, entropy, 0, x_accu, y_accu, np.ones((1100,vocab))/vocab, accu=True)
    return accu, accu_b, gap_test, gap_test_b, gap_test_2, gap_test_b_2
    # return accu, accu_b, gap_cal, gap_cal_b, gap_cal_2, gap_cal_b_2


if __name__=='__main__':
    indicator, p, x, y, p_accu, x_accu, y_accu = load_data(artificial=False)
    G = generate_G(indicator)
    st = time.time()
    category = ['female-adj', 'male-adj', 'female-occupation', 'male-occupation', 'pleasant', 'unpleasant']
    test_accu_all, test_accu_b_all, gap_test_all, gap_test_b_all, gap_test_all_2, gap_test_b_all_2 = [], [], [], [], [], []
    for i in tqdm(range(10)):
        test_accu, test_accu_b, gap_test, gap_test_b, gap_test_2, gap_test_b_2 = experiment(G, p, x, y, p_accu, x_accu, y_accu, i)
        test_accu_all.append(test_accu)
        test_accu_b_all.append(test_accu_b)
        gap_test_all.append(gap_test)
        gap_test_b_all.append(gap_test_b)
        gap_test_all_2.append(gap_test_2)
        gap_test_b_all_2.append(gap_test_b_2)
    test_accu_all, test_accu_b_all, gap_test_all, gap_test_b_all, gap_test_all_2, gap_test_b_all_2 = \
        np.array(test_accu_all), np.array(test_accu_b_all), np.array(gap_test_all), np.array(gap_test_b_all), np.array(gap_test_all_2), np.array(gap_test_b_all_2)
    print('test_accuracy:', np.mean(test_accu_all), np.std(test_accu_all))
    print('test_accuracy_baseline:', np.mean(test_accu_b_all), np.std(test_accu_b_all))
    for i in range(len(category)):
        print(category[i], np.mean(gap_test_all[:,i]), np.std(gap_test_all[:,i]))
        print(category[i]+'_baseline', np.mean(gap_test_b_all[:,i]), np.std(gap_test_b_all[:,i]))
    for i in range(len(category)):
        print(category[i], np.mean(gap_test_all_2[:,i]), np.std(gap_test_all_2[:,i]))
        print(category[i]+'_baseline', np.mean(gap_test_b_all_2[:,i]), np.std(gap_test_b_all_2[:,i]))
    print('time spent: ', time.time()-st)

    # indicator, p, x, y, p_accu, x_accu, y_accu = load_data(artificial=False)
    # G = generate_G(indicator)
    # st = time.time()
    # category = ['female-adj', 'male-adj', 'female-occupation', 'male-occupation', 'pleasant', 'unpleasant']
    # test_accu_all, test_accu_b_all, gap_test_all, gap_test_b_all, gap_test_all_2, gap_test_b_all_2 = [], [], [], [], [], []
    # for i in tqdm(range(50)):
    #     test_accu, test_accu_b, gap_test, gap_test_b, gap_test_2, gap_test_b_2 = experiment(G, p, x, y, p_accu, x_accu, y_accu, i)
    #     test_accu_all.append(test_accu)
    #     test_accu_b_all.append(test_accu_b)
    #     gap_test_all.append(gap_test)
    #     gap_test_b_all.append(gap_test_b)
    #     gap_test_all_2.append(gap_test_2)
    #     gap_test_b_all_2.append(gap_test_b_2)
    # test_accu_all, test_accu_b_all, gap_test_all, gap_test_b_all, gap_test_all_2, gap_test_b_all_2 = \
    #     np.array(test_accu_all), np.array(test_accu_b_all), np.array(gap_test_all), np.array(gap_test_b_all), np.array(gap_test_all_2), np.array(gap_test_b_all_2)
    # print('test_accuracy:', np.mean(test_accu_all), np.std(test_accu_all))
    # print('test_accuracy_baseline:', np.mean(test_accu_b_all), np.std(test_accu_b_all))
    # for i in range(len(category)):
    #     print(category[i], np.mean(gap_test_all[:,i]), np.std(gap_test_all[:,i]))
    #     print(category[i]+'_baseline', np.mean(gap_test_b_all[:,i]), np.std(gap_test_b_all[:,i]))
    # for i in range(len(category)):
    #     print(category[i], np.mean(gap_test_all_2[:,i]), np.std(gap_test_all_2[:,i]))
    #     print(category[i]+'_baseline', np.mean(gap_test_b_all_2[:,i]), np.std(gap_test_b_all_2[:,i]))
    # print('time spent: ', time.time()-st)