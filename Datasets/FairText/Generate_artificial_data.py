import os
import copy
indicator_path = {'female':"datasets/FairText/Data/female-word.txt", 'male':"datasets/FairText/Data/male-word.txt",
            'well_profession':"datasets/FairText/Data/male-related occupations.txt", 
            'less_profession':"datasets/FairText/Data/female-related occupations.txt",
            'pleasant':"datasets/FairText/Data/pleasant-adj.txt", 'unpleasant':"datasets/FairText/Data/unpleasant-adj.txt",
            'female-adj':"datasets/FairText/Data/female-adj.txt", 'male-adj':"datasets/FairText/Data/male-adj.txt"}
indicator = {}
for key in indicator_path:
    f = open(indicator_path[key],'r', encoding='gb2312', errors='ignore').readlines()
    indicator[key] = list(map(lambda s:s[:-1], f))

data_path = 'datasets/FairText/Data/a.txt'
f = open(data_path, 'r', encoding='utf-8').readlines()
f1 = list(map(lambda x:x[3:-3]+' .', f))
f2 = list(map(lambda x:x.split(' '),f1))
output = []
for i in range(len(f2)):
    for j in range(len(f2[i])):
        if f2[i][j]=='[OCCUPATION]':
            f2[i][j] = 'professor'
    for k in range(len(f2[i])):
        if f2[i][k]=='he':
            break  
    output.append(copy.deepcopy(f2[i]))
    output.append(copy.deepcopy(f2[i]))
    output[-1][k] = 'she'
    # for word in indicator['female']:
    #     output.append(copy.deepcopy(f2[i]))
    #     output[-1][k] = word
    # for word in indicator['male']:
    #     output.append(copy.deepcopy(f2[i]))
    #     output[-1][k] = word


f2 = ['he is a [OCCUPATION] .'.split(' '), 'he works as a [OCCUPATION] .'.split(' '), 'he wants to be a [OCCUPATION] .'.split(' ')]
print(f2)
for i in range(len(f2)):
    for j in range(len(f2[i])):
        if f2[i][j]=='[OCCUPATION]':
            f2[i][j] = 'professor'
    for k in range(len(f2[i])):
        if f2[i][k]=='he':
            break  
    for word in indicator['female']:
        output.append(copy.deepcopy(f2[i]))
        output[-1][k] = word
    for word in indicator['male']:
        output.append(copy.deepcopy(f2[i]))
        output[-1][k] = word

print(len(output))
with open(r'datasets\\FairText\\Data\\artificial_corpus.txt','w') as f:
   for sentence in output:
        f.write(' '.join(sentence))
        f.write('\n')
f.close()