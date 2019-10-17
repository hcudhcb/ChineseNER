# coding=utf-8
import codecs
import numpy as np
import torch
def calculate(x,y,id2word,id2tag,res=[]):
    all = []
    x = x.permute(1,0,2)
    x = x.view(len(x),-1)
    print(len(x),len(x[0]),len(y),len(y[0]))
    if not x is np.ndarray:
        x.numpy()
    if isinstance(y,list):
        y = np.array(y)
    elif torch.is_tensor(y):
        y.numpy()
    for i in range(len(x)):
        entity=[]
        for j in range(len(x[i])):
            if x[i][j].numpy()==0 or y[i][j]==0:
                continue
            if id2tag[y[i][j]][0]=='B':
                entity=[id2word[x[i][j].numpy()]+'/'+id2tag[y[i][j]]]
            elif id2tag[y[i][j]][0]=='M' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j].numpy()]+'/'+id2tag[y[i][j]])
            elif id2tag[y[i][j]][0]=='E' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j].numpy()]+'/'+id2tag[y[i][j]])
                entity.append(str(j))
                res.append(entity)
                entity=[]
            else:
                entity=[]
        all.append(res)
    return all
    
    
def calculate3(x,y,id2word,id2tag,res=[]):
    '''
    使用这个函数可以把抽取出的实体写到res.txt文件中，供我们查看。
    注意，这个函数每次使用是在文档的最后添加新信息，所以使用时尽量删除res文件后使用。
    '''
    with codecs.open('./res.txt','a','utf-8') as outp:
        entity=[]
        for j in range(len(x)): #for every word
            if x[j]==0 or y[j]==0:
                continue
            if id2tag[y[j]][0]=='B':
                entity=[id2word[x[j]]+'/'+id2tag[y[j]]]
            elif id2tag[y[j]][0]=='M' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[j]][1:]:
                entity.append(id2word[x[j]]+'/'+id2tag[y[j]])
            elif id2tag[y[j]][0]=='E' and len(entity)!=0 and entity[-1].split('/')[1][1:]==id2tag[y[j]][1:]:
                entity.append(id2word[x[j]]+'/'+id2tag[y[j]])
                entity.append(str(j))
                res.append(entity)
                st = ""
                for s in entity:
                    st += s+' '
                #print st
                outp.write(st+'\n')
                entity=[]
            else:
                entity=[]
    return res

def matrix(res,all):
    hits = [i for i in res if i in all]
    if len(hits)!=0:
        acc = float(len(hits))/len(res)
        recall = float(len(hits))/len(all)
        f1 = 2*acc*recall / (acc+recall)
    else:
        acc,recall,f1 = 0.0
    return acc,recall,f1

