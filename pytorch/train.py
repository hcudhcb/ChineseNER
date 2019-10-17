# coding=utf-8
import pickle
import numpy as np
import pdb
with open('../data/Bosondata.pkl', 'rb') as inp:
	word2id = pickle.load(inp)
	id2word = pickle.load(inp)
	tag2id = pickle.load(inp)
	id2tag = pickle.load(inp)
	x_train = pickle.load(inp)
	y_train = pickle.load(inp)
	x_test = pickle.load(inp)
	y_test = pickle.load(inp)
	x_valid = pickle.load(inp)
	y_valid = pickle.load(inp)
print("train len:",len(x_train))
print("test len:",len(x_test))
print("valid len", len(x_valid))

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from torch.utils.data import Dataset,DataLoader
from BiLSTM_CRF import BiLSTM_CRF
from resultCal import calculate,matrix
from tensorboardX import SummaryWriter

class Mydataset(Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label
    def __getitem__(self, index):
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.data)

#############
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCHS = 5
batch_size=32
tag2id[START_TAG]=len(tag2id)
tag2id[STOP_TAG]=len(tag2id)

train_x = []
train_y = []
val_x = []
val_y = []
test_x = []
test_y = []
model = BiLSTM_CRF(len(word2id)+1, tag2id, EMBEDDING_DIM, HIDDEN_DIM,batch_size)

optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
writer = SummaryWriter('./log')
for sentence, tags in zip(x_train,y_train):
    train_x.append(np.array(sentence))
    train_y.append(np.array([tag2id[t] for t in tags]))
for sentence, tags in zip(x_valid,y_valid):
    val_x.append(np.array(sentence))
    val_y.append(np.array([tag2id[t] for t in tags]))
for sentence, tags in zip(x_test,y_test):
    test_x.append(np.array(sentence))
    test_y.append(np.array([tag2id[t] for t in tags]))
train_x = torch.LongTensor(train_x).view(len(train_x),len(train_x[0]),-1)
train_y = torch.LongTensor(train_y)
val_x = torch.LongTensor(val_x).view(len(val_x),len(val_x[0]),-1)
val_y = torch.LongTensor(val_y)
test_x = torch.LongTensor(test_x).view(len(test_x),len(test_x[0]),-1)
test_y = torch.LongTensor(test_y)


dataset = Mydataset(train_x,train_y)
validation = Mydataset(val_x,val_y)
test = Mydataset(test_x,test_y)
trainLoader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
valLoader = DataLoader(dataset=validation,batch_size=batch_size,shuffle=True)
testLoader = DataLoader(dataset=test,batch_size=batch_size,shuffle=True)

for epoch in range(EPOCHS):
    train_loss = 0.0
    entityres = []
    entityall = []
    for i,data in enumerate(trainLoader):
        x,y = data
        x = x.permute(1,0,2)
        #y = y.permute(1,0,2)
        optimizer.zero_grad()
        loss = model.neg_log_likelihood(x,y)
        train_loss += loss
        _,pred = model(x)
        entityres = calculate(x, pred, id2word, id2tag, entityres)
        entityall = calculate(x, y, id2word, id2tag, entityall)
        train_acc,train_recall,train_f1 = matrix(entityres,entityall)
        iter = epoch * len(trainLoader) + i
        writer.add_scalar("train_loss", train_loss, iter)
        writer.add_scalar("train_acc",train_acc,iter)
        writer.add_scalar("train_recall",train_recall,iter)
        writer.add_scalar("train_f1",train_f1,iter)
        loss.backward()
        optimizer.step()
    val_loss = 0.0
    val_res = []
    val_all = []
    for i,data in enumerate(valLoader):
        x,y = data
        x = x.permute(1, 0, 2)
        #y = y.permute(1, 0, 2)
        with torch.no_grad():
            loss = model.neg_log_likelihood(x,y)
            _,pred = model(x)
            val_loss += loss
            val_res = calculate(x,pred,id2word,id2tag,val_res)
            val_all = calculate(x,y,id2word,id2tag,val_all)
            val_acc, val_recall, val_f1 = matrix(entityres, entityall)
            iter = epoch * len(valLoader) + i
            writer.add_scalar("val_loss",val_loss,iter)
            writer.add_scalar("val_acc",val_acc,iter)
            writer.add_scalar("val_recall",val_recall,iter)
            writer.add_scalar("val_f1",val_f1,iter)
    
    path_name = "./model/model"+str(epoch)+".pkl"
    print(path_name)
    torch.save(model, path_name)
    print("model has been saved")

