# -*-coding:utf-8-*-
'''
这个代码使用pulearning的方法
随机抽取未知样本中的1477种，

时间：8月30日
'''
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import numpy as np
import time
import random

#建立混合神经网络，设置参数
rowLayerNum=3 #行神经网络
rowInputNode=215
rowHideNode=120     #可以改
rowOutNode=70       #可以改，但是要和下面的colOutNode、combineInputNode相同

colLayerNum=4 #列神经网络层数
colInputNode=581
colHide1Node=290    #可以改
colHide2Node=150    #可以改
colOutNode=70       #可以改，但是要和下面的rowOutNode、combineInputNode相同
combineLayerNum=3 #联合神经网络

combineInputNode=70     #可以改，但是要和下面的rowOutNode、combineInputNode相同
combineOutNode=1

batchSize=200 #每批次训练的次数     #和循环迭代次数都没使用上
Iterations=200 #训练迭代次数


# 行神经网络
class rowNet(nn.Module):
    def __init__(self):
        super(rowNet, self).__init__()
        self.InputLayer = nn.Linear(rowInputNode, rowHideNode)
        self.hid1Layer = nn.Linear(rowHideNode, 60)
        self.hideLayer = nn.Linear(60, rowOutNode)

    def forward(self, xRow):
        x = self.InputLayer(xRow)
        x = F.relu(x)
        x = self.hideLayer(x)
        out = F.relu(x)
        return out


class colNet(nn.Module):
    def __init__(self):
        super(colNet, self).__init__()
        self.InputLayer = nn.Linear(colInputNode, colHide1Node)
        self.hide1Layer = nn.Linear(colHide1Node, colHide2Node)
        self.hide2Layer = nn.Linear(colHide2Node, colOutNode)

    def forward(self, xCol):
        x = self.InputLayer(xCol)
        x = F.relu(x)
        x = self.hide1Layer(x)
        x = F.relu(x)
        x = self.hide2Layer(x)
        x = F.relu(x)
        out = x
        return out


class combineNet(nn.Module):
    def __init__(self):
        super(combineNet, self).__init__()
        self.rowModel = rowNet()
        self.colModel = colNet()
        self.inputLayer = nn.Linear(combineInputNode, combineOutNode)


    def forward(self, xRow, xCol):

        out1 = self.rowModel(xRow)
        out2 = self.colModel(xCol)
        combineOut = torch.multiply(out1, out2)
        x = self.inputLayer(combineOut)
        out = torch.sigmoid(x)
        return out





Xpositive = pd.read_feather('positive.feather')
# XrowPositive = Xpositive.iloc[:1033, 1:216]
# XcolPositive = Xpositive.iloc[:1033, 217:]



XrowPositive_train = Xpositive.iloc[:886, 1:216]    #60%的正样本用于训练
XcolPositive_train = Xpositive.iloc[:886, 217:]

XPositive_recall = Xpositive.iloc[886:1033, :]   #10%的样本用于召回计算



XrowPositiveTest = Xpositive.iloc[1033:, 1:216]
XcolPositiveTest = Xpositive.iloc[1033:, 217:]

Xnevigate = pd.read_feather('negative.feather')
Xnevigate =pd.concat([Xnevigate,XPositive_recall])      #将正样本当做负样本添加进去

XrowNevigate = Xnevigate.iloc[:, 1:216]
XcolNevigate = Xnevigate.iloc[:, 217:]

print(XrowPositive_train.shape)
print(XrowNevigate.shape)

print(Xpositive)
print(Xnevigate)





def singleFunction(unkonwnSampleForTrain,unkonwnSampleForTest,inter,flag,lr,nameStr):   #单独训练，不对未知样本数据进行排序，随机抽取
    if len(unkonwnSampleForTrain)>0 and len(unkonwnSampleForTest)>0:
        xRowTrain=pd.concat([XrowPositive_train,XrowNevigate.iloc[unkonwnSampleForTrain]])
        xColTrain=pd.concat([XcolPositive_train,XcolNevigate.iloc[unkonwnSampleForTrain]])
        yTrain=[[1] for _ in range(len(XrowPositive_train))]+[[0] for _ in range(len(unkonwnSampleForTrain))]
        model = combineNet()
        criterion = nn.BCELoss()  # 交叉熵误差
        optimizer = Adam(model.parameters(), lr=lr)
        xRowTrainTensor=torch.tensor(np.array(xRowTrain),dtype=torch.float)
        xColTrainTensor=torch.tensor(np.array(xColTrain),dtype=torch.float)
        yTensorTrain=torch.tensor(yTrain,dtype=torch.float)
        lossList=[]
        for _ in range(inter):
            predict = model(xRowTrainTensor, xColTrainTensor)
            loss = criterion(predict, yTensorTrain)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossList.append(loss.item())
        print('loss:',loss.item())
        xRowTest =XrowNevigate.iloc[unkonwnSampleForTest]   #训练的不放回
        xColTest =XcolNevigate.iloc[unkonwnSampleForTest]


        xRowTestTensor = torch.tensor(np.array(xRowTest),dtype=torch.float)
        xColTestTensor = torch.tensor(np.array(xColTest),dtype=torch.float)
        unkonwnSampleForTestPre = model(xRowTestTensor, xColTestTensor)
        unkonwnSampleForTestPreList = np.array(unkonwnSampleForTestPre.tolist()).reshape(-1)

        unkonwnSampleForTrain=random.sample(range(0,123438+147),886)    #从未知样本集随机抽取886个样本进行训练
        unkonwnSampleForTest = [i for i in range(seqlenth) if i not in unkonwnSampleForTrain]

        torch.save(model, './singleModelSave/model'+str(flag)+'.pt')
    lossDataFrame=pd.DataFrame(lossList)
    lossDataFrame.to_csv('./singleLossSave/'+nameStr+'.csv')
    return unkonwnSampleForTrain,unkonwnSampleForTest



def twoStepFunction(unkonwnSampleForTrain,unkonwnSampleForTest,inter,flag,lr,nameStr):  #两步法模型训练
    if len(unkonwnSampleForTrain)>0 and len(unkonwnSampleForTest)>0:   #非空列表判断
        xRowTrain=pd.concat([XrowPositive_train,XrowNevigate.iloc[unkonwnSampleForTrain]])
        xColTrain=pd.concat([XcolPositive_train,XcolNevigate.iloc[unkonwnSampleForTrain]])
        yTrain=[[1] for _ in range(len(XrowPositive_train))]+[[0] for _ in range(len(unkonwnSampleForTrain))]
        model = combineNet()
        criterion = nn.BCELoss()  # 交叉熵误差
        optimizer = Adam(model.parameters(), lr=lr)
        xRowTrainTensor=torch.tensor(np.array(xRowTrain),dtype=torch.float)
        xColTrainTensor=torch.tensor(np.array(xColTrain),dtype=torch.float)
        yTensorTrain=torch.tensor(yTrain,dtype=torch.float)
        lossList=[]
        for _ in range(inter):
            predict = model(xRowTrainTensor, xColTrainTensor)
            loss = criterion(predict, yTensorTrain)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossList.append(loss.item())
        print('loss:',loss.item())
        xRowTest =XrowNevigate.iloc[unkonwnSampleForTest]   #训练的不放回
        xColTest =XcolNevigate.iloc[unkonwnSampleForTest]

        # xRowTest = XrowNevigate.loc[:,:]      #训练的放回
        # xColTest = XcolNevigate.loc[:,:]
        xRowTestTensor = torch.tensor(np.array(xRowTest),dtype=torch.float)
        xColTestTensor = torch.tensor(np.array(xColTest),dtype=torch.float)
        unkonwnSampleForTestPre=model(xRowTestTensor,xColTestTensor)
        unkonwnSampleForTestPreList=np.array(unkonwnSampleForTestPre.tolist()).reshape(-1)
        preForChoice=sorted(enumerate(unkonwnSampleForTestPreList), key=lambda x:x[1])    #将预测值和对应的序号取出，按照取值排序
        preForChoiceDict=dict(preForChoice)
        preForChoiceSeq=list(preForChoiceDict.keys())

        unkonwnSampleForTrain=preForChoiceSeq[:886]    #取出未知样本集中被预测最接近1的886个样本用于之后训练
        t1=time.time()
        unkonwnSampleForTest = [i for i in range(seqlenth) if i not in unkonwnSampleForTrain]
        t2=time.time()
        print('time:',t2-t1)

        torch.save(model, './loopModelSave/model'+str(flag)+'.pt')
    lossDataFrame=pd.DataFrame(lossList)
    lossDataFrame.to_csv('./loopLossSave/'+nameStr+'.csv')
    return unkonwnSampleForTrain,unkonwnSampleForTest

# size=1033
positiveTrainSize=886   #从正样本中抽出作为训练集，正样本的个数为1477,占60%
positiveTestSize=444    #测试样本数
positiveRecallSize=147  #放入未知样本训练，看能识别出的数

seqlenth=123438    #未知样本的个数
# seqlenth1=123438+147
unkonwnSampleForTrainFinal=[i for i in range(123438,123438+147)]
# unkonwnSampleForTrainFinal=np.array(unkonwnSampleForTrainFinal)     #初次训练将要召回的值全部导入训练

seqNum=[i for i in range(seqlenth)]
unkonwnSampleForTrain=np.random.choice(a=seqNum,size=positiveTrainSize-147,replace=False)    #从未知样本中挑选出1033个样本
for s in unkonwnSampleForTrainFinal:
    unkonwnSampleForTrain=np.append(unkonwnSampleForTrain,s)
unkonwnSampleForTest=[i for i in range(seqlenth) if i not in unkonwnSampleForTrain] #将未知样本中剩下的当做测试样本



####两步法模型训练##################
inter=100   #训练的次数
# learnRateInit=0.5
learnRateInit=0.005
loopNum=60
saveNum_unkonwnSampleForTrain=[]
# for i in range(loopNum):
# #     learnRate=learnRateInit*(0.9**i)
#     learnRate = learnRateInit
#     print('i=',i,'learnRate',learnRate)
#
#     flag=i
#     unkonwnSampleForTrain, unkonwnSampleForTest = twoStepFunction(unkonwnSampleForTrain, unkonwnSampleForTest,inter=inter, flag=flag,lr=learnRate,nameStr='N'+str(i)+'lossList')
#     # unkonwnSampleForTrain, unkonwnSampleForTest =singleFunction(unkonwnSampleForTrain, unkonwnSampleForTest,inter=inter, flag=flag,lr=learnRate,nameStr='N'+str(i)+'lossList')  #一步法
#     saveNum_unkonwnSampleForTrain.append(unkonwnSampleForTrain)
# saveNum_unkonwnSampleForTrainDataFrame=pd.DataFrame(saveNum_unkonwnSampleForTrain)
# saveNum_unkonwnSampleForTrainDataFrame.to_csv('./loopModelSave/saveNum_unkonwnSampleForTrain.csv')
# print('训练结束')


def sortMaxNumFunction(inter, lr):  # 在进行两步法训练后，挑选的样本中重复出现次数最多的前886个样本用于训练，进一步计算
    print('开始运行sortMaxNum程序')
    unkonwnSampleForTrainDataFrame = pd.read_csv('./loopModelSave/nevChoiceCount.csv')
    unkonwnSampleForTrain = unkonwnSampleForTrainDataFrame['label']
    unkonwnSampleForTrain = unkonwnSampleForTrain[:886].values

    xRowTrain = pd.concat([XrowPositive_train, XrowNevigate.iloc[unkonwnSampleForTrain]])
    xColTrain = pd.concat([XcolPositive_train, XcolNevigate.iloc[unkonwnSampleForTrain]])
    yTrain = [[1] for _ in range(len(XrowPositive_train))] + [[0] for _ in range(len(unkonwnSampleForTrain))]
    model = combineNet()
    criterion = nn.BCELoss()  # 交叉熵误差
    optimizer = Adam(model.parameters(), lr=lr)
    xRowTrainTensor = torch.tensor(np.array(xRowTrain), dtype=torch.float)
    xColTrainTensor = torch.tensor(np.array(xColTrain), dtype=torch.float)
    yTensorTrain = torch.tensor(yTrain, dtype=torch.float)
    lossList = []
    for _ in range(inter):
        predict = model(xRowTrainTensor, xColTrainTensor)
        loss = criterion(predict, yTensorTrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lossList.append(loss.item())
    print('loss:', loss.item())
    torch.save(model, './loopModelSave/modelFinal.pt')
    lossDataFrame = pd.DataFrame(lossList)
    lossDataFrame.to_csv('./loopLossSave/' + 'loosFinal.csv')
    print('程序运行结束')
# sortMaxNumFunction(inter=inter,lr=learnRateInit)

'''####两步法模型计算结果的评估参数计算################
accuracyList=[]
for i in range(60):
    model = torch.load('./loopModelSave/model'+str(i)+'.pt')
    # model = torch.load('./singleModelSave/model' + str(i) + '.pt')
    xRowTest =XrowNevigate.iloc[:123438,:]
    xColTest =XcolNevigate.iloc[:123438,:]
    xRowTestTensor = torch.tensor(np.array(xRowTest),dtype=torch.float)
    xColTestTensor = torch.tensor(np.array(xColTest),dtype=torch.float)
    predictTest = model(xRowTestTensor, xColTestTensor)
    predictTestList=predictTest.tolist()
    predictTestDataFrame=pd.DataFrame(predictTestList)
    predictTestDataFrame.to_csv('./loopPreTest/twoStepPreTest'+str(i)+'.csv',header=None,index=None)
    # predictTestDataFrame.to_csv('./singlePreTest/twoStepPreTest' + str(i) + '.csv', header=None, index=None)

    xrowPositiveTestTensor = torch.tensor(np.array(XrowPositiveTest),dtype=torch.float)
    xcolPositiveTestTensor = torch.tensor(np.array(XcolPositiveTest),dtype=torch.float)
    predictPosiTest = model(xrowPositiveTestTensor, xcolPositiveTestTensor)
    predictPosiTestList=predictPosiTest.tolist()
    predictPosiTestDataFrame=pd.DataFrame(predictPosiTestList)

    len_all=predictPosiTestDataFrame.shape[0]
    len_1=predictPosiTestDataFrame[predictPosiTestDataFrame[0]>0.999].shape[0]

    print("model"+str(i)+"正样本测试集准确率：{}".format(len_1/len_all))
    accuracyList.append(len_1/len_all)
    predictPosiTestDataFrame.to_csv('./loopPosiPreTest/twoStepPosiPreTest'+str(i)+'.csv',header=None,index=None)
    # predictPosiTestDataFrame.to_csv('./singlePosiPreTest/twoStepPosiPreTest' + str(i) + '.csv', header=None, index=None)

recallRateList=[]
for i in range(60):
    model = torch.load('./loopModelSave/model'+str(i)+'.pt')
    # model = torch.load('./singleModelSave/model' + str(i) + '.pt')

    XrowPositive_recall = Xpositive.iloc[886:1033, 1:216]   #10%的样本用于召回计算
    XcolPositive_recall=Xpositive.iloc[886:1033, 217:]

    xRowRecallTensor = torch.tensor(np.array(XrowPositive_recall),dtype=torch.float)
    xColRecallTensor = torch.tensor(np.array(XcolPositive_recall),dtype=torch.float)
    predictTest = model(xRowRecallTensor, xColRecallTensor)
    predictTestList=predictTest.tolist()
    predictTestDataFrame=pd.DataFrame(predictTestList)
    predictTestDataFrame.to_csv('./loopPreRecall/twoStepPreRecall'+str(i)+'.csv',header=None,index=None)
    # predictTestDataFrame.to_csv('./singlePreRecall/twoStepPreRecall' + str(i) + '.csv', header=None, index=None)

    df=pd.read_csv('./loopPreRecall/twoStepPreRecall'+str(i)+'.csv',header=None,index_col=False)
    # df = pd.read_csv('./singlePreRecall/twoStepPreRecall' + str(i) + '.csv', header=None, index_col=False)
    length=df.shape[0]
    df1=df[df[0]>0.999]
    print('model',i,'召回率',":",df1.shape[0]/length)
    recallRateList.append(df1.shape[0]/length)


accuracy_recallDataFrame = pd.DataFrame({'accuracy':accuracyList, 'recall':recallRateList})
accuracy_recallDataFrame.to_csv('./loopAcc/accuracy_recall.csv')
# accuracy_recallDataFrame.to_csv('./singleAcc/accuracy_recall.csv')

####两步法模型计算结果的评估参数计算 end################
'''


