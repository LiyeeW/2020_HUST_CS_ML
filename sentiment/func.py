# -*- coding: UTF-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier              #模型一：47%正确率
from sklearn.ensemble import RandomForestClassifier             #模型二：52%正确率
from sklearn import tree                                        #模型三：50%正确率
from sklearn import naive_bayes                                 #模型四：55%正确率（伯努利）, 48%(高斯), 53%（多项式）
from sklearn.externals import joblib                            #下载训练好后的模型
from sklearn.metrics import confusion_matrix                    #混淆矩阵模型
import csv                                                      #按行读写csv文件
import jieba
import jieba.analyse
import pandas as pd
import numpy as np
import math                                                     #计算关键词赋分
import re
import matplotlib.pyplot as plt
import os                                                       #判断文件是否存在
from langconv import *

def deleteOthers(text):
    index = text.find('//@')
    if (index == -1):                                           #若不含上级用户内容，则直接跳过
        text = text
    elif (index == 0):                                            #若含上级用户，且本级用户没有发言内容，则截取上级用户内容
        index = text.find(':')
        indexNext = text[3:].find('//@')                        #寻找有无上上级用户
        if(indexNext!=-1):
            indexNext = indexNext+3
            if(indexNext == index+1):                           #如果上级用户也无内容，则递归处理
                text = deleteOthers(text[indexNext:])
            else:
                text = text[index+1:indexNext]                          #若上级用户有内容，则只保留上级用户
        else:
            text = text[index + 1:]
    elif(index>0):
        text = text[:index]
    index = text.find('回复@')
    if(index==-1):
        return text
    index = text.find(':')
    return text[index+1:]


def textUpper(text):
    text = text.upper()
    text = Converter('zh-hans').convert(text)
    return text

def deleteTopic(text):
    index = text.find("#")  # 目标删除开头
    indexEnd = 0
    while(index != indexEnd+2):
        indexEnd = text[index+1:].find("#")+index+1
        if(indexEnd==index):
            break
        text = text[:index+1] + ' ' + text[indexEnd:]
        indexEnd = index
        index = text[index+3:].find("#") + index+3
    return text



def deleteWeiboFormat(text):
    index = text.find("我参与了@")                               #删除“投票动态”中的格式部分
    if (index != -1):                                           #只保留原投票标题和用户所选选项
        textBackend=text[index+1:]
        text = text[:index] + '' + text[textBackend.find("发起的")+index+1 + 3:textBackend.find("】")+index+1 + 1] + '' + text[textBackend.find(
            "我投给了“")+index+1 + 5:textBackend.find("”这个选项")+index+1] + '' + text[textBackend.find("快来表态吧~")+index+1 + 6:]
    index = text.find('我围观了')
    if(index!=-1):
        text = '我围观了'
    text = re.sub('展开全文c', '', text)
    text = re.sub('查看图片', '', text)
    text = re.sub('L.+?视频', '', text)
    text = text.replace('?','')
    return text

def textFormalize(train_text, train_lable):                                        #将原始训练集中的微博规范化，并划分为lable不同的三组
    train_data = []
    for row in range(len(train_text)):
        train_data.append([train_text[row], train_lable[row]])
    train_data = pd.DataFrame(data=train_data, columns=["text", "lable"])
    train_data = np.array(train_data.groupby(["lable"])["text"])            #根据'lable'将微博划分成三组
    negText = list(train_data[0, 1])
    neuText = list(train_data[1, 1])
    posText = list(train_data[2, 1])
    data = np.array([negText, neuText, posText])                #3*15000*2
    return data

def getMaxStdDev(countList):                                        #计算单个词在三种情况下的标准差率、较大频率值和赋分
    retList = [0, 0]
    minFreq = 0.015                                                # 设定阈值，过滤频率值极低的词汇
    for i in range(1, 3):                                          #0、1、2依次表示-1消极、0中立、1积极
        if max(countList) > minFreq:
            if (countList[i] > countList[0]):
                retList[i - 1] = round(countList[i] / countList[0], 1)
            else:
                retList[i - 1] = -round(countList[0] / countList[i], 1)
            if (abs(retList[i - 1]) > 4):
                retList[i - 1] = (retList[i - 1] * 30 // 3) / 10  # 对大数近似处理
            if (abs(retList[i - 1]) > 7):
                retList[i - 1] = round(retList[i - 1], 0)  # 对大数近似处理
            if (abs(retList[i - 1]) > 15):
                retList[i - 1] = (retList[i - 1] // 4) * 4  # 对大数近似处理
    return retList



def getGroupWordsFrequency(data):                               #统计得到各组自身词频和与相邻组的相关数据，用于提取重要的关键词
    lables = ['countNeg', 'countNeu', 'countPos']               # 用于构造字符串
    flag = 0
    jieba.load_userdict('addWords.TXT')                         #添加自定义词汇
    # # 以下停用词是成对出现的字符中的另一半，同样为了减少特征冗余
    stop = ['”', '】', '）', '》', ')']
    # 将以下常见微博用语统称为“转发”
    repost = ['转发', '快转', 'Repost', 'RepostWeibo']
    i = -1
    diction = []
    print('开始')
    for lable in data:                                          #data由textFormalize得到
        i = i+1
        wordNum = 0                                             #该组的词汇个数（含重复）
        myDiction = dict()
        for text in lable:                                      #遍历处理该组的每一句
            words = jieba.lcut(text)
            for word in words:                                  #遍历处理该句的每一词
                wordNum = wordNum + 1
                if(word in stop):                               #为停用词则跳过
                    continue
                if (word.isdecimal()):                          #数字归为“数字“，减少冗余
                    myDiction['数字'] = myDiction.get('数字', 0) + 1
                elif (word in ['##']):                          #连在一起的#记作两个#
                    myDiction['##'] = myDiction.get('##', 0) + 2
                elif (word in repost):                          #转发格式归为“转发“，减少冗余
                    myDiction['转发'] = myDiction.get('转发', 0) + 1
                else:                                           #其他词汇则按自身处理
                    myDiction[word] = myDiction.get(word, 0) + 1
        temp = []
        for key in myDiction.keys():                            #遍历处理该组的字典
            temp.append([key, myDiction[key]])                  #转换为list
        myDiction = pd.DataFrame(data=temp, columns=['word', lables[i]])#转换为df
        myDiction[lables[i]] = myDiction[lables[i]] / wordNum * 1000    #将频数转换后放大后的频率值
        diction.append(myDiction)

    myDiction = pd.merge(diction[0], diction[1], how='inner')   #将三组的df按交集合并为一个df
    myDiction = pd.merge(myDiction, diction[2], how='inner')
    rating = []
    for index, row in myDiction.iterrows():
        retList = getMaxStdDev([row[lables[i]] for i in range(3)])
        if (retList[0]==0):
            continue
        rating.append([row['word'], retList[0], retList[1]])
    tempDf = pd.DataFrame(data=rating, columns=['word', 'neu/neg', 'pos/neg'])
    tempDf.sort_values(by=['neu/neg', 'pos/neg'], ascending=[False, False], inplace=True)  # 分数名列前茅者作为特征词
    tempDf.reset_index(drop=True, inplace=True)
    classID = -1
    oneKeep = -1
    twoKeep = -1
    refoIDList = []
    classIDList = []
    for index, row in tempDf.iterrows():
        if (row[1] != oneKeep):
            classID = classID + 1
            oneKeep = row[1]
            twoKeep = row[2]
        elif (row[2] != twoKeep):
            classID = classID + 1
            twoKeep = row[2]
        refoIDList.append(index)
        classIDList.append(classID)
    classIDList = pd.DataFrame(classIDList)
    refoIDList = pd.DataFrame(refoIDList)
    classIDList = pd.concat([refoIDList, classIDList], axis=1)
    tempDf = pd.concat([classIDList, tempDf], axis=1)
    with open('key.csv', 'w', newline='', encoding='UTF_8_sig') as f:
        writer = csv.writer(f)
        for index, row in tempDf.iterrows():
            writer.writerow(row)
    return tempDf




def getData(path):                                              #用于读取训练集和测试集的原始csv文件
    with open(path, 'r', encoding='UTF-8')as f:
        csvReader = csv.reader(f)
        dataRaw = list(csvReader)
        dataNum = len(dataRaw)
        Text = []
        Lable = []
        for row in range(1, dataNum):
            lable = dataRaw[row][1]                             #读取每一条微博的分类
            text = dataRaw[row][0]                              #读取并规范每一条微博的原文
            # text = deleteTopic(text)
            text = deleteOthers(text)
            text = deleteWeiboFormat(text)
            text = textUpper(text)
            Lable.append(lable)
            Text.append(text)
        f.close()
    return Text, Lable                                          #返回内容和标签


def getTextVec(data, key_df):                                 #用于获取训练集\测试集的输入特征矩阵
    # lenKey = key_df.iloc[-1, 1]+1
    keyWords = list(key_df.iloc[:, 2])
    lenKey = len(keyWords)
    jieba.load_userdict('addWords.TXT')
    stop = ['”', '】', '）', '》', ')']
    repost = ['转发', '轉發', '快转', 'Repost', 'RepostWeibo']
    i = 0
    textVec = []
    for Text in data:                                           #遍历数据的每一条文本
        i = i+1
        print(i)
        text = [0]*lenKey
        words = jieba.lcut(Text)
        num = 0                                                 #记录一条文本的三种总关键词数
        for word in words:
            if (word in stop):
                continue
            if (word.isdecimal()):
                word = '数字'
            # elif (word in stars):
            #     word = '明星'
            elif (word in ['##']):
                word = '##'
            elif (word in repost):
                word = '转发'
            try:
                word2 = keyWords.index(word)
                word3 = key_df.iloc[word2, 1]
                # text[word3] = text[word3] + 1  # 关键词出现一次，特征矢量对应位置则加一
                text[word2] = text[word2] + 1  # 关键词出现一次，特征矢量对应位置则加一
                num = num + 1
            except ValueError:
                continue
        # if (num != 0):                                          #如果该句子的该类关键词数不为0，则将频数转换为频率
        #     for row in range(len(text)):
        #         text[row] = text[row]/num
        textVec.append(text)
    return textVec


def cmShow(shouldBe, testResult):
    classes = ['消极', '中立', '积极']                             #坐标刻度
    confusion = confusion_matrix(y_true=shouldBe, y_pred=testResult)
    indices = range(len(confusion))
    plt.xticks(indices, classes)                                #第一个表示坐标的显示顺序，第二个坐标刻度
    plt.yticks(indices, classes)
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.title('Confusion matrix')
    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    for first_index in range(len(confusion)):                   #第几行
        for second_index in range(len(confusion[first_index])): #第几列
            plt.text(first_index, second_index, confusion[first_index][second_index])

    plt.imshow(confusion, cmap=plt.cm.Greens)                    #颜色风格为绿
    plt.colorbar()
    plt.show()





text, lable = getData('train_fixed.csv')
train_text, test_text, train_lable, test_lable = train_test_split(text, lable, train_size=0.9, test_size=0.1, random_state=1)
print(len(train_text), len(test_text))
train_data = textFormalize(train_text, train_lable)
key_df = getGroupWordsFrequency(train_data)
train_vec = getTextVec(train_text, key_df)
print('Train set has been loaded.')
test_vec = getTextVec(test_text, key_df)
print('Test set has been loaded.')
clf = naive_bayes.MultinomialNB()                                      #模型选择
print("Training")
classifier = clf.fit(train_vec, train_lable)                                        #训练模型
joblib.dump(clf, 'mnb.model')                                       #下载模型
clf = joblib.load('mnb.model')
print("Predicting")
score = clf.predict(test_vec)                                        #获得测试结果的概率值
errScore = 0
score = [eval(x) for x in score]
shouldBe = [eval(x) for x in test_lable]                              #实际值
for index in range(len(score)):
    result = score[index]
    if(result != shouldBe[index]):
        errScore = errScore + 1
        print(index, '预测值:', result, '实际值:', shouldBe[index], test_text[index])
    else:
        print('OK')
errScore = errScore/len(score)                                           #计算错误率
print('错误率', errScore)
cmShow(shouldBe, score)                                                    #打印混淆矩阵图


testResult = pd.DataFrame(score)

testText = pd.DataFrame(test_text)
testResult = pd.concat([testText, testResult], axis = 1)                        #输出分类结果csv文件
with open('testResult.csv', 'w', newline='', encoding='UTF_8_sig') as f:
    writer = csv.writer(f)
    for index, row in testResult.iterrows():
        writer.writerow(row)








