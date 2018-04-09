
'''
垃圾邮件分类
email/spam/%d.txt     25个垃圾邮件
email/ham/%d.txt      25个非垃圾邮件
'''
import numpy as np 
import random 
import re


def createVocabList(dataSet):
    vocabSet = set([])  					
    for document in dataSet:				
        vocabSet = vocabSet | set(document) 
    return list(vocabSet)



def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)									
    for word in inputSet:												
        if word in vocabList:											
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec	

'''
修改setOfWords2Vec   下者元素可以重复出现
''' 
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)										
    for word in inputSet:												
        if word in vocabList:											
            returnVec[vocabList.index(word)] += 1
    return returnVec	


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)							
    numWords = len(trainMatrix[0])							
    pAbusive = sum(trainCategory)/float(numTrainDocs)		
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)	#创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0; p1Denom = 2.0                        	#分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:							#统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:												#统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)							#取对数，防止下溢出          
    p0Vect = np.log(p0Num/p0Denom)          
    return p0Vect,p1Vect,pAbusive	

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    	#对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

''''
'\W'  正则表达式 匹配任何非单词字符
'\w'            匹配包括下划线的任何单词字符
功能： 接受大字符串，并将其解析为字符串列表，同时去掉少于两个字符串的字符串，并转换为小写字母
'''
def textParse(bigString):
	listofTokens = re.split(r'\W',bigString)
	return [tok.lower() for tok in listofTokens if len(tok) > 2]

def spamTest():
	docList=[]; classList=[]; fullText=[]
	#遍历25个txt文件
	for i in range(1,26):
		wordList = textParse(open('email/spam/%d.txt' % i,'r').read())
		docList.append(wordList)
		fullText.append(wordList)
		#垃圾邮件标记
		classList.append(1)
		wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
		docList.append(wordList)
		fullText.append(wordList)
		classList.append(0)

	vocabList = createVocabList(docList)
	#trainingset 存储测试集的索引
	trainingSet = list(range(50)); testSet=[]
	#50个邮件中飞机挑选40个作为测试集
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	#创建训练集
	trainMat = []; trainClasses =[]
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])



	p0V, p1V, pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList,docList[docIndex])
		if classifyNB(np.array(wordVector), p0V,p1V,pSpam) != classList[docIndex]:
			errorCount +=1
			print("分类错误的测试集： ", docList[docIndex])

	print('错误率： %.2f%%' %(float(errorCount) / len(testSet) * 100))

if __name__ == '__main__':
	spamTest()
