import  numpy as np 
from functools import reduce

'''
函数实现：
创建实验样本
Returns:
	postingList - 实验样本切分的词条
	clas sVec - 类别标签向量
类别 0： 非侮辱性  1：侮辱性
'''

def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1] 
	return postingList,classVec		

'''
将切分好的实验词条整理成不重复的词汇表

input:
	dataSet  样本数据
returns：
	vocabSet  词汇表

利用python中的set数据类型
求并集利用“  | ”运算
'''

def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

'''
根据词汇表，将inputSet（输入的文档）转化为向量，
表示词汇表中的单次在文档中是否出现 出现 1  不出现 0
input：
	vocabList
	inputSet 
return:
	returnVec 

'''

def  setOfWords2Vec(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList :
			returnVec[vocabList.index(word)] = 1
		else: print("the word %s is not nin my Vocabulary" % word)
	return returnVec

'''
计算概率
trainMatrix     文档矩阵
trainCategory   文档所属类别

'''
def  trainNB0(trainMatrix, trainCategory):
	#文档数目
	numTrainDocs = len(trainMatrix)
	#词汇表的长度
	numWords = len(trainMatrix[0])
	#文档属于侮辱列的概率
	pAbusive=sum(trainCategory)/float(numTrainDocs)
	p0Num=np.ones(numWords);p1Num=np.ones(numWords)

	p0Denom=2.0;p1Denom=2.0
	'''
	遍历所有文档，只要词汇表中的词在侮辱性文档中出现则对应的词汇数目就会加1，文档总的侮辱性词汇数目也会加1

	'''
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])

	p1Vect = np.log(p1Num/p1Denom)
	p0Vect = np.log(p0Num/p0Denom)

	return p1Vect, p0Vect, pAbusive


'''
定义分类器
vec2Classify - 待分类的词条数组
p0Vec - 侮辱类的条件概率数组
p1Vec -非侮辱类的条件概率数组
pClass1 - 文档属于侮辱类的概率 pAb


'''
def classifyNB(vec2Classify, p0vec,p1vec,pClass1):
	#对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
	p1 = sum(vec2Classify * p1vec) + np.log(pClass1)
	p0 = sum(vec2Classify * p0vec) + np.log(1.0 - pClass1)
	print('p1:',p1)
	print('p0:',p0)
	if p1 > p0:
		return 1
	else:
		return 0


if __name__ == '__main__':

	listPosts,listClasses = loadDataSet()
	myVocabList = createVocabList(listPosts)
	#print(myVocabList)
	#print(setOfWords2Vec(myVocabList,listPosts[0]))
	
	trainMat=[]
	for postinDoc in listPosts:
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
	p1V, p0V, pAb=trainNB0(trainMat,listClasses)
	#print(p0V,p1V,pAb)
	testEntry =['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
	thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
	if classifyNB(thisDoc,p0V,p1V,pAb):
		print(testEntry,'属于侮辱类')
	else:
		print(testEntry,'属于非侮辱类')	
							
	testEntry =['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))			
	if classifyNB(thisDoc,p0V,p1V,pAb):
		print(testEntry,'属于侮辱类')										
	else:
		print(testEntry,'属于非侮辱类')	
