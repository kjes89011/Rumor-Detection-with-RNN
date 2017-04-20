#-*- coding: utf-8 -*-
import json,os
from os import listdir
import time
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# Neural Network
import numpy as np
np.random.seed(3)	#固定seed讓每次的random都一樣
# For add layer sequentally
from keras.models import Sequential
# For add fully connected layer
from keras.layers import Dense,Activation,SimpleRNN
#keras 的手寫數字1~9的數據集
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam,Adagrad

mypath = "./data/"
Evenlist = listdir(mypath)
TIME_STEPS = 7
IMPUT_SIZE = 300	
BATCH_SIZE = 1
BATCH_INDEX = 0
OUTPUT_SIZE = 2
CELL_SIZE = 150
LR = 0.01

def ContinuousInterval(intervalL):
	maxInt = []
	tempInt = [intervalL[0]]
	for q in range(1,len(intervalL)):
		if intervalL[q]-intervalL[q-1] > 1:
			if len(tempInt) > len(maxInt):
				maxInt = tempInt
			tempInt = [intervalL[q]]
		else:
			tempInt.append(intervalL[q])
	if len(maxInt)==0:
		maxInt = tempInt
	return maxInt

totalData = []
totalDataLabel = []
counter = 0
totalDoc = 0
totalpost = 0
tdlist1 = 0
Pos = 0
Neg = 0
maxpost = 0
minpost = 62827
for event in Evenlist:
	totalDoc += 1
	fe = open(os.path.join(mypath,event,"event.json"),"r")
	EventJson = json.load(fe)
	Label = EventJson["label"]
	TweetList = []
	TidList = listdir(os.path.join(mypath,event))
	if len(TidList) == 1:
		tdlist1 += 1
		continue
	if len(TidList) >= maxpost:
		maxpost = len(TidList)
	if len(TidList) <= minpost:
		minpost = len(TidList) 
	for TweetId in TidList:
		if TweetId == "event.json":
			continue
		totalpost +=1
		ft = open(os.path.join(mypath,event,TweetId,"source-tweets",TweetId+".json"),"r")
		TweetJson = json.load(ft)
		tweetText = TweetJson["text"]
		#Convert time to Unix timestamp that easilly to caculate
		Time = time.mktime(datetime.datetime.strptime(TweetJson["created_at"], "%a %b %d %H:%M:%S +0000 %Y").timetuple())
		TweetList.append({"text":tweetText,"time":Time})
		ft.close()
	if Label == '0':
		Pos +=1
	else:
		Neg +=1
	#Sort by time
	TweetList = sorted(TweetList, key=lambda k: k['time'])

	#find Time Invertal of tweet
	#Initialization
	TotalTimeLine = TweetList[-1]['time']-TweetList[0]['time']
	IntervalTime = TotalTimeLine/TIME_STEPS
	k = 0
	PreConInt = []
	while True:
		k += 1
		tweetIndex = 0
		output = []
		if TotalTimeLine == 0:
			output.append(''.join(tweet["text"] for tweet in TweetList))
			break
		Start = TweetList[0]['time']
		Interval = int(TotalTimeLine/IntervalTime)
		Intset = []
		for inter in range(0,Interval):
			empty = 0
			interval = []
			for q in range(tweetIndex,len(TweetList)):
				if TweetList[q]['time'] >= Start and TweetList[q]['time'] < Start+IntervalTime:
					empty += 1
					interval.append(TweetList[q]["text"])
				#紀錄超出interval的tweet位置，下次可直接從此開始
				elif TweetList[q]['time'] >= Start+IntervalTime:
					tweetIndex = q-1
					break
			# empty interval
			if empty == 0:
				output.append([])
			else:
				#add the last tweet
				if TweetList[-1]['time'] == Start+IntervalTime:
					interval.append(TweetList[-1]["text"])
				Intset.append(inter)
				output.append(interval)
			Start = Start+IntervalTime
		ConInt = ContinuousInterval(Intset)
		if len(ConInt)<TIME_STEPS and len(ConInt) > len(PreConInt):
			IntervalTime = int(IntervalTime*0.5)
			PreConInt = ConInt
			if IntervalTime == 0:
				output = output[ConInt[0]:ConInt[-1]+1]
				break
		else:
			# print(len(ConInt))
			output = output[ConInt[0]:ConInt[-1]+1]
			break
	counter+=1
	fe.close()
	# Debug
	# for batchtweet in output:
	# 	print(batchtweet)
		# for tweet in batchtweet:
		# 	print tweet["text"]

	# 不確定是不是把Interval的所有字都串在一起
	for q in range(0,len(output)):
		output[q] = ''.join(s for s in output[q])

	#Caculate Tfidf
	vectorizer = CountVectorizer()
	transformer = TfidfTransformer()

	tf = vectorizer.fit_transform(output)
	tfidf = transformer.fit_transform(tf)
	# Debug
	# print(tfidf.toarray())
	Allvocabulary = vectorizer.get_feature_names()
	# print(vectorizer.get_feature_names())
	Input = []
	
	for interval in tfidf.toarray():
		interval = sorted(interval,reverse=True)
		while len(interval) < IMPUT_SIZE:
			interval.append(-1)
		Input.append(interval[:IMPUT_SIZE])
	if len(Input) < TIME_STEPS:
		for q in range(0,TIME_STEPS-len(Input)):
			Input.insert(0,[-1] * IMPUT_SIZE)
	totalData.append(Input[:TIME_STEPS])
	totalDataLabel.append(Label)
	
print("totalDoc : "+str(totalDoc))
print("tdlist1 : "+str(tdlist1))
print("Pos : "+str(Pos))
print("Neg : "+str(Neg))
print("totalpost : "+str(totalpost))
print("maxpost : "+str(maxpost))
print("minpost : "+str(minpost))

X_train = np.array(totalData[:int(counter/5*4)])
# for q in X_train:
# 	print(q)
y_train = np.array(totalDataLabel[:int(counter/5*4)])
# for q in y_train:
# 	print(q)
print(X_train.shape)
X_test = np.array(totalData[int(counter/5*4):])
y_test = np.array(totalDataLabel[int(counter/5*4):])
print(X_test.shape)
#RNN
# X_train = X_train.reshape(-1,TIME_STEPS,IMPUT_SIZE)		#normalize
# X_test = X_test.reshape(-1,TIME_STEPS,IMPUT_SIZE)		#normalize
y_train = np_utils.to_categorical(y_train,num_classes = 2)
y_test = np_utils.to_categorical(y_test,num_classes = 2)
#print(y_train.shape)
model = Sequential()

#RNN cell
model.add(SimpleRNN(CELL_SIZE,input_shape=(TIME_STEPS,IMPUT_SIZE)))

#output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

#optimizer
Adagrad = Adagrad(LR)

model.compile(optimizer = Adagrad,loss = 'mean_squared_error',metrics = ['accuracy'])

#train
print("Training---------")

model.fit(X_train,y_train,epochs=600,batch_size=BATCH_SIZE)

print("\nTesting---------")
cost,accuracy = model.evaluate(X_test,y_test,batch_size=y_test.shape[0], verbose=False)
print('test cost: ',cost)
print('test accuracy: ',accuracy)

# for step in range(3001):
# 	X_batch = X_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE, :, :]
# 	Y_batch = y_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE, :]
# 	cost = model.train_on_batch(X_batch,Y_batch)
# 	BATCH_INDEX += BATCH_SIZE
# 	if BATCH_INDEX >= X_train.shape[0] :
# 		BATCH_INDEX = 0
	
# 	if step % 500 == 0 :
# 		#test
# 		print("\nTesting---------")
# 		cost,accuracy = model.evaluate(X_test,y_test,batch_size=y_test.shape[0], verbose=False)
# 		print('test cost: ',cost)
# 		print('test accuracy: ',accuracy)

# W,b = model.layers[2].get_weights()
# print('Weight = ',W,"\nBias = ",b)
