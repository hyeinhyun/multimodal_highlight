#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:24:06 2020

@author: hihyun
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords




def time_ds(time,default):
    default_date=default.split(' ')[0]
    default_time=default.split(' ')[1]
    d_time=datetime(int(default_date.split('-')[0]),int(default_date.split('-')[1]),int(default_date.split('-')[2]),
    int(default_time.split(':')[0]),int(default_time.split(':')[1]),int(default_time.split(':')[2]))
    
    time_date=time.split(' ')[0]
    time_time=time.split(' ')[1]
    t_time=datetime(int(time_date.split('-')[0]),int(time_date.split('-')[1]),int(time_date.split('-')[2]),
    int(time_time.split(':')[0]),int(time_time.split(':')[1]),int(time_time.split(':')[2]))
    delta=t_time-d_time
    return delta.seconds
    

def read(file_name):
    tempLine=[]
    vocab=OrderedDict()
    with open(file_name) as f:
        #default_time=f.readlines()[1].split(']')[0][1:]
        for lineNo,line in enumerate(f.readlines()[1:]):
            text=line.split(']')[1]
            text=re.sub(r'\<[^)]*\>', '', text)
            text=re.sub('[^a-zA-Z]', ' ', text)
            text=text.lower()
            text=text.split(' ')
            stops = set(stopwords.words('english'))
            no_stops = [word for word in text if not word.strip() in stops]
            no_stops=[word for word in no_stops if word!='']
            if len(no_stops)>=3:  
                temp={"time":int(time_ds(line.split(']')[0][1:],default_time)),
                    "text":no_stops,"lineno":lineNo+1}
                tempLine.append(temp)
                for item in temp["text"]:
                    if item not in vocab:
                        vocab[item]=0
    f.close()
    return tempLine,vocab

def data_preprocessing(temp,vocab):
    vocabulary_list=[]
    for line in temp:
        _vocabulary = copy.copy(vocab)
        for item in line["text"]:
            if item in _vocabulary:
                _vocabulary[item]+=1
        vocabulary_list.append(list(_vocabulary.values()))
        print_raw_comment(temp)

        # print(len(self.lines))
    print(len(vocabulary_list))

    return temp,np.array(vocabulary_list),len(vocab),vocab
def print_raw_comment(lines):
    with open("./raw.txt","w") as f:
        for line in lines:
            f.write(" ".join(line["text"])+"\n")

def store_word2vec_calc(file="./raw.txt"):
    sentences = word2vec.Text8Corpus(file)  # 加载语料
    model = word2vec.Word2Vec(sentences, size=300,min_count=1)
    fw = open("./word2vec_model", "wb")
    pickle.dump(model, fw)
    fw.close()
    
    
import numpy as np
from gensim.models import word2vec
import logging
import pickle
from operator import itemgetter, attrgetter
from collections import OrderedDict



def cos_distance(vector1,vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))



def grab_word2vec_calc():
    fr = open("./word2vec_model", "rb")
    model = pickle.load(fr)
    fr.close()
    return model


class Vertice(object):
    def __init__(self):
        self.S=set()
        self.time=None
        self.index=0
        self.comment=None
        self.sentence_vec=None

class Edge(object):
    def __init__(self):
        self.x=None
        self.y=None
        self.w=None

class SAGModel(object):

    def __init__(self):#value initialize
        self.vertice_list=[]
        self.edge_list=[]
        self.edge_dict={}
        self.gamma_t=0.11
        self.rio=0.36
        self.tag_number=20
        #self.comment_popularity=[]
        self.T=50

    #check
    def _initialize_vertice(self,word_2_vec,temp,vocab):
        #temp,np.array(vocabulary_list),len(vocab),vocab
        lines,self.vocabulary_list,self.vocabulary_size,self.vocabulary=data_preprocessing(temp,vocab)
        self.N=len(lines)
        for index,line in enumerate(lines):
            v=Vertice()
            v.S.add(index)
            v.time,v.index,v.comment =line["time"],index,line["text"][:]
            #calc mean sentence vector by word2vec
            total=np.zeros(300)
            for item in line["text"]:
                if item in word_2_vec:
                    total+=word_2_vec[item]
            v.sentence_vec=total/len(line["text"])
            self.vertice_list.append(v)
        #print(self.vertice_list)
        return self.vertice_list


    def _initialize_edge(self):
        for i,vertice in enumerate(self.vertice_list):
            for j in range(i+1,len(self.vertice_list)):
                e=Edge()
                e.x=i
                e.y=j
                e.w=cos_distance(vertice.sentence_vec,self.vertice_list[j].sentence_vec)\
                    *np.exp(-1*self.gamma_t*(np.abs(vertice.time-self.vertice_list[j].time)))
                #print(e.w)
                if e.w>=0.3:
                    self.edge_list.append(e)
                    self.edge_dict[str(i)+","+str(j)]=e
        self.edge_list=sorted(self.edge_list,key=attrgetter('w'),reverse=True)



    def initialize(self):
        word_2_vec = grab_word2vec_calc()
        temp,vocab=read('./hasanabi.log')
        self._initialize_vertice(word_2_vec,temp,vocab)
        self._initialize_edge()


    def _calc_S_size(self):
        S_dict=OrderedDict()

        for vertice in self.vertice_list:
            #print("vertice.S")
            #print(list(vertice.S))
            _key=",".join([str(v) for v in sorted(list(vertice.S))])
            if _key in S_dict:
                S_dict[_key] +=1
            else:
                S_dict[_key]=1
        #print (S_dict)
        #print("len S_dict")
        #print(len(S_dict))
        return S_dict

    def _calc_popularity(self):
        comment_popularity=[]
        S_dict=self._calc_S_size()
        #print(len(S_dict))
        #print(list(S_dict.values()))
        total=1
        for item in S_dict.values():
            total*=item

        _denumberator=np.power(total,1/len(S_dict))
        #print(_denumberator)
        for vertice in self.vertice_list:
            comment_popularity.append(len(vertice.S)/_denumberator)
        #print("popularity")
        #print(comment_popularity)
        return np.array(comment_popularity)



    def _cacl_M_n(self):
        m=np.zeros((self.N,self.N))
        for i in range(0,self.N):
            for j in range(0,self.N):
                if self.vertice_list[i].S == self.vertice_list[j].S and i!=j:
                    if str(i)+","+str(j) in self.edge_dict:
                        m[i][j]=self.edge_dict[str(i)+","+str(j)].w
                    elif str(j)+","+str(i) in self.edge_dict:
                        m[i][j] = self.edge_dict[str(j) + "," + str(i)].w
        #print("m")
        #print(m)
        return m


    def _calc_I(self,m):
        I=np.zeros((2*self.T+1,self.N))
        I[0,:]=1

        for k in range(1,self.T+1):
            for i in range(self.N-1,-1,-1):
                if i==self.N-1:
                    I[2 * k - 1][i] = I[2 * k - 2][i]
                else:
                    total=0
                    for j in range(i+1,self.N):
                        total+=m[i][j]*I[2*k-1][j]
                    I[2 * k - 1][i] = I[2 * k - 2][i]+total

            for i in range(0,self.N):
                if i==0:
                    I[2*k][i]=I[2*k-1][i]/(I[2*k-1][i])
                else:
                    total=0
                    for j in range(0,i):
                        total+=m[j][i]*I[2*k][j]
                    I[2*k][i]=I[2*k-1][i]/(I[2*k-1][i]+total)
        #print("I")
        #print(I)
        #print("I[20]")
        #print(I[self.T])
        return I[self.T]


    def _calc_SW_IDF(self,W):
        result=[]
        for i in range(0,self.vocabulary_size):
            sum_w_j=0

            count=0
            for j in range(0,self.N):
                if self.vocabulary_list[j][i]>0:
                    sum_w_j+=W[j]
                    count+=1
            #print(i)
            #print(self.vocabulary_list.shape)
            #result.append((np.log(self.N/(1+np.sum(self.vocabulary_list[:,i])))*sum_w_j,i))
            result.append((np.log(self.N / (1 + count)) * sum_w_j, i))
        #print("result")
        #print(result)
        return result

    def _tag_extraction(self):
        self.initialize()
        for i,edge in enumerate(self.edge_list):
            #print("haha")
            #print(self.vertice_list[edge.x].S)
            #print(self.vertice_list[edge.y].S)
            #print("hehe")

            s_all=list(self.vertice_list[edge.x].S | self.vertice_list[edge.y].S)
            s1_length=len(self.vertice_list[edge.x].S)
            s2_length=len(self.vertice_list[edge.y].S)
            #print(s_all)
            total=0.0
            for i,item in enumerate(s_all):
                for j in range(i+1,len(s_all)):
                    i_j=str(item)+","+str(s_all[j])
                    j_i = str(s_all[j]) + "," + str(item)
                    if i_j in self.edge_dict:
                        total+=self.edge_dict[i_j].w
                    elif j_i in self.edge_dict:
                        total += self.edge_dict[j_i].w

            if total/((s1_length+s2_length)*((s1_length+s2_length)-1)/2)>self.rio:
                for item in s_all:
                    self.vertice_list[item].S|=set(s_all)
                    #self.vertice_list[edge.y].S|=set(s_all)

        m=self._cacl_M_n()
        P=self._calc_popularity()
        I=self._calc_I(m)
        W=P*I

        #print("W")
        #print(W)
        #print(len(W))
        #print(self.vocabulary.keys())

        #store_w(W)
        #W=grab_w()
        result=self._calc_SW_IDF(W)
        print(result)
        self._display_tag(result)

    def _display_tag(self,result):
        with open("data/result.txt","w") as f:
            #for item in sorted(result,key=lambda x:x[0],reverse=True)[:self.tag_number]:
            #print("sorted(result, key=lambda x: x[0], reverse=True)")
            #print(sorted(result, key=lambda x: x[0], reverse=True))
            for item in sorted(result, key=lambda x: x[0], reverse=True)[:self.tag_number]:
                f.write(list(self.vocabulary.keys())[item[1]])
                f.write("\n")

import copy   
from datetime import datetime, timedelta
from collections import OrderedDict
import numpy as np
from gensim.models import word2vec
import pickle
file_name="./hasanabi.log"
#default time 미리 해야
f=open(file_name)
default_time=f.readlines()[1].split(']')[0][1:]
f.close()
temp,vocab=read(file_name)
a,b,c,d=data_preprocessing(temp,vocab)
store_word2vec_calc()
SAGModel()._tag_extraction()

b[2696][1]

    
"""
                     for item in temp["text"]:
                        if item not in vocabulary:
                            vocabulary[item]=0
    f=open(file_name)
    default_time=f.readlines()[1].split(']')[0][1:]
    test="14:14:42"
    time1 = datetime(2020,3,7,14,14,42)
    time2 = datetime(2020,3,7,14,14,41)
    (time1-time2).seconds
    
    
#data preprocessing
test=pd.read_csv('./jake_test.csv')
test_word=[]
test=test.rename(columns={"# Log started: 2020-02-29 15:58:27 +0900": "log"})
test['log']

for i in test['log']:
    #print("before : ",i)
    i=re.sub(r'\[[^)]*\]', '', i)#[] 제거
    i=re.sub(r'\<[^)]*\>', '', i)#<> del
    i=re.sub('[^a-zA-Z]', ' ', i) #영어아닌거 제거
    i.lower()
    #print("\n after : ",i)
    test_word.append(i.strip())
test_word=[x for x in test_word if x]
nltk.download('stopwords')
stops = set(stopwords.words('english'))
no_stops = [word for word in test_word if not word in stops]

for i in range(len(no_stops)):
    if test_word[i] != no_stops[i]:
        print("\n before : ",test_word[i])
        print("after : ",no_stops[i])


stemmer = nltk.stem.SnowballStemmer('english')
stemmer_words = [stemmer.stem(word) for word in no_stops]
stemmer_words =[x for x in stemmer_words if x!='d']
  """  
#with open(infile) as f:
   # f = f.readlines()
   
   
   #vocabulary
   