#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:08:53 2020

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
    with open(file_name,encoding="utf-8") as f:
        #default_time=f.readlines()[1].split(']')[0][1:]
        for lineNo,line in enumerate(f.readlines()[1:]):
            try:
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
            except:#마지막
                pass
                        
    f.close()
    return tempLine,vocab
def read_cap(file_name):
    tempLine=[]
    vocab=OrderedDict()
    #nltk.download('stopwords')

    
    with open(file_name,encoding='utf8') as f:
        #default_time=f.readlines()[1].split(']')[0][1:]
        textt = ""
        lineNot = 0
        check = False
        for lineNo,line in enumerate(f.readlines()[1:]):
            if(lineNo%2 == 1):
                text=line
                text=re.sub(r'\<[^)]*\>', '', text)
                text=re.sub('[^a-zA-Z]', ' ', text)
                text = text.strip()
                text=text.lower()
                
                #print(text)
                stops = set(stopwords.words('english'))
                
                
                if not text in stops and not len(text)==0:
                    check = True
                    textt = text
                    lineNot = lineNo
                    
                    
            if(  check == True and lineNo%2 == 0):
                line = line.rstrip('\n') 
                temp={"time":int(line.strip()),"text":[textt],"lineno":lineNot+1}
                tempLine.append(temp)
                check = False
                #print(temp["time"])
                if temp['text'][0] not in vocab:
                        vocab[temp['text'][0]]=0
        
    #print(tempLine)
    #print(tempLine)
    return tempLine,vocab

def sum_temp(tempLine):
    tempList = []
    index = 0
    for temp in tempLine:
        if len(tempList)==0:
            tempList.append(temp)
        else:
            if tempList[index]['time'] == temp['time']:
                tempList[index]['text'] = tempList[index]['text']+temp['text']                   
            else:
                index = index+1
                tempList.append(temp)
    return tempList
def make_od(temp):
    vocab=OrderedDict()
    for i in temp:
        for item in i["text"]:
            if item not in vocab:
                vocab[item]=0
    return vocab
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
    with open("./raw_0325.txt","w") as f:
        for line in lines:
            f.write(" ".join(line["text"])+"\n")

def store_word2vec_calc(file="./raw_0325.txt"):
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



    def initialize(self,word_2_vec,temp,vocab):
        #word_2_vec = grab_word2vec_calc()
        #temp,vocab=read('./hasanabi.log')
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

    def _tag_extraction(self,word_2_vec,temp,vocab):
        self.initialize(word_2_vec,temp,vocab)
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
        self._display_tag(result)
        return result

    def _display_tag(self,result):
        with open("data/result_0325.txt","w",encoding="utf-8") as f:
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


if __name__ == '__main__':
    file_name="./xqcow.log"
    file_cap="./speech_to_text/stt_result.txt"
    #comment 관련
    #default time 미리 해야
    f=open(file_name,encoding="utf-8")
    default_time=f.readlines()[1].split(']')[0][1:]
    f.close()
    temp_comment,vocab_comment=read(file_name)
    temp_cap,vocab_cap=read_cap(file_cap)
    temp_cap=sum_temp(temp_cap)
    temp=temp_comment+temp_cap #sampling 필요
    
    vocab=make_od(temp)
    a,b,c,d=data_preprocessing(temp,vocab)
    store_word2vec_calc()
    word_2_vec=grab_word2vec_calc()
    
    
    result=SAGModel()._tag_extraction(word_2_vec,temp,vocab)
    
    listlist=[]
    for item in sorted(result,key=lambda x: x[0],reverse=True)[:20]:
        listlist.append(item[1])
    
    indexs=[]
    for index,i in enumerate(b):
        for j in listlist:
            if i[j]==1:
                indexs.append(index)
                break
    with open("data/result_sentence_0325.txt","w") as f:
        for i in indexs:
            f.write(str(a[i]))
            f.write('\n')
    #####분산구하기########
    
    #1. 인덱스 구하기
    index=[list(vocab.keys()).index('weird'),
     list(vocab.keys()).index('monkaw'),
     list(vocab.keys()).index('craig'),
     list(vocab.keys()).index('trainwreckstv'),
     list(vocab.keys()).index('lulw'),
     list(vocab.keys()).index('train'),
     list(vocab.keys()).index('stfu'),
     list(vocab.keys()).index('squadq'),
     list(vocab.keys()).index('kkonaw'),
     list(vocab.keys()).index('one'),
     list(vocab.keys()).index('dougzilla'),
     list(vocab.keys()).index('chat'),
     list(vocab.keys()).index('pogu'),
     list(vocab.keys()).index('wtf'),
     list(vocab.keys()).index('try'),
     list(vocab.keys()).index('fuck'),
     list(vocab.keys()).index('el'),
     list(vocab.keys()).index('omegalul'),
     list(vocab.keys()).index('god'),
     list(vocab.keys()).index('clap')]
    
    results1={}
    results2={}
    results3={}
    results4={}
    results5={}


    for idx_index in index:
        list_time1=[]
        list_time2=[]
        list_time3=[]
        list_time4=[]
        list_time5=[]
        for idx,i in enumerate(b):
            if i[idx_index] !=0:
                num=int(a[idx]['time'])
                if num<900:#15
                    list_time1.append(num)
                elif 900<=num and num<1800:#15~30
                    list_time2.append(num)
                elif 1800<=num and num<2700:#30~45
                    list_time3.append(num)
                else:
                    list_time4.append(num)
                list_time5.append(num)#전체
                
              #  print(num)
              #  for k in range(i[idx_index]):
                    #list_time.append(num)
        results1[list(vocab.keys())[idx_index]+'1']=np.var(list_time1)
        results2[list(vocab.keys())[idx_index]+'2']=np.var(list_time2)
        results3[list(vocab.keys())[idx_index]+'3']=np.var(list_time3)
        results4[list(vocab.keys())[idx_index]+'4']=np.var(list_time4)
        results5[list(vocab.keys())[idx_index]]=np.var(list_time5)


"""
def read(file_name):
    tempLine=[]
    vocab=OrderedDict()
    #nltk.download('stopwords')

    
    with open(file_name,encoding='utf8') as f:
        #default_time=f.readlines()[1].split(']')[0][1:]
        textt = ""
        lineNot = 0
        check = False
        for lineNo,line in enumerate(f.readlines()[1:]):
            if(lineNo%2 == 1):
                text=line
                text=re.sub(r'\<[^)]*\>', '', text)
                text=re.sub('[^a-zA-Z]', ' ', text)
                text = text.strip()
                text=text.lower()
                
                #print(text)
                stops = set(stopwords.words('english'))
                
                
                if not text in stops:
                    check = True
                    textt = text
                    lineNot = lineNo
                    
                    
            if(  check == True and lineNo%2 == 0):
                line = line.rstrip('\n') 
                temp={"time":line,"text":[textt],"lineno":lineNot+1}
                tempLine.append(temp)
                check = False
                #print(temp["time"])
                if temp['text'][0] not in vocab:
                        vocab[temp['text'][0]]=0
        
    #print(tempLine)
    #print(tempLine)
    return tempLine,vocab

def sum_temp(tempLine):
    tempList = []
    index = 0
    for temp in tempLine:
        if len(tempList)==0:
            tempList.append(temp)
        else:
            if tempList[index]['time'] == temp['time']:
                tempList[index]['text'] = tempList[index]['text']+temp['text']                   
            else:
                index = index+1
                tempList.append(temp)
    return tempList
    
    

if __name__ == '__main__':
    
    tempLine, vocab = read('C:/Users/Gangmin/Desktop/gangmin/캡스톤/tsvt/speech_to_text/stt_result.txt')
    print(sum_temp(tempLine))
"""
