# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:04:19 2020

@author: Gangmin
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import OrderedDict
from datetime import datetime, timedelta

import copy


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
                if not text in stops and not len(text) == 0:
                    check = True
                    textt = text
                    lineNot = lineNo
                    
                    
            if(  check == True and lineNo%2 == 0):
                line = line.rstrip('\n') 
                temp={"time":line,"text":textt,"lineno":lineNot+1}
                temp={"time":line,"text":[textt],"lineno":lineNot+1}
                tempLine.append(temp)
                check = False
                #print(temp["time"])
                if temp['text'] not in vocab:
                        vocab[temp['text']]=0
                if temp['text'][0] not in vocab:
                        vocab[temp['text'][0]]=0
        
    #print(tempLine)
    print(vocab)
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
    
