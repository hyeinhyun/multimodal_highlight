#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:17:47 2020

@author: hihyun
"""

import json
import urllib.request
from datetime import datetime
from datetime import timedelta
import re

def collectClip(v_id,  Clientid, File):
    print(v_id)
    #url = "https://api.twitch.tv/kraken/video/top?channel=" + channel + "&limit=" + str(lim)
    url="https://api.twitch.tv/kraken/videos/"+str(v_id)
    print(url)
    req = urllib.request.Request(url, headers = {"Client-ID": Clientid, "Accept" : "application/vnd.twitchtv.v5+json"})
    u = urllib.request.urlopen(req)
    c = u.read().decode('utf-8')
    js = json.loads(c)
    #print(js)

    return collectChat(js, Clientid, File,v_id)

def collectChat(j, clientId, f,v_id):
    #id=v_id[:-1]
    id=v_id
    print(id)
    #id = j['clips'][num]['vod']['id']
    #offset = j['clips'][num]['vod']['offset']
    #duration = j['clips'][num]['duration']
    #print(duration)

    cursor = ""
    count = 0

    while(1):
        try:
            url2 = ""
            if count == 0:
                url2 = "https://api.twitch.tv/kraken/videos/" + str(id) + "/comments"
            else:
                url2 = "https://api.twitch.tv/kraken/videos/" + str(id) + "/comments?cursor=" + str(cursor)
            req2 = urllib.request.Request(url2, headers = {"Client-ID": clientId, "Accept" : "application/vnd.twitchtv.v5+json"})
            u2 = urllib.request.urlopen(req2)
            c2 = u2.read().decode('utf-8')
            j2 = json.loads(c2)
            #print(j2)
            endCount = 0
            try:
                for number, com in enumerate(j2['comments']):



                    dateString = com['created_at']
                    if "." in dateString:
                        dateString = re.sub(r".[0-9]+Z","Z", dateString)
                    date = datetime.strptime(dateString, "%Y-%m-%dT%H:%M:%SZ")
                    #print(date)
                    leng=j['length']
                    off=com['content_offset_seconds']
                    if float(leng)<float(off):
                        return 0

                    f.write(str(j['title']) + "\t" +
                            str(j['game']) + "\t" +
                            str(j['views']) + "\t" +
                            str(j['length']) + "\t" +
                            str(date + timedelta(hours=9)) + "\t" +
                            str(com['content_offset_seconds'])+"\t"+
                            str(com['message']['body']) + "\n")
                    print(date)

            except Exception as e:
                print(e)

            if endCount == 1:
                break

            if j2['_next']:
                cursor = j2['_next']

            count = count + 1

        except Exception as e:
            print(e)
                
if __name__ == "__main__":

#     with open('video_list.txt') as f:
#         v_li=f.readlines()
#     for i in v_li:
#         print(i)
#         v_id=i
#         file_path=v_id+'.txt'
#         file = open(file_path, "w", encoding="utf-8")
#         Limit = 1
#         ClientId = "pnbcpj842zq89uy7hlhkakepr6xej7" # Client id 추가 #
#         collectClip(v_id, ClientId, file)
#         file.close()

    file=open('test.txt',"w",encoding="utf-8")
    ClientId = "pnbcpj842zq89uy7hlhkakepr6xej7" # Client id 추가 #

    collectClip("496371424",ClientId,file)
    
    
    
