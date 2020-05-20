from PIL import Image
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS']="/home/ubuntu/2020_1/cap/google_hyein.json"

import io
import os
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image
import csv


def image_to_byte_array(image:Image):
    imgbyteArr=io.BytesIO()
    image.save(imgbyteArr,format='PNG')
    imgbyteArr=imgbyteArr.getvalue()
    return imgbyteArr
def crop_image(img):
    t_b=img.crop((670,10,700,40))
    t_b=image_to_byte_array(t_b)
    
    g_b=img.crop((750,10,810,40))
    g_b=image_to_byte_array(g_b)
    
    k_b=img.crop((910,15,950,50))
    k_b=image_to_byte_array(k_b)
    
    k_r=img.crop((985,15,1025,50))
    k_r=image_to_byte_array(k_r)
    
    g_r=img.crop((1135,10,1200,45))
    g_r=image_to_byte_array(g_r)
    
    t_r=img.crop((1260,10,1285,40))
    t_r=image_to_byte_array(t_r)
    
    status=img.crop((700,200,1200,300))
    status=image_to_byte_array(status)
    
    replay=img.crop((0,0,200,150))
    replay=image_to_byte_array(replay)
    return t_b, g_b,k_b,k_r,g_r,t_r,status,replay

def vision_api(file_name):
    client=vision.ImageAnnotatorClient()
    try:
        img =Image.open(file_name)
        t_b, g_b,k_b,k_r,g_r,t_r,status,replay=crop_image(img)
        img.close()
        dash_list=[t_b, g_b,k_b,k_r,g_r,t_r,status,replay]
        name_list=["tower_blue","gold_blue","kill_blue","kill_red","gold_red","tower_red","status","replay"]
        text_list={}
        for idx,imgbyteArr in enumerate(dash_list):
            image=types.Image(content=imgbyteArr)
            response=client.document_text_detection(image=image)
            texts=response.text_annotations
            try:
                text_list[name_list[idx]]=(texts[0].description.strip()).lower()
            except:
                text_list[name_list[idx]]=''
            """
            for i,text in enumerate(texts):
                print('\n" : {}"'.format(text.description))
                if i==0:
                    text_list[name_list[idx]]=text.description[:-1]
            """


            if response.error.message:
                raise Exception(
                    '{}\nFor more info on error messages, check: '
                    'https://cloud.google.com/apis/design/errors'.format(
                        response.error.message))
        return score_cal(text_list)

    finally:
        client.transport.channel.close()

        
def score_cal(text_list):
    name_list=["tower_blue","gold_blue","kill_blue","kill_red","gold_red","tower_red","status","replay"]
    score_list={}
    if "REPLAY" in text_list['replay']:
        score_list['replay']=True
        score_list['tower_blue']=0
        score_list['gold_blue']=0
        score_list['kill_blue']=0
        score_list['kill_red']=0
        score_list['gold_red']=0
        score_list['tower_red']=0
        status=[0,0]
        score_list['status']=status 

    else:
        score_list['replay']=False
        ######t_b
        try:
            t_b=text_list['tower_blue'].replace('о','0')
            score_list['tower_blue']=int(t_b)
        except:
            #print("can not recognize tower_blue")
            score_list['tower_blue']=0

        ######g_b
        try:
            g_b=text_list['gold_blue'].replace(',','.')
            score_list['gold_blue']=float(g_b[:-1])#k 제거
        except:
            #print("can not recognize gold_blue")
            score_list['gold_blue']=0
        
        ######k_b
        try:
            k_b=text_list['kill_blue'].replace('о','0')        
            score_list['kill_blue']=int(k_b)
        except:
            #print("can not recognize kill_blue")
            score_list['kill_blue']=0

        
        ######k_r
        try:
            k_r=text_list['kill_red'].replace('о','0')        
            score_list['kill_red']=int(k_r) 
        except:
            #print("can not recognize kill_red")
            score_list['kill_red']=0
        
        ######g_r
        try:
            g_r=text_list['gold_red'].replace(',','.')
            score_list['gold_red']=float(g_r[:-1])#k 제거
        except:
            #print("can not recognize gold_red")
            score_list['gold_red']=0
        
        ######tower_red
        try:
            t_r=text_list['tower_red'].replace('о','0')
            score_list['tower_red']=int(t_r)    
        except:
            #print("can not recognize tower_red")
            score_list['tower_red']=0
        
        ######status
        status=[]
        if "dragon" in status or "龙" in status:
            status=[1,0]
            score_list['status']=status
        elif "baron" in status or "男爵" in status:
            status=[0,1]
            score_list['status']=status
        else:
            status=[0,0]
            score_list['status']=status   
    return score_list



#######main
fieldnames = ['replay', 'tower_blue','gold_blue','kill_blue','kill_red','gold_red','tower_red','status']
with open('result4.csv', 'a', newline='') as csvfile:
    fieldnames = ['replay', 'tower_blue','gold_blue','kill_blue','kill_red','gold_red','tower_red','status']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

for i in range(10000):
    if i%100==0:
        print("**********{}************".format(i))
    file_name='/home/ubuntu/gangmin/frame/frame{}.jpg'.format(i)
    with open('result4.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(vision_api(file_name))
