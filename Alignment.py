# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:25:15 2020

@author: kakeru
"""
import os 
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

start=time.time()
    

def fileImport(path):
    
    image_all=glob.glob(path)
    image_set=int(len(image_all)/5)
    
    return image_all,image_set


def filenameNum(num):
    
    if 0<=num<=9:
        filenum='000'+str(num)
    elif 10<=num<=99:
        filenum='00'+str(num)
    elif 100<=num<=999:
        filenum='0'+str(num)
    elif 1000<=num<=9999:
        filenum=str(num)
    
    return filenum

def imageRead(path,filename):
    
    imgB=cv2.imread(path+'IMG_'+str(filename)+'_1.tif',0)
    imgG=cv2.imread(path+'IMG_'+str(filename)+'_2.tif',0)
    imgR=cv2.imread(path+'IMG_'+str(filename)+'_3.tif',0)
    imgNIR=cv2.imread(path+'IMG_'+str(filename)+'_4.tif',0)
    imgRE=cv2.imread(path+'IMG_'+str(filename)+'_5.tif',0)
    
    return imgB,imgG,imgR,imgNIR,imgRE
    



def alignImages(img1,img2,max_pts,good_match_rate,min_match):
    
    orb=cv2.ORB_create(max_pts)

    kp1,des1=orb.detectAndCompute(img1,None)
    kp2,des2=orb.detectAndCompute(img2,None)

    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    
    matches=bf.match(des1,des2)
    matches=sorted(matches,key=lambda x:x.distance)
    good=matches[:int(len(matches)* good_match_rate)]
    
    if len(good)>min_match:
        src_pts=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        
        h,mask=cv2.findHomography(dst_pts,src_pts,cv2.RANSAC)
        
        img=cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],outImg=None,flags=2)
        
        height,width=img1.shape
        dst_img=cv2.warpPerspective(img2,h,(width,height))
        #cv2.imshow('gray',dst_img)
        return dst_img,h
        
    else:
        #cv2.imshow('gray',img)
        return img,np.zeros((3,3))

def rgbAlignment(imgB,imgG,imgR):
    
    rgb[:,:,1]=imgG[:,:]
    img_aligned,h=alignImages(imgG,imgB,500,0.5,10)
    rgb[:,:,0]=img_aligned[:,:]
    img_aligned,h=alignImages(imgG,imgR,500,0.5,10)
    rgb[:,:,2]=img_aligned[:,:]
    
    return rgb



def bufferCut(buff_height,buff_width,rgb):
    
    x0=0+buff_height
    y0=0+buff_width
    x1=height-buff_height
    y1=width-buff_width
    
    rgb=rgb[x0:x1,y0:y1]
    
    return rgb
    

if __name__=='__main__':
    
    path='C:\\Users\\zkkr6\\temp\\sugarbeet\\11_row\\Row\\'
    image_all,image_set=fileImport(path+'*.tif')
    
    for f in range(1,image_set):
        filename=filenameNum(f)
        imgB,imgG,imgR,imgNIR,imgRE=imageRead(path,filename)
        height,width=imgB.shape
        rgb=np.zeros((height,width,3),dtype=np.uint8)
        rgb=rgbAlignment(imgB,imgG,imgR)
        rgb=bufferCut(40,10,rgb)
        
        cv2.imwrite('C:\\Users\\zkkr6\\temp\\sugarbeet\\11_row\\RGB_jpg\\IMG_'+str(filename)+'_rgb.jpg',rgb)
        print('image '+str(filename)+' finished')

elapsed_time=time.time()-start
print('elapsed_time;{0}'.format(elapsed_time)+'[sec]')    

cv2.waitKey(0)
cv2.destroyAllWindows()






