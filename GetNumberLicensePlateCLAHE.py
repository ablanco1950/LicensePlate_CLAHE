# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:1 7:29 2022

@author: Alfonso Blanco
"""
######################################################################
# PARAMETERS
######################################################################

dir=""


dirname= "test6Training\\images"

dirname_labels = dir +"test6Training\\labels"

TabClipLimits=[]
TabClipLimits.append(1.0)
TabClipLimits.append(3.0)
TabClipLimits.append(5.0)
TabClipLimits.append(-14.0)
TabClipLimits.append(6.0)
TabClipLimits.append(4.0)
TabClipLimits.append(15.0)
TabClipLimits.append(92.0)

######################################################################

import pytesseract

import numpy as np

import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

X_resize=220
Y_resize=70

import os
import re

import imutils
#####################################################################
"""
Copied from https://gist.github.com/endolith/334196bac1cac45a4893#

other source:
    https://stackoverflow.com/questions/46084476/radon-transformation-in-python
"""

from skimage.transform import radon

import numpy
from numpy import  mean, array, blackman, sqrt, square
from numpy.fft import rfft

try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, numpy.argmax(x))[0]
except ImportError:
    from numpy import argmax


def GetRotationImage(image):

   
    I=image
    I = I - mean(I)  # Demean; make the brightness extend above and below zero
    
    
    # Do the radon transform and display the result
    sinogram = radon(I)
   
    
    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
      
    # rms_flat does no exist in recent versions
    #r = array([mlab.rms_flat(line) for line in sinogram.transpose()])
    r = array([sqrt(mean(square(line))) for line in sinogram.transpose()])
    rotation = argmax(r)
    #print('Rotation: {:.2f} degrees'.format(90 - rotation))
    #plt.axhline(rotation, color='r')
    
    # Plot the busy row
    row = sinogram[:, rotation]
    N = len(row)
    
    # Take spectrum of busy row and find line spacing
    window = blackman(N)
    spectrum = rfft(row * window)
    
    frequency = argmax(abs(spectrum))
   
    return rotation, spectrum, frequency


 
# Copied from https://learnopencv.com/otsu-thresholding-with-opencv/ 
def OTSU_Threshold(image):
# Set total number of bins in the histogram

    bins_num = 256
    
    # Get the image histogram
    
    hist, bin_edges = np.histogram(image, bins=bins_num)
   
    # Get normalized histogram if it is required
    
    #if is_normalized:
    
    hist = np.divide(hist.ravel(), hist.max())
    
     
    
    # Calculate centers of bins
    
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    
    weight1 = np.cumsum(hist)
    
    weight2 = np.cumsum(hist[::-1])[::-1]
   
    # Get the class means mu0(t)
    
    mean1 = np.cumsum(hist * bin_mids) / weight1
    
    # Get the class means mu1(t)
    
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Maximize the inter_class_variance function val
    
    index_of_max_val = np.argmax(inter_class_variance)
    
    threshold = bin_mids[:-1][index_of_max_val]
    
    #print("Otsu's algorithm implementation thresholding result: ", threshold)
    return threshold

#########################################################################
def ApplyCLAHE(gray):
#https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45
    
    gray_img_eqhist=cv2.equalizeHist(gray)
    hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])
    clahe=cv2.createCLAHE(clipLimit=200,tileGridSize=(3,3))
    gray_img_clahe=clahe.apply(gray_img_eqhist)
    return gray_img_clahe
#########################################################################
def FindLicenseNumber (gray,x_center,y_center, width,heigh, x_offset, y_offset,  License, x_resize, y_resize, \
                       Resize_xfactor, Resize_yfactor, BilateralOption):
#########################################################################
# adapted from:
#  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
# by Alfonso Blanco García
########################################################################  
    
    if BilateralOption ==1:
       gray= cv2.bilateralFilter(gray,3, 75, 75)

    
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    x_start= x_center - width*0.5
    x_end=x_center + width*0.5
    
    y_start= y_center - heigh*0.5
    y_end=y_center + heigh*0.5
    
    X_start=int(x_start*416)
    X_end=int(x_end*416)
    
    Y_start=int(y_start*416)
    Y_end=int(y_end*416)
    
    
    
    # Clipping the boxes in two positions helps
    # in license plate reading
    X_start=X_start + x_offset   
    Y_start=Y_start + y_offset
    
    
    #print ("X_start " + str(X_start))
    #print ("X_end " + str(X_end))
    #print ("Y_start " + str(Y_start))
    #print ("Y_end " + str(Y_end))
    
    TotHits=0
         
    gray=gray[Y_start:Y_end, X_start:X_end]
    
       
    X_resize=x_resize
    Y_resize=y_resize
     
    
    gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
    
    rotation, spectrum, frquency =GetRotationImage(gray)
    rotation=90 - rotation
    #print("Car" + str(NumberImageOrder) + " Brillo : " +str(SumBrightnessLic) +   
    #      " Desviacion : " + str(DesvLic))
    if (rotation > 0 and rotation < 30)  or (rotation < 0 and rotation > -30):
      
        gray=imutils.rotate(gray,angle=rotation)
    
    gray_img_clahe=ApplyCLAHE(gray)
    
    th=OTSU_Threshold(gray_img_clahe)
    max_val=255
    ret, o1 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(o1, lang='eng',  \
    config='--psm 13 --oem 3')
       
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with CLAHE and THRESH_BINARY" )
        TotHits=TotHits+1
        return 1, o1
    ret, o2 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(o2, lang='eng',  \
    config='--psm 13 --oem 3')
       
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with CLAHE and THRESH_BINARY_INV" )
        TotHits=TotHits+1
        return 2, o2
    ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
    text = pytesseract.image_to_string(o3, lang='eng',  \
    config='--psm 13 --oem 3')
       
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with CLAHE and THRESH_TOZERO" )
        TotHits=TotHits+1
        return 3, o3
    ret, o4 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO_INV)
    text = pytesseract.image_to_string(o4, lang='eng',  \
    config='--psm 13 --oem 3')
       
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with CLAHE and THRESH_TOZERO_INV" )
        TotHits=TotHits+1
        return 4, o4
    ret, o5 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TRUNC)
    text = pytesseract.image_to_string(o5, lang='eng',  \
    config='--psm 13 --oem 3')
       
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with CLAHE and THRESH_TRUNC" )
        TotHits=TotHits+1
        return 5, o5
    ret ,o6=  cv2.threshold(gray_img_clahe, th, max_val,  cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(o6, lang='eng',  \
    config='--psm 13 --oem 3')
      
    text = ''.join(char for char in text if char.isalnum())
   
    if text==Licenses[i]:
       print(text + "  Hit with CLAHE and THRESH_OTSU" )
       TotHits=TotHits+1
       return 6, o6
    
   
    #   Otsu's thresholding
    ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
   
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_BINARY" )
        TotHits=TotHits+1
        return 7 , gray1  
         
         
    
    #   Otsu's thresholding
    ret2,gray1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
   
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_BINARY_INV" )
        TotHits=TotHits+1
        return 8, gray1
    
    #   Otsu's thresholding
    ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 13 --oem 3')
       
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
            print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TRUNC" )
            TotHits=TotHits+1
            return 9, gray1  
   
   
    #   Otsu's thresholding
    
    ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 13 --oem 3')
   
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
            print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TOZERO" )
            TotHits=TotHits+1
            return 10, gray1    
                        
    
    #   Otsu's thresholding
    
    ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 13 --oem 3')
   
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
            print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TOZERO_INV" )
            TotHits=TotHits+1
            return 11, gray1        
    
   
    #############################################3####
    #https://www.mo4tech.com/opencv-contrast-constrained-adaptive-histogram-equalization.html
    ##################################################
    for j in range(len(TabClipLimits)):
        clipLimite=TabClipLimits[j]
        clahe = cv2.createCLAHE(clipLimit=clipLimite)
       
        # Use different clipLimit values
        clahe.setClipLimit(clipLimite)
        gray = clahe.apply(gray)
        
        text = pytesseract.image_to_string(gray, lang='eng',  \
        config='--psm 13 --oem 3')
           
        text = ''.join(char for char in text if char.isalnum())
        
        if text==Licenses[i]:
            print(text + "  Hit with CLAHE ClipLimit = " +str(clipLimite) )
            TotHits=TotHits+1
            return 12, gray
   
      
    print(Licenses[i] + " NOT RECOGNIZED") 
    return 0, gray
def loadlabelsRoboflow (dirname ):
 #########################################################################
 
 ########################################################################  
     lblpath = dirname + "\\"
     
     labels = []
    
     Conta=0
     print("Reading labels from ",lblpath)
     
     
     
     for root, dirnames, filenames in os.walk(lblpath):
         
                
         for filename in filenames:
             
             if re.search("\.(txt)$", filename):
                 Conta=Conta+1
                 # case test
                 
                 filepath = os.path.join(root, filename)
               
                 f=open(filepath,"r")

                 ContaLin=0
                 for linea in f:
                     
                     lineadelTrain =linea.split(" ")
                     if lineadelTrain[0] == "0":
                         ContaLin=ContaLin+1
                         labels.append(linea)
                         break
                 f.close() 
                 if ContaLin==0:
                     print("Rare labels without tag 0 on " + filename )
                   
                 
     
     return labels
 ########################################################################
def loadimagesRoboflow (dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     Licenses=[]
     
     
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
         
         
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                 License=filename[:len(filename)-4]
                
                 image = cv2.imread(filepath)
                 
                 #Color Balance
                #https://blog.katastros.com/a?ID=01800-4bf623a1-3917-4d54-9b6a-775331ebaf05
                
                 img = image
                    
                 r, g, b = cv2.split(img)
                
                 r_avg = cv2.mean(r)[0]
                
                 g_avg = cv2.mean(g)[0]
                
                 b_avg = cv2.mean(b)[0]
                
                 
                 # Find the gain occupied by each channel
                
                 k = (r_avg + g_avg + b_avg)/3
                
                 kr = k/r_avg
                
                 kg = k/g_avg
                
                 kb = k/b_avg
                
                 
                 r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
                
                 g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
                
                 b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
                
                 
                 balance_img = cv2.merge([b, g, r])
                 
                 image=balance_img
                 
                   
                 images.append(image)
                 Licenses.append(License)
                 
                 
                
                 Cont+=1
     
     return images, Licenses


# COPIED FROM https://programmerclick.com/article/89421544914/
def gamma_trans (img, gamma): # procesamiento de la función gamma
         gamma_table = [np.power (x / 255.0, gamma) * 255.0 for x in range (256)] # Crear una tabla de mapeo
         gamma_table = np.round (np.array (gamma_table)). astype (np.uint8) #El valor del color es un número entero
         return cv2.LUT (img, gamma_table) #Tabla de búsqueda de color de imagen. Además, se puede diseñar un algoritmo adaptativo de acuerdo con el principio de homogeneización de la intensidad de la luz (color).
def nothing(x):
    pass
###########################################################
# MAIN
##########################################################

labels=loadlabelsRoboflow(dirname_labels)

imagesComplete, Licenses=loadimagesRoboflow(dirname)



print("Number of imagenes : " + str(len(imagesComplete)))
print("Number of  labels : " + str(len(labels)))
print("Number of   licenses : " + str(len(Licenses)))

TotHits=0

for i in range (len(imagesComplete)):
      
        gray=imagesComplete[i]
        
        License=Licenses[i]
        
        #if License < "EATTHE":
        #    print("SALTA " + License)
        #    continue
        
        lineaLabel =labels[i].split(" ")
        
        # Meaning of fields in files labels
        #https://github.com/ultralytics/yolov5/issues/2293
        #
        x_center=float(lineaLabel[1])
        y_center=float(lineaLabel[2])
        width=float(lineaLabel[3])
        heigh=float(lineaLabel[4])
        
        Cont=1
        
        x_off=3
        y_off=2
        
        x_resize=220
        y_resize=70
        
        Resize_xfactor=1.78
        Resize_yfactor=1.78
        
        ContLoop=0
        
        SwFounded=0
        
        BilateralOption=0
        
        SwFounded, gray_new= FindLicenseNumber (gray,x_center,y_center, width,heigh, x_off, y_off,  License, x_resize, y_resize, \
                               Resize_xfactor, Resize_yfactor, BilateralOption)
          
        if SwFounded > 0:
               TotHits+=1
  
              
print("")           
print("Total Hits = " + str(TotHits ) + " from " + str(len(imagesComplete)) + " images readed")


      
                 
        