# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:1 7:29 2022

@author: Alfonso Blanco
"""
######################################################################
# PARAMETERS
######################################################################
#
dir=""
dirname= dir +"test6Training\\images"

dirname_labels = dir +"test6Training\\labels"

dirname_codfilters=dir + "test6Training\\codfilters"

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
#####################################################################
def ThresholdStable(image):
    # -*- coding: utf-8 -*-
    """
    Created on Fri Aug 12 21:04:48 2022
    Author: Alfonso Blanco García
    
    Looks for the threshold whose variations keep the image STABLE
    (there are only small variations with the image of the previous 
     threshold).
    Similar to the method followed in cv2.MSER
    https://datasmarts.net/es/como-usar-el-detector-de-puntos-clave-mser-en-opencv/https://felipemeganha.medium.com/detecting-handwriting-regions-with-opencv-and-python-ff0b1050aa4e
    """
    
    import cv2
    import numpy as np
   

    thresholds=[]
    Repes=[]
    Difes=[]
    
    gray=image 
    grayAnt=gray

    ContRepe=0
    threshold=0
    for i in range (255):
        
        ret, gray1=cv2.threshold(gray,i,255,  cv2.THRESH_BINARY)
        Dife1 = grayAnt - gray1
        Dife2=np.sum(Dife1)
        if Dife2 < 0: Dife2=Dife2*-1
        Difes.append(Dife2)
        if Dife2<22000: # Case only image of license plate
        #if Dife2<60000:    
            ContRepe=ContRepe+1
            
            threshold=i
            grayAnt=gray1
            continue
        if ContRepe > 0:
            
            thresholds.append(threshold) 
            Repes.append(ContRepe)  
        ContRepe=0
        grayAnt=gray1
    thresholdMax=0
    RepesMax=0    
    for i in range(len(thresholds)):
        #print ("Threshold = " + str(thresholds[i])+ " Repeticiones = " +str(Repes[i]))
        if Repes[i] > RepesMax:
            RepesMax=Repes[i]
            thresholdMax=thresholds[i]
            
    #print(min(Difes))
    #print ("Threshold Resultado= " + str(thresholdMax)+ " Repeticiones = " +str(RepesMax))
    return thresholdMax

 
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
    #hist=cv2.calcHist(gray,[0],None,[256],[0,256])
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
    TotFailures=0   
    
     
    gray=gray[Y_start:Y_end, X_start:X_end]
    
    
    
    X_resize=x_resize
    Y_resize=y_resize
    
      
    gray=cv2.resize(gray,None,fx=Resize_xfactor,fy=Resize_yfactor,interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.resize(gray, (X_resize,Y_resize), interpolation = cv2.INTER_AREA)
    
    rotation, spectrum, frquency =GetRotationImage(gray)
    rotation=90 - rotation
   
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
        return 1
    ret, o2 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(o2, lang='eng',  \
    config='--psm 13 --oem 3')
       
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with CLAHE and THRESH_BINARY_INV" )
        TotHits=TotHits+1
        return 2
    ret, o3 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO)
    text = pytesseract.image_to_string(o3, lang='eng',  \
    config='--psm 13 --oem 3')
       
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with CLAHE and THRESH_TOZERO" )
        TotHits=TotHits+1
        return 3
    ret, o4 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TOZERO_INV)
    text = pytesseract.image_to_string(o4, lang='eng',  \
    config='--psm 13 --oem 3')
       
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with CLAHE and THRESH_TOZERO_INV" )
        TotHits=TotHits+1
        return 4
    ret, o5 = cv2.threshold(gray_img_clahe, th, max_val, cv2.THRESH_TRUNC)
    text = pytesseract.image_to_string(o5, lang='eng',  \
    config='--psm 13 --oem 3')
       
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with CLAHE and THRESH_TRUNC" )
        TotHits=TotHits+1
        return 5
    ret ,o6=  cv2.threshold(gray_img_clahe, th, max_val,  cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(o6, lang='eng',  \
    config='--psm 13 --oem 3')
      
    text = ''.join(char for char in text if char.isalnum())
   
    if text==Licenses[i]:
       print(text + "  Hit with CLAHE and THRESH_OTSU" )
       TotHits=TotHits+1
       return 6
    #
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    #
    
    gray1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    cv2.THRESH_BINARY,11,2) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
       
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with adaptive Threshold Mean and THRESH_BINARY" )
        TotHits=TotHits+1
        return 7
    
     
   
      
    gray1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    cv2.THRESH_BINARY_INV,11,2) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
   
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with adaptive Threshold  Mean and THRESH_BINARY_INV" )
        TotHits=TotHits+1
        return 8  
   
    #perjudicial
    #gray= cv2.bilateralFilter(gray,3, 75, 75)
    gray1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY,11,2)  
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
   
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with adaptive Threshold Gaussian and THRESH_BINARY" )
        TotHits=TotHits+1
        return 9  
         
    
   
    gray1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY_INV,11,2)  
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
   
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with adaptive Threshold Gaussian and THRESH_BINARY_INV" )
        TotHits=TotHits+1
        return 10  
        
   
    #   Otsu's thresholding
    ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
   
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_BINARY" )
        TotHits=TotHits+1
        return 11   
         
         
    
    #   Otsu's thresholding
    ret2,gray1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
   
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_BINARY_INV" )
        TotHits=TotHits+1
        return 12
    
    
    #   Otsu's thresholding
    ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 13 --oem 3')
       
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
            print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TRUNC" )
            TotHits=TotHits+1
            return 13  
   
   
   
    #   Otsu's thresholding
    
    ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 13 --oem 3')
   
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
            print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TOZERO" )
            TotHits=TotHits+1
            return 14    
                        
    
    #   Otsu's thresholding
    
    ret2,gray1 = cv2.threshold(gray,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(gray1, lang='eng',  \
        config='--psm 13 --oem 3')
   
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
            print(text + "  Hit with Otsu's thresholding of cv2 and THRESH_TOZERO_INV" )
            TotHits=TotHits+1
            return 15        
    
    
    ####################################################
    # experimental formula based on the brightness
    # of the whole image 
    ####################################################
    
    SumBrightness=np.sum(imagesComplete[i])  
    threshold=(SumBrightness/177600.00) 
    
    #####################################################
   
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with  Brightness and THRESH_TRUNC" )
        TotHits=TotHits+1
        return 16
       
        
    gray1 = cv2.medianBlur(gray,3)  
    ret, gray1=cv2.threshold(gray1,threshold,255,  cv2.THRESH_TRUNC) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with Brightness and THRESH_TRUNC and medianblur" )
        TotHits=TotHits+1
        return 17
  
        
    gray1 = cv2.GaussianBlur(gray,(3,3), sigmaX=0, sigmaY=0)  
    ret, gray1=cv2.threshold(gray1,threshold,255,  cv2.THRESH_TRUNC) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with Brightness and THRESH_TRUNC and GaussianBlur" )
        TotHits=TotHits+1
        return 18
    
    gray1= cv2.bilateralFilter(gray,3, 75, 75)  
    ret, gray1=cv2.threshold(gray1,threshold,255,  cv2.THRESH_BINARY) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with Brightness and THRESH_BINARY" )
        TotHits=TotHits+1
        return 19 
       
  
       
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_BINARY_INV) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with Brightness and THRESH_BINARY_INV" )
        TotHits=TotHits+1
        return 20
  
    
    gray1= cv2.bilateralFilter(gray,3, 75, 75)
    ret, gray1=cv2.threshold(gray1,threshold,255,  cv2.THRESH_TOZERO) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with  Brightness and THRESH_TOZERO" )
        TotHits=TotHits+1
        return 21
       
               
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO_INV) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with  Brightness and THRESH_TOZERO_INV" )
        TotHits=TotHits+1
        return 22
   
    
    threshold=ThresholdStable(gray)
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TRUNC) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with Stable and THRESH_TRUNC" )
        TotHits=TotHits+1
        return 23
       
       
    gray1= cv2.bilateralFilter(gray,3, 75, 75) 
    ret, gray1=cv2.threshold(gray1,threshold,255,  cv2.THRESH_BINARY) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with Stable and THRESH_BINARY" )
        TotHits=TotHits+1
        return 24
    
    
       
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_BINARY_INV) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit con Stable y THRESH_BINARY_INV" )
        TotHits=TotHits+1
        return 25            
        
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Acierto con Stable y THRESH_TOZERO" )
        TotHits=TotHits+1
        return 26
   
      
    ret, gray1=cv2.threshold(gray,threshold,255,  cv2.THRESH_TOZERO_INV) 
    text = pytesseract.image_to_string(gray1, lang='eng',  \
    config='--psm 13 --oem 3')
    text = ''.join(char for char in text if char.isalnum())
    
    if text==Licenses[i]:
        print(text + "  Hit with Stable and THRESH_TOZERO_INV" )
        TotHits=TotHits+1
        return 27   
  
    #https://en.wikipedia.org/wiki/Kernel_(image_processing)
    #https://stackoverflow.com/questions/4993082/how-can-i-sharpen-an-image-in-opencv, respuesta 66
      
    for z in range(4,8):
    
       kernel = np.array([[0,-1,0], [-1,z,-1], [0,-1,0]])
       gray1 = cv2.filter2D(gray, -1, kernel)
              
       text = pytesseract.image_to_string(gray1, lang='eng',  \
       config='--psm 13 --oem 3 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 ')
      
       text = ''.join(char for char in text if char.isalnum())
       
       if text==Licenses[i]:
           print(text + "  Hit with Sharpen filter" )
           TotHits=TotHits+1
           return 28 

    
        #https://en.wikipedia.org/wiki/Kernel_(image_processing)
        #https://en.wikipedia.org/wiki/Kernel_(image_processing)
        
    gray2= cv2.bilateralFilter(gray,3, 75, 75)
    for z in range(5,11):
       kernel = np.array([[-1,-1,-1], [-1,z,-1], [-1,-1,-1]])
       gray1 = cv2.filter2D(gray2, -1, kernel)
       
       
       text = pytesseract.image_to_string(gray1, lang='eng',  \
       config='--psm 13 --oem 3 -c tessedit_char_whitelist= ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 ')
      
       text = ''.join(char for char in text if char.isalnum())
       
       if text==Licenses[i]:
           print(text + "  Hit with Sharpen filter modified" )
           TotHits=TotHits+1
           return 29 
           #break

   
     
   
    print(Licenses[i] + " NOT RECOGNIZED")
      
     
    return 0
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
     Thresholds=[]
     
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
                 
                
                 images.append(image)
                 Licenses.append(License)
                 
                 
                
                 Cont+=1
     
     return images, Licenses

###########################################################
# MAIN
##########################################################

labels=loadlabelsRoboflow(dirname_labels)

imagesComplete, Licenses=loadimagesRoboflow(dirname)



print("Number of imagenes : " + str(len(imagesComplete)))
print("Number of  labels : " + str(len(labels)))
print("Number of   licenses : " + str(len(Licenses)))

TotHits=0
TotFailures=0

NumberImageOrder=0

for i in range (len(imagesComplete)):
        
        NumberImageOrder=NumberImageOrder+1
        
        gray=imagesComplete[i]
        
        License=Licenses[i]
        
        lineaLabel =labels[i].split(" ")
        
        # Meaning of fields in files labels
        #https://github.com/ultralytics/yolov5/issues/2293
        #
        x_center=float(lineaLabel[1])
        y_center=float(lineaLabel[2])
        width=float(lineaLabel[3])
        heigh=float(lineaLabel[4])
      
        x_off=3
        y_off=2
        
        x_resize=220
        y_resize=70
        
        Resize_xfactor=1.78
        Resize_yfactor=1.78
        
        ContLoop=0
        
        SwFounded=0
        
        BilateralOption=0
        
        while (SwFounded==0 and ContLoop < 6):
            ContLoop+=1
            SwFounded= FindLicenseNumber (gray,x_center,y_center, width,heigh, x_off, y_off,  License, x_resize, y_resize, \
                               Resize_xfactor, Resize_yfactor, BilateralOption)
            if ContLoop==1 and SwFounded==0:
                x_off=4
                print("SECOND TRY")
                
            if ContLoop==2 and SwFounded==0:
                 x_off=2
                 print("THIRD TRY")
                 
            if ContLoop==3 and SwFounded==0:
                   x_off=3
                   y_off=3
                   print("FOURTH TRY")
            if ContLoop==4 and SwFounded==0:
                  x_off=3
                  y_off=2
                  Resize_xfactor=2.0
                  Resize_yfactor=2.0
                  print("FIFTH TRY")
            
            if ContLoop==5 and SwFounded==0:
                   x_off=3
                   y_off=2
                   Resize_xfactor=1.78
                   Resize_yfactor=1.78
                   BilateralOption=1
                   print("SIXTH ATTEMPT")     
           
            
        if SwFounded > 0:
            CodFilter=SwFounded*10+ContLoop
            lineaw=[]
            with open( dirname_codfilters+"\\" + License +".txt","w") as  w:
                
                 lineaw.append(str(CodFilter))
                 lineaWrite =','.join(lineaw)
                 lineaWrite=lineaWrite + "\n"
                 w.write(lineaWrite)
            TotHits+=1
        else:
            TotFailures +=1
      
print("")           
print("Total Hits = " + str(TotHits ))
print("Total Failures = " + str(TotFailures ))
      
                 
        