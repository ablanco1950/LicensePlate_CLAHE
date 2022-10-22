# LicensePlate_CLAHE

Through the use of Contrast Limited Adaptive Histogram Equalization (CLAHE) filters, completed with otsu filters, a direct reading of car license plates with
success rates above 70% and an acceptable time is achieved

Requirements:

pytesseract

numpy

cv2

re

imutils

skimage  (is scikit-image)

In the download directory you should find the downloaded test6Training.zip and must unzip folder: test6Training with all its subfolders, containing the images and labels needed for the test. This directory must be in the same directory where is te program GetNumberLicensePlateCLAHE.py ( unziping may create two directorries with te name test6Training an the images may not be founded when executing it)

from the download directory, run:

GetNumberLicensePlateCLAHE.py

=======================================================================

Also is included module CLAHETrainingLisencePlateSeveralFiltersCropsAndResizes.py

With its execution 100% success is achieved, since as the name of the  itself module indicates, 50 classes of filters and treatments are considered, some self-made and others standardized. Along with various treatments modifying the dimensions of the crops and resizes. which implies a long execution time.

This module generates in test6Training/codfilters a file for each car license plate with the code = code_filter* 10 + code_crop_resize. 
With these files, a Y_train file could be made for treatment by means of CNN or SVM, but no results have been obtained, probably due to
the reduced number of images, the excessive number of classes (500) compared to the number of records (117) or errors in the design of the CNN network.

References:


https://www.mo4tech.com/opencv-contrast-constrained-adaptive-histogram-equalization.html


#https://towardsdatascience.com/image-enhancement-techniques-using-opencv-and-python-9191d5c30d45


https://riull.ull.es/xmlui/bitstream/handle/915/10237/Umbralizacion%20adaptativa%20de%20imagenes%20based%20en%20histograms%20space%20-%20color.pdf?sequence=1


https://www.roboflow.com, where most of the contributed images come from,  that have been labeled with yolo or the  tool of labeling provided by roboflow

https://learnopencv.com/otsu-thresholding-with-opencv/ 

https://gist.github.com/endolith/334196bac1cac45a4893#

https://stackoverflow.com/questions/46084476/radon-transformation-in-python

