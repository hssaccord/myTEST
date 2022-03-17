
# https://medium.com/analytics-vidhya/computer-vision-and-deep-learning-part-2-586b6a0d3220  --- main
# https://github.com/Esri/raster-deep-learning/blob/master/docs/writing_deep_learning_python_raster_functions.md
import cv2
import numpy as np


from matplotlib import pyplot as plt
cv_image= cv2.imread("/home/jameshung/Pictures/forest01.jpg",0)
one =cv2.adaptiveThreshold(cv_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
ret2, two = cv2.threshold(cv_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(cv_image,(5,5),0)
ret3,three = cv2.threshold(blur, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
images = [cv_image, 0, one,
          cv_image, 0, two,
          blur, 0, three]
titles = ['Original Image','Histogram','Adaptive Mean Thresholding',
          'Original Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding of Gaussian Blur Image"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
cv2.waitKey(1000)
