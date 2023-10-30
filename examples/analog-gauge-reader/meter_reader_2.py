import sys
import os
import cv2 
import numpy as np
import scipy 
import imutils
from imutils.perspective import four_point_transform

def process_image(image_filename, cannyThreshold1, cannyThreshold2):
    #return bounding boxes and filteredcontours 
    minHeight = 10
    maxHeight = 50
    minWidth = 10
    maxWidth = 50 
    
    source = cv2.imread(image_filename)
    cols, rows = source.shape[:2]
    print(("Shape: %d cols & %d rows.")%(cols, rows))
    source = imutils.resize(source, height=500)

    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    #source_gray = cv2.GaussianBlur(source_gray, (5, 5), 0)
    edges = cv2.Canny(source_gray, 50, cannyThreshold1, cannyThreshold2)
    print("Detected edges, detecting contours")
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(contours)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)    
     
    displayCnt = None
    prev_cnt = None
    source_copy = source_gray.copy() 
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(source_copy, (x,y), (x+w,y+h), (255,255,255), 2) 
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    cv2.imshow("image", source_copy)
    cv2.waitKey()
    
    
def func(source_gray, displayCnt, inv_option):
    minHeight = 10
    maxHeight = 50
    minWidth = 10
    maxWidth = 50 
    warped = four_point_transform(source_gray, displayCnt.reshape(4,2))
    threshed = cv2.threshold(warped, 0, 255, inv_option | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    threshed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
    cv2.imshow("thresholded image", threshed)
    cv2.waitKey() 
    cnts, hierarchy = cv2.findContours(cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((2,2))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if inv_option == cv2.THRESH_BINARY_INV:
        linecolor = (0, 255, 0)
    else:
        linecolor = (255, 0, 0)
        
    digitCnts = []
    cnts = sorted(cnts, key=cv2.contourArea, reverse=False)
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(threshed, (x,y),(x+w,y+h),linecolor,2)
        # if the contour is sufficiently large, it must be a digit
        if w >= minWidth and (h >= minHeight and h <= maxHeight):
            digitCnts.append(c) 
    
    cv2.imshow("image", threshed)
    cv2.waitKey() 

if __name__ == "__main__":
    print("Meter Reader Code. Adapted from https://en.kompf.de/cplus/emeocv.html & https://pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/")
    if len(sys.argv) < 2:
        print("Error: Missing arguments.")
        print("Usage: python meter_reader.py <input image>")
        sys.exit(0)
    image_filename = sys.argv[1]
    cannyThreshold1 = 200
    cannyThreshold2 = 255
    process_image(image_filename, cannyThreshold1, cannyThreshold2)
    