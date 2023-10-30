#https://medium.com/@nayak.abhijeet1/analogue-gauge-reader-using-computer-vision-62fbd6ec84cc 
#https://www.softwebsolutions.com/resources/number-plate-recognition-using-computer-vision.html 
#https://www.reddit.com/r/computervision/comments/5okxww/how_can_i_recognize_the_digits_in_this_picture/
#https://en.kompf.de/cplus/emeocv.html
#https://pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
#https://stackoverflow.com/questions/37745519/use-pytesseract-ocr-to-recognize-text-from-an-image 
#https://github.com/arnavdutta/Meter-Reading/blob/master/meter_reading_processing.py (seems to be a usable reference)
#https://github.com/skaringa/emeocv/blob/master/ImageProcessor.cpp#L208
#Started off as Simplified Python implementation of code listed at 
#   https://en.kompf.de/cplus/emeocv.html 

import sys
import os
import cv2 
import numpy as np
import scipy 
import imutils
from imutils.perspective import four_point_transform
import pytesseract 

def rotate(source_gray, rotationDegrees, cols, rows):
    rotation = cv2.getRotationMatrix2D((cols/2, rows/2), rotationDegrees, 1)
    source_rotated = cv2.warpAffine(source_gray, rotation, (cols, rows)) 
    return source_rotated
    
def detect_skew(source_gray, cannyThreshold1, cannyThreshold2):
    skew = 0
    edges = cv2.Canny(source_gray, cannyThreshold1, cannyThreshold2) 
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 140)
    theta_min = 60 * np.pi/180.0
    theta_max = 120 * np.pi/180.0
    theta_avr = 0.0
    theta_deg = 0.0
    
    filtered_lines = []
    for i in range(0, len(lines)):
        theta = (lines[i])[0][1]
        print(theta)
        if (theta > theta_min and theta < theta_max) :
            theta_avr = theta_avr + theta;
            filtered_lines.append(lines[i])
            
    if len(filtered_lines) > 0:
        theta_avr = theta_avr / len(filtered_lines)
        theta_deg = (theta_avr / np.pi * 180.0) - 90
        
    skew = theta_deg 
    return (skew, edges) 
    
def find_counter_digits(edges):
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print(contours)
    boundingBoxes, filteredContours = filterContours(contours)
    alignedBoundingBoxes =  findAlignedBoundingBoxes(boundingBoxes)
    sort(alignedBoundingBoxes)
    #after sort, extract the bounding boxes and set to same size. 
    
def findAlignedBoundingBoxes(boundingBoxes):
    pass 

def sort(boundingBoxes):
    pass
    
def filterContours(contours):
    #return boudning boxes and filteredcontours 
    minHeight = 10
    maxHeight = 400
    minWidth = 10
    maxWidth = 400 
    boundingBoxes = []
    filteredContours = [] 
    
    for i in range(0, len(contours)):
        #print(contours[i])
        bounds = cv2.boundingRect(contours[i]) 
        #returns x,y,width,height 
        if (bounds[3] >= minHeight) and (bounds[3] <= maxHeight) :#and bounds[2] > 5 and bounds[2] <= bounds[3]:
            boundingBoxes.append(bounds)
            filteredContours.append(contours[i])
        
    print(boundingBoxes)
    return boundingBoxes, filteredContours
    
    
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
    fourPointContours = [] 
    prev_cnt = None
    source_copy = source_gray.copy() 
    
    source_for_contours = source_gray.copy() 
    cv2.drawContours(source_for_contours, cnts, -1, (255, 0, 0), 2)
    cv2.imshow("image", source_for_contours)
    cv2.waitKey()
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(source_copy, (x,y), (x+w,y+h), (255,255,255), 2) 
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if the contour has four vertices, then we have found
	# the thermostat display
        if len(approx) == 4:
            fourPointContours.append(approx)
            
            
    fourPointContours = sorted(fourPointContours, key=cv2.contourArea, reverse=True)    
    
    cv2.imshow("image", source_copy)
    cv2.waitKey()
    ############################################################################
    # threshold the warped image, then apply a series of morphological
    # operations to cleanup the thresholded image
    try:
        digitsCnts = func(source_gray.copy(), displayCnt, cv2.THRESH_BINARY_INV, "threshed1.png")
        print(("digit contours: %d") % (len(digitsCnts)))
        if len(digitsCnts) < 3:
            print("Unable to find digit contours with cv2.THRESH_BINARY_INV")
            print("Trying with cv2.THRESH_BINARY")
            digitsCnts = func(source_gray.copy(), displayCnt, cv2.THRESH_BINARY, "threshed2.png")
    except AttributeError as ae:
        func_odometer(source_gray)
    
    
    
    
def func_odometer(source_gray):
    #odometer like display
    #
    pass 
def func(source_gray, displayCnt, inv_option, filename):
    minHeight = 10
    maxHeight = 50
    minWidth = 10
    maxWidth = 50 
    warped = four_point_transform(source_gray, displayCnt.reshape(4,2))
    threshed = cv2.threshold(warped, 0, 255, inv_option | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    threshed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
    cnts, hierarchy = cv2.findContours(cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((2,2))), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digitCnts = []
    print(len(cnts))
    cnts = sorted(cnts, key=cv2.contourArea, reverse=False)
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(threshed,(x,y),(x+w,y+h),(255,0,0),2)

        # if the contour is sufficiently large, it must be a digit
        if w >= minWidth and (h >= minHeight and h <= maxHeight):
            digitCnts.append(c)
    
    cv2.imshow("image", threshed)
    cv2.waitKey() 
    cv2.imwrite(filename, threshed)
    return digitCnts

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
    