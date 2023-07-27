#https://medium.com/@nayak.abhijeet1/analogue-gauge-reader-using-computer-vision-62fbd6ec84cc 
#https://www.softwebsolutions.com/resources/number-plate-recognition-using-computer-vision.html 
#https://www.reddit.com/r/computervision/comments/5okxww/how_can_i_recognize_the_digits_in_this_picture/
#https://en.kompf.de/cplus/emeocv.html
#https://pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
#https://stackoverflow.com/questions/37745519/use-pytesseract-ocr-to-recognize-text-from-an-image 
#https://github.com/arnavdutta/Meter-Reading/blob/master/meter_reading_processing.py (seems to be a usable reference)

#https://github.com/skaringa/emeocv/blob/master/ImageProcessor.cpp#L208

#Simplified Python implementation of code listed at 
#   https://en.kompf.de/cplus/emeocv.html 

import sys
import os
import cv2 
import numpy as np
import scipy 

def rotate(source_gray, rotationDegrees, cols, rows):
    rotation = cv2.getRotationMatrix2D((cols/2, rows/2), rotationDegrees, 1)
    source_rotated = cv2.warpAffine(source_gray, rotation, (cols, rows)) 
    return source_rotated
    
def detect_skew(source_gray, cannyThreshold1, cannyThreshold2):
    skew = 0
    edges = cv2.Canny(source_gray, cannyThreshold1, cannyThreshold2) 
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 140)
    
    return skew, edges 
    
def find_counter_digits(edges):
    #cv::findContours(edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); 
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print(contours)
    
def process_image(image_filename, cannyThreshold1, cannyThreshold2):
    rotationDegrees = 0
    source = cv2.imread(image_filename)
    cols, rows = source.shape[:2]
    print(("%d, %d")%(cols, rows))
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    source_gray = rotate(source_gray, rotationDegrees, cols, rows)
    skew, edges = detect_skew(source_gray, cannyThreshold1, cannyThreshold2)
    print(edges)
    #rotate(skew, skew, cols, rows)
    find_counter_digits(edges)


if __name__ == "__main__":
    print("Meter Reader Code. Implemented from https://en.kompf.de/cplus/emeocv.html")
    if len(sys.argv) < 2:
        print("Error: Missing arguments.")
        print("Usage: python meter_reader.py <input image>")
        sys.exit(0)
    image_filename = sys.argv[1]
    cannyThreshold1 = 100
    cannyThreshold2 = 200
    process_image(image_filename, cannyThreshold1, cannyThreshold2)
    