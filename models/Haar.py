import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

def detect(gray, eye_cascade, scale):
    eyes = eye_cascade.detectMultiScale(
    gray,
    scaleFactor=scale,
    minNeighbors=4,
    minSize=(15, 15) )
    return eyes

def find_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    scales = [1.05, 1.075, 1.1, 1.15, 1.2]
    i = 0
    eyes = []
    while len(eyes) != 2 and i<5:
        eyes = detect(gray, eye_cascade, scales[i])
        i += 1

    
    if len(eyes) != 2:
        print("Cound not find.")
    else:
        eye1 = eyes[0]
        eye2 = eyes[1]
        x = min(eye1[0], eye2[0]) -5
        y = min(eye1[1], eye2[1]) 
        w = max(eye1[0]+eye1[2], eye2[0]+eye2[3]) +5
        h = max(eye1[1]+eye1[3], eye2[1]+eye2[3])
        mask = np.zeros_like(image)
        mask[y:h, x:w] = image[y:h, x:w]

    return mask