import numpy as np
import cv2
from matplotlib import pyplot as plt
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976
from skimage import io, color

def color_loss(image1, image2):
    
    res=0;
    img =image1
    width =image1.shape[0]
    
   
    lab1 = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

    L1,A1,B1=cv2.split(lab1)
    img2 = image2
    lab2 = cv2.cvtColor(img2,cv2.COLOR_BGR2LAB)
    L2,A2,B2=cv2.split(lab2)
    for i in range(width):
        for k in range(width):
            color1 = LabColor(lab_l=L1[i][k], lab_a=A1[i][k], lab_b=B1[i][k])
# Color to be compared to the reference.
            color2 = LabColor(lab_l=L2[i][k], lab_a=A2[i][k], lab_b=B2[i][k])
# This is your delta E value as a float.
            delta_e = delta_e_cie1976(color1, color2)
            res+= delta_e
            #print(delta_e)
    return res
    



















