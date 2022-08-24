import os
import glob
import json
import cv2
import numpy as np
import natsort
import sys

# Function to mao 2D lidar to attack zones
def output_attack_mask(img_path, k):
#    print("RUNNING ATTACK MASK ON " + img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img / 255; #1 if mask, 0 otherwises
    for i in range(0, np.shape(img)[0]):
        for j in range(0, np.shape(img)[1]):
            if img[i][j] == 1:
                continue
            for m in range(1, k):
                toTry = [[i + m, j], [i - m, j], [i, j-m], [i, j + m]]
                for walk in toTry:
                    if(walk[0] >= 0 and walk[1] >= 0 and walk[0] < np.shape(img)[0] and walk[1] < np.shape(img)[1] and img[walk[0]][walk[1]] == 1):
                        img[i][j] = 1
                        break
                if img[i][j] == 1:
                    break
            if img[i][j] == 1:
                continue
    return img
            

