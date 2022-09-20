import cv2
import filters
import numpy as np
import math

def objectDetection(img, color):
    filters.RGB(img, color, showExtra=True)