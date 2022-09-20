# import the opencv library
import cv2
import numpy as np
# import crappyV2_CV

# define a video capture object
vid = cv2.VideoCapture(0)

lower_HSV = [0,0,0]
upper_HSV = [255,255,255]

name = "frame"
cv2.namedWindow(name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(name, 400, 700)

def hue_lower(H):
    lower_HSV[0] = H


def sat_lower(S):
    lower_HSV[1] = S


def val_lower(V):
    lower_HSV[2] = V


def hue_upper(H):
    upper_HSV[0] = H


def sat_upper(S):
    upper_HSV[1] = S


def val_upper(V):
    upper_HSV[2] = V

cv2.createTrackbar("HL", name, 0, 255, hue_lower)
cv2.createTrackbar("SL", name, 0, 255, sat_lower)
cv2.createTrackbar("VL", name, 0, 255, val_lower)
cv2.setTrackbarPos("VL", name, 234)
cv2.createTrackbar("HU", name, 0, 255, hue_upper)
cv2.setTrackbarPos("HU", name, 132)
cv2.createTrackbar("SU", name, 0, 255, sat_upper)
cv2.setTrackbarPos("SU", name, 18)
cv2.createTrackbar("VU", name, 0, 255, val_upper)
cv2.setTrackbarPos("VU", name, 242)

def HSV_Filter(image):
    lower_arr = np.array(lower_HSV)
    upper_arr = np.array(upper_HSV)

    full_mask = cv2.inRange(image, lower_arr, upper_arr)
    final_result = cv2.bitwise_and(image, image, mask=full_mask)
    return final_result

while (True):
    ret, frame = vid.read()

    # HSV_Filter.change_arr()

    ret = HSV_Filter(frame)

    cv2.imshow(name,ret)
    # crappyV2_CV.objectDetection(frame, 'red')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
