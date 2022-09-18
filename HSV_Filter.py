import cv2
import numpy as np
import sys

image = cv2.imread(cv2.samples.findFile("Lena.jpg"))
# result = image.copy()

if image is None:
    sys.exit("Could not read img")

RGB_lower = [0, 0, 0]
RGB_upper = [255, 255, 255]


def change_arr(arr, color, val):
    arr[color] = val

    lower_arr = np.array(RGB_lower)
    upper_arr = np.array(RGB_upper)

    full_mask = cv2.inRange(image, lower_arr, upper_arr)
    final_result = cv2.bitwise_and(image, image, mask=full_mask)

    cv2.imshow(win_name, final_result)

win_name: str = "win"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 400, 700)


cv2.createTrackbar("red lower", win_name, 0, 255, lambda val: change_arr(RGB_lower, 0, val))
cv2.createTrackbar("red upper", win_name, 0, 255, lambda val: change_arr(RGB_upper, 0, val))
cv2.setTrackbarPos("red upper", win_name, 255)
cv2.createTrackbar("gre lower", win_name, 0, 255, lambda val: change_arr(RGB_lower, 1, val))
cv2.createTrackbar("gre upper", win_name, 0, 255, lambda val: change_arr(RGB_upper, 1, val))
cv2.setTrackbarPos("gre upper", win_name, 255)
cv2.createTrackbar("blu lower", win_name, 0, 255, lambda val: change_arr(RGB_lower, 2, val))
cv2.createTrackbar("blu upper", win_name, 0, 255, lambda val: change_arr(RGB_upper, 2, val))
cv2.setTrackbarPos("blu upper", win_name, 255)

k = cv2.waitKey(0)

while (True):
    k = cv2.waitKey()
    if k == ord("q"):
        break
