import os
from definitions import ROOT_DIR, SMOL_BLU, SMOL_RED, BIG_BLU, BIG_RED
import cv2
import numpy as np
from matplotlib import pyplot as plt

lower_blu = [239, 219, 0]
upper_blu = [255, 255, 194]
lower_red = [0, 0, 242]
# upper_red = [111, 145, 255]
upper_red = [111, 255, 255]#tune
lower_black = [33, 0, 0]
upper_black = [121, 61, 67]


def hue_lower(H):
    lower_black[0] = H
    lower_red[0] = H


def sat_lower(S):
    lower_black[1] = S
    lower_red[1] = S


def val_lower(V):
    lower_black[2] = V
    lower_red[2] = V


def hue_upper(H):
    upper_black[0] = H
    upper_red[0] = H


def sat_upper(S):
    upper_black[1] = S
    upper_red[1] = S


def val_upper(V):
    upper_black[2] = V
    upper_red[2] = V


def make_slid(name):
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


def RGB(image, color, showExtra=False):  # TODO have a setting to switch between red, blu, and black
    result = image.copy()

    # ==just blu==
    if color == 'blue':
        lower_arr = np.array(lower_blu)
        upper_arr = np.array(upper_blu)

        full_mask = cv2.inRange(image, lower_arr, upper_arr)

        result = cv2.bitwise_and(result, result, mask=full_mask)

    # ==just red==
    if color == 'red':
        lower_arr = np.array(lower_red)
        upper_arr = np.array(upper_red)

        full_mask = cv2.inRange(image, lower_arr, upper_arr)

        result = cv2.bitwise_and(result, result, mask=full_mask)

    # ==just black==
    if color == 'black':
        lower_arr = np.array(lower_black)
        upper_arr = np.array(upper_black)

        full_mask = cv2.inRange(image, lower_arr, upper_arr)

        result = cv2.bitwise_and(result, result, mask=full_mask)

    if showExtra:
        # cv2.namedWindow("RGB")
        # cv2.moveWindow("RGB", 0, 10)
        # cv2.displayOverlay("HSV", "HSV")
        cv2.imshow("RGB", result)
        # cv2.imshow("HSV", result_black)

    return result

"""def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: int = 0):
    # Adjusts contrast and brightness of an uint8 image.
    # contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    # brightness: [-255, 255] with 0 leaving the brightness as is
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)


contrast = 14
brightness = -1395

output = adjust_contrast_brightness(image, contrast, brightness)


B, G, R = cv2.split(output)"""

def blob_detection(image, showExtra=False):
    # https://learnopencv.com/blob-detection-using-opencv-python-c/
    # good guide on blob detection
    # must invert for blob detection
    inv_img = cv2.bitwise_not(image)
    inv_img = cv2.cvtColor(inv_img, cv2.COLOR_BGR2GRAY)  # converts BGR to black and white

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 100

    params.filterByCircularity = True
    params.maxCircularity = .8
    params.minCircularity = 0.1

    params.filterByConvexity = True
    params.minConvexity = 0.1

    params.filterByInertia = True
    # params.maxInertiaRatio = 0.1
    params.maxInertiaRatio = 1
    params.minInertiaRatio = 0.01
    # adjusts how "circular" something is
    # value of 1 means it's a perfect circle

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(inv_img)

    blank = np.zeros((1, 1))
    display = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if showExtra:
        # cv2.namedWindow("blob detection")
        # cv2.moveWindow("blob detection", 645, 10)
        # cv2.displayOverlay("blob detection", "blob detection")
        cv2.imshow("blob detection", display)

    return keypoints


def ROI(image, keypoints, showExtra=False):
    final_mask = np.zeros(image.shape[:2], dtype="uint8")
    for i in keypoints:
        mask = np.zeros(image.shape[:2], dtype="uint8")
        radius = i.size  # *1.5
        cv2.rectangle(mask, (int(i.pt[0] - radius), int(i.pt[1] - radius)),
                      (int(i.pt[0] + radius), int(i.pt[1] + radius)),
                      255, -1)
        final_mask += mask  # combining all the smaller masks

    img_to_op = cv2.bitwise_and(image, image, mask=final_mask)

    if showExtra:
        cv2.namedWindow("ROI")
        cv2.moveWindow("ROI", 1290, 10)
        # cv2.displayOverlay("ROI", "ROI")
        cv2.imshow("ROI", img_to_op)

    return img_to_op


def perspective_transform(image, img_cord, panelType: str, showExtra=False):
    pts1 = np.float32(img_cord)
    # transformed points
    cords = {'blue': (439, 400),
             'red': (451, 400)}
    """'smolBlu': (439, 400),
    'smolRed': (451, 400),
    'bigBlu': (725, 400),
    'bigRed': (725, 400)}"""
    if panelType is None:
        raise Exception("third argument for panel type is empty")
    w, h = cords[panelType]
    pts2 = np.float32([[0, 0], [w, 0],
                       [0, h], [w, h]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, cords[panelType])

    if showExtra:
        # Wrap the transformed image
        cv2.imshow('frame1' + str(img_cord[0][0] + img_cord[3][1]), result)  # Transformed Capture
    return result


def sobel_ED(image, showExtra=False):
    # Convert to graycsale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (11, 11), 0)
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    if showExtra:
        cv2.imshow('Sobel X', sobelx)
        cv2.imshow('Sobel Y', sobely)
        cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    return sobelxy


def canny_ED(image, showExtra=False):
    # Canny Edge Detection
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(image=img_gray, threshold1=100, threshold2=200)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    if showExtra:
        cv2.imshow('Canny Edge Detection', edges)
    return edges


def contours(image, showExtra: bool = False):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply binary thresholding

    # link explains thresholding very well
    # (https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_OTSU)
    # cv2.imshow('Binary image', thresh) # visualize the binary image

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    # cnts, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)#idk diff params
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if showExtra:
        # draw contours on the original image
        image_copy = image.copy()
        cv2.drawContours(image=image_copy, contours=cnts, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)
        # see the results
        cv2.imshow('None approximation', image_copy)
    return cnts


# just returns img version of the function above
def contoursImg(image, showExtra: bool = False):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply binary thresholding

    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_OTSU)  # try cv2.THRESH_BINARY next time?
    # visualize the binary image
    # cv2.imshow('Binary image', thresh)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    # cnts, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)#idk diff params
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours on the original image
    blank_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.drawContours(image=blank_image, contours=cnts, contourIdx=-1, color=(255, 255, 255), thickness=1,
                     lineType=cv2.LINE_AA)
    if showExtra:
        cv2.imshow('None approximation', blank_image)
    return blank_image


def adaptiveGaussianThresh(image, showExtra=False):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    th = cv2.blur(th, (5, 5))
    if showExtra:
        cv2.imshow('adaptive Gaussian Thresholding', th)
    return th


# TODO move to the top
smolBlu = adaptiveGaussianThresh(cv2.imread(SMOL_BLU))
smolRed = adaptiveGaussianThresh(cv2.imread(SMOL_RED))
bigBlu = adaptiveGaussianThresh(cv2.imread(BIG_BLU))
bigRed = adaptiveGaussianThresh(cv2.imread(BIG_RED))
greyTemplate = {'blue': smolBlu, 'red': smolRed}#{'smolBlu': smolBlu, 'smolRed': smolRed, 'bigBlu': bigBlu, 'bigRed': bigRed}


def convolutionMatching(image, panelType: str, showExtra: bool = False, coloredImg=None, verbose=False) -> bool:
    """
    checking an image against a template image to see how accurate it is
    returns a bool depending on how close of a match it is

    :param panelType:
    :param image: input image must be gray scale
    :param showExtra:
    :param coloredImg: must have an argument if sowExtra is true
    :return:
    """
    img = image.copy()
    # Apply template Matching
    if type(panelType) != str:
        raise Exception("panelType argument is not a string")
    w, h = greyTemplate[panelType].shape[::-1]
    res = cv2.matchTemplate(img, greyTemplate[panelType], cv2.TM_CCOEFF_NORMED)
    # cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, -1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if verbose:
        print("certainty", max_val)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(img, top_left, bottom_right, 0, 2)
    if showExtra:
        if coloredImg is None:
            raise Exception("third argument for image is empty")
        cv2.rectangle(coloredImg, top_left, bottom_right, (0, 0, 255), 2)
        cv2.imshow('Convolution Matching', coloredImg)
        # cv2.imshow('final img', img)
        # cv2.imshow('template', smolBlu)
        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.show()

    # TODO output rect should return near the up right corner to verify if it correct
    return max_val > 0.02


def findRect(box, box2, img,showExtra: bool = False):  # TODO optimise
    s = box.sum(axis=1)
    TL = box[np.argmin(s)]
    diff = np.diff(box, axis=1)
    BL = box[np.argmax(diff)]

    s = box2.sum(axis=1)
    TR = box2[np.argmax(s)]
    diff = np.diff(box2, axis=1)
    BR = box2[np.argmin(diff)]

    # cv2.circle(img, TL, 3, (255, 0, 0), 3)  # blue
    # cv2.circle(img, BR, 3, (255, 255, 255), 3)  # white
    # cv2.circle(img, BL, 3, (0, 255, 0), 3)  # green
    # cv2.circle(img, TR, 3, (0, 0, 255), 3)  # red

    # create linear equation
    points = [TL, BL]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
    # top offset
    f = lambda x1: int(m * x1 + c)
    x = (TL[0] - BL[0]) * .85
    if x == 0:
        dist = TL[1] - BL[1]
        TL[1] += dist * 0.9
        BL[1] -= dist * 0.9
    else:
        TL = (int(TL[0] + x), f(TL[0] + x))
        BL = (int(BL[0] - x), f(BL[0] - x))

    points = [TR, BR]
    x_coords, y_coords = zip(*points)
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m2, c2 = np.linalg.lstsq(A, y_coords, rcond=None)[0]
    f = lambda x1: int(m2 * x1 + c2)
    x = (TR[0] - BR[0]) * .85
    if x == 0:
        dist = TR[1] - BR[1]
        TR[1] += dist * .9
        BR[1] -= dist * .9
    else:
        TR = (int(TR[0] + x), f(TR[0] + x))
        BR = (int(BR[0] - x), f(BR[0] - x))

    # cv2.circle(img, TL, 3, (255, 0, 0), 3)  # blue
    # cv2.circle(img, BR, 3, (255, 255, 255), 3)  # white
    # cv2.circle(img, BL, 3, (0, 255, 0), 3)  # green
    # cv2.circle(img, TR, 3, (0, 0, 255), 3)  # red

    if showExtra:
        pts = np.array([TL, BR, TR, BL], np.int32)
        cv2.polylines(img, [pts], True, (255, 255, 0), 2)

    return TL, BR, BL, TR


def on_radio(*args):
    print('radio', args)
    if args[1] == "blue":
        global blue_bool
        blue_bool = args[0]
        if not args[0]:
            cv2.destroyWindow('blu')
            cv2.destroyWindow('blu+white')
    elif args[1] == "canny":
        global canny_bool
        canny_bool = args[0]
        if not args[0]:
            cv2.destroyWindow('Canny Edge Detection')
    elif args[1] == "contours":
        global contoure_bool
        contoure_bool = args[0]
        if not args[0]:
            cv2.destroyWindow("None approximation")
    else:
        global cam_view_bool
        cam_view_bool = args[0]
        if not args[0]:
            cv2.destroyWindow("final cut down")
