import filters
import cv2
import numpy as np
import math


def objectDetection(img, color):
    filtered_img = filters.RGB(img, color)
    key_points = filters.blob_detection(filtered_img)
    filtered_img = filters.ROI(filtered_img, key_points)
    cnts = filters.contours(filtered_img, True)
    # print(cnts)

    for i in range(len(cnts)):
        # do these when u see them in code===
        min_dist: float = 10000
        min_rect, min_tl2 = None, None
        # ====================================

        rect = cv2.minAreaRect(cnts[i])  # creates the smolest rect possible
        box = cv2.boxPoints(rect)  # gets cords of the corners of the box
        # print(box)
        int_box = np.int0(box)  # make box arr into np arr
        # getting top left
        s = int_box.sum(axis=1)
        top_left = int_box[np.argmin(s)]

        cv2.drawContours(img, [int_box], 0, (0, 255, 0), 2)  # draw green box on img
        # cv2.putText(img, str(i), top_left, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        # doing distance comparisons
        # basically comparing each box with each other box in n^2 time
        for j in range(len(cnts[(i + 1):])):  # doesn't have to compair the first box with itself
            j += i + 1

            # same box code as above
            rect2 = cv2.minAreaRect(cnts[j])
            box2 = cv2.boxPoints(rect2)
            intBox2 = np.int0(box2)
            # getting top left corner
            s = intBox2.sum(axis=1)
            topLeft2 = intBox2[np.argmin(s)]

            # getting distance between boxes
            distance = math.dist(top_left, topLeft2)
            if distance < min_dist:
                min_dist = distance
                min_rect = intBox2  # TODO
                min_tl2 = topLeft2

        if min_rect is None:  # if it sees nothing
            continue
        # basically finding the int_box's closest/min rect
        if min_tl2[0] > top_left[0]:  # if box is on left and box2 is on right
            plateRect = filters.findRect(int_box, min_rect, img)
        else:  # if box is on the right and box2 is on left
            plateRect = filters.findRect(min_rect, int_box, img)

        result = filters.perspective_transform(img, plateRect, color, False)
        th = filters.adaptiveGaussianThresh(result, False)
        isPanel = filters.convolutionMatching(th, color, False, result, verbose=False) # explain the certainty

        pts = np.array([plateRect[0], plateRect[1], plateRect[3], plateRect[2]], np.int32)
        if isPanel:
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            # finding center
            # x = [p[0] for p in pts]
            # y = [p[1] for p in pts]
            # centroid = (sum(x) / len(pts), sum(y) / len(pts))
            # print("plate cord", centroid)
        else:
            cv2.polylines(img, [pts], True, (0, 0, 255), 1)
