import cv2
import filters
import numpy as np
import math


def objectDetection(img, color):
    filtered_img = filters.RGB(img, color, showExtra=False)
    key_points = filters.blob_detection(filtered_img, showExtra=False)
    filtered_img = filters.ROI(filtered_img, key_points, showExtra=False)
    cnts = filters.contours(filtered_img, True)

    for i in range(len(cnts)):
        min_dist: float = 10000
        min_rect, min_tl2 = None, None

        rect = cv2.minAreaRect(cnts[i])
        box = cv2.boxPoints(rect)
        int_box = np.int0(box)

        # getting top left cord
        s = int_box.sum(axis=1)
        top_left = int_box[np.argmin(s)]

        cv2.drawContours(img, [int_box], 0, (0, 255, 0), 2)
        # cv2.putText(img, str(i), top_left, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        for j in range(len(cnts[(i + 1):])):
            j += i + 1

            # same box code a s above
            rect2 = cv2.minAreaRect(cnts[j])
            box2 = cv2.boxPoints(rect2)
            int_box2 = np.int0(box2)

            # getting top left cord
            s2 = int_box2.sum(axis=1)
            top_left2 = int_box2[np.argmin(s2)]

            # getting distance between boxes
            distance = math.dist(top_left, top_left2)
            if distance < min_dist:
                min_dist = distance
                min_rect = int_box2
                min_tl2 = top_left2

            if min_rect is None:
                continue
            # if box is on left and box 2 on right
            if min_tl2[0] > top_left[0]:
                plate_rect = filters.findRect(int_box, min_rect, img, False)
            else:
                plate_rect = filters.findRect(min_rect, int_box, img, False)


