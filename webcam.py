# import the opencv library
import cv2
import numpy as np
import crappyV2
import filters

# define a video capture object
vid = cv2.VideoCapture(0)

win_name = "win"
cv2.namedWindow(win_name)
filters.make_slid(win_name)

while (True):
    ret, frame = vid.read()

    crappyV2.objectDetection(frame, 'red')


    cv2.imshow(win_name,frame)
    # crappyV2_CV.objectDetection(frame, 'red')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
