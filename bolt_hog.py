import cv2
import numpy as np
import os

from utils.img import get_size
from Tracker import Tracker


def main():
    img = cv2.imread(os.path.abspath(os.path.dirname(__file__)) + '/bolt.png')
    split = 8
    shape = get_size(img)
    tracker = Tracker(split, shape)

    init = False


    top_left = np.array((0, 0))
    bottom_right = shape

    while 1:
        draw = img.copy()
        # img = np.zeros(img.shape)
        if init:
            draw = tracker.track_hog(img)
            break
        else:
            pass
            # draw = cv2.rectangle(draw, tuple(top_left),
            #                      tuple(bottom_right), (0, 255, 0),
            #                      thickness=2)

        cv2.imshow('Video', draw)
        retval = cv2.waitKey(10)

        if retval == 27:
            break
        elif retval == 32:
            if not init:
                tracker.set_reference(img, top_left, bottom_right)
                init = True
        elif retval == 112:
            cv2.waitKey(0)
            # else:
            #     tracker.compare(img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
