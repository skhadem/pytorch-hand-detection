import cv2
import numpy as np

from utils.img import get_size
from Tracker import Tracker


def main():
    print("\nInitializing video connection ...\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to initialize video connection")
        return -1

    _, img = cap.read()
    split = 30
    shape = get_size(img)
    tracker = Tracker(split, shape)

    init = False

    h_w = (shape // split) * (4 * split / 10, 4 * split / 10)
    h_w = h_w.astype('int')

    top_left = (shape - h_w) // 2
    bottom_right = top_left + h_w

    while 1:
        _, img = cap.read()
        draw = img.copy()
        # img = np.zeros(img.shape)
        if init:
            draw = tracker.track_hog(img)
        else:
            draw = cv2.rectangle(draw, tuple(top_left + 2),
                                 tuple(bottom_right + 2), (0, 255, 0),
                                 thickness=2)

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
