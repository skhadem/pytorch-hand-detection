import freenect
import cv2
import numpy as np
import copy

from utils.img import get_size
from Tracker import Tracker


def get_video():
    img = freenect.sync_get_video()[0][:, :, ::-1]
    return img


def main():
    print("\nInitializing video connection ...\n")

    img = get_video().copy()

    split = 50
    shape = get_size(img)
    tracker = Tracker(split, shape)

    init = False
    extract_w = 200
    extract_h = 200
    h_w = np.array((extract_w, extract_h))

    h_w = (shape // split) * (4 * split / 10, 4 * split / 10)
    # h_w = (shape // split) * 4
    h_w = h_w.astype('int')

    top_left = (shape - h_w) // 2
    bottom_right = top_left + h_w

    while 1:
        img = get_video().copy()
        draw = img.copy()
        # img = np.zeros(img.shape)
        if init:
            draw = tracker.track(img)
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
