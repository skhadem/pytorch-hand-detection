import numpy as np
import cv2
import os

from Tracker import Tracker
from utils.img import get_size


def main():
    cap = cv2.VideoCapture(os.path.abspath(os.path.dirname(__file__)) + '/OJ.gif')
    if not cap.isOpened():
        return -1

    split = 50
    _, frame = cap.read()
    size = get_size(frame)
    tracker = Tracker(split, size)
    init = False
    top_left = np.array(([120, 90]))
    bottom_right = np.array(([175, 124]))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        draw = frame.copy()
        if not init:
            draw = cv2.rectangle(draw, tuple(top_left),
                                 tuple(bottom_right), (0, 255, 0),
                                 thickness=2)
            tracker.set_reference(frame, top_left, bottom_right)
        else:
            draw = tracker.track(frame)

        cv2.imshow('fame', draw)
        duration = 100 if init else 0
        key = cv2.waitKey(duration)

        if key == 27:
            break
        elif key == 32:
            init = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()