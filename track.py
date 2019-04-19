import freenect
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

from data_structures import Regions


def get_size(img):
    if len(img.shape) == 3:
        return np.array(img.shape[:-1][::-1])
    else:
        return np.array(img.shape[::-1])


class Tracking:
    def __init__(self, split, img_size, hist_shape=[75, 75, 75], keep_hist=20):
        self.active_region = Regions()
        self.regions = self.construct_regions(split, img_size)
        self.hist_shape = hist_shape
        size = np.product(np.array(hist_shape)).astype('int')
        self.history = np.array([np.zeros(size, dtype='float32') for _ in range(0, keep_hist)])
        self.frame_num = 0

    def construct_regions(self, split, img_size):
        region_size = img_size // split

        # regions = []
        regions = Regions()
        for x in range(0, img_size[0], region_size[0]):
            for y in range(0, img_size[1], region_size[1]):
                # regions.append(Region(np.array([x,y]), region_size))
                regions.add_region((x, y), region_size)

        return regions

    def set_reference(self, img, top_left, bottom_right):
        filtered = filter_img(img)
        ref_area = filtered[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # first time
        if self.active_region.num_regions == 0:
            for reg in self.regions:
                if all(reg.top_left >= top_left) and all(reg.bottom_right <= bottom_right):
                    self.active_region._add_region(reg)

        hist = self.calc_feature_hist(ref_area)
        self.iir_filter(hist)

    def calc_feature_hist(self, img):
        hist = cv2.calcHist([img], [0, 1, 2], None, self.hist_shape, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def iir_filter(self, histogram, weight=1):
        # self.history *= (1 - weight)
        self.history[self.frame_num % len(self.history)] = histogram
        # self.history[self.frame_num % len(self.history)] *= weight
        self.ref_hist = np.mean(self.history[np.sum(self.history, axis=1) != 0], axis=0)
        plt.plot(self.ref_hist)
        plt.pause(0.05)
        plt.draw()
        # plt.clf()

    def compare(self, new_hist):
        for method in [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT,
                       cv2.HISTCMP_BHATTACHARYYA, cv2.HISTCMP_KL_DIV]:
            compare = cv2.compareHist(self.ref_hist, new_hist, method)
            print("%s:%s" % (method, compare))

        print("\n")

    def test_draw(self, img):
        # for row in self.regions.regions:
        #     for reg in row:
        #         reg.draw(img)

        # for row in self.regions.regions:
        #     for reg in row:
        #         reg.draw(img)

        draw = img.copy()
        draw1 = img.copy()
        draw2 = img.copy()
        draw3 = img.copy()
        draw4 = img.copy()

        self.active_region.draw_edges(draw)
        cv2.imshow('sa', draw)
        cv2.waitKey(0)

        # self.regions.center_region().draw(img, color=(0,0,255))

        # regions = self.regions.get_search_region(self.active_region, threshold=3)

        left_regions, right_regions, top_regions, bottom_regions = self.regions.get_search_regions(self.active_region,
                                                                                                   threshold=2)
        print(self.active_region)
        self.active_region.shift_area(left_regions)
        print(left_regions)
        self.active_region.draw_edges(draw1)
        cv2.imshow('sa', draw1)
        cv2.waitKey(0)

        left_regions, right_regions, top_regions, bottom_regions = self.regions.get_search_regions(self.active_region,
                                                                                                   threshold=2)
        self.active_region.shift_area(top_regions)
        self.active_region.draw_edges(draw2)
        cv2.imshow('sa', draw2)
        cv2.waitKey(0)

        left_regions, right_regions, top_regions, bottom_regions = self.regions.get_search_regions(self.active_region,
                                                                                                   threshold=2)
        self.active_region.shift_area(right_regions)
        self.active_region.draw_edges(draw3)
        cv2.imshow('sa', draw3)
        cv2.waitKey(0)

        left_regions, right_regions, top_regions, bottom_regions = self.regions.get_search_regions(self.active_region,
                                                                                                   threshold=2)
        self.active_region.shift_area(bottom_regions)
        self.active_region.draw_edges(draw4)
        cv2.imshow('sa', draw4)
        cv2.waitKey(0)

    def track(self, img):
        self.frame_num += 1
        filtered = filter_img(img)

        # sub_pot = 1
        # for region in self.regions:
        #     idxs = region.get_idxs()
        #     roi = img[idxs[0]:idxs[1], idxs[2]:idxs[3]]
        #     region.check(roi, self.ref_hist)
        #
        #
        # #     colors = ('b','g','r')
        # #     for i,c in enumerate(colors):
        # #          histr = cv2.calcHist([roi],[i],None,[256],[0,256])
        # #          plt.subplot(10,10,sub_pot)
        # #          plt.plot(histr,color=c)
        # #          plt.xlim([0,256])
        # #     sub_pot += 1
        # #
        # #
        # # plt.pause(0.05)
        # # plt.draw()
        # # plt.clf()
        #
        # for region in self.regions:
        #     region.draw(img)
        #
        #
        # cv2.imshow('tracking', img)
        # cv2.waitKey(0)

        # draw, draw1 = img.copy(), img.copy()

        draw = img.copy()
        # self.active_region.draw_edges(draw)
        search_regions = self.regions.get_search_regions_v2(self.active_region, threshold=2)
        search_regions.append(self.active_region)

        features = []
        scores = []
        regions = []

        for i, regs in enumerate(search_regions):
            if regs.num_regions == 0:
                scores.append(-1)
                features.append([0])
                continue

            top_left, bottom_right = regs.get_bbox()
            pixels = filtered[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            hist = np.array(self.calc_feature_hist(pixels))
            score = cv2.compareHist(self.ref_hist, hist, cv2.HISTCMP_INTERSECT)

            # print(names[i])
            # self.compare(hist)

            features.append(hist)
            scores.append(score)

        scores = np.array(scores)
        print(scores)
        # cv2.waitKey(0)

        if any(scores > 2):
            best = np.argmax(scores)
            self.active_region = search_regions[best]

            if self.active_region.num_regions == -1:
                print('\n%s\n' % self.active_region.num_regions)
                print(self.active_region)
                cv2.imshow('debug', draw)
                cv2.waitKey(0)

            self.iir_filter(features[best], weight=0.9)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (0, draw.shape[0] - 10)
        fontScale = 0.6
        fontColor = (0, 255, 0)
        lineType = 2

        cv2.putText(draw, 'Frame: %s' % self.frame_num,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        self.active_region.draw_edges(draw)

        # cv2.imshow('all', draw)
        # cv2.waitKey(0)

        return draw


def filter_img(img):
    # _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # img[img<255-20] += 20

    img = cv2.GaussianBlur(img, (15, 15), 30)

    return img


def get_video():
    img = freenect.sync_get_video()[0][:, :, ::-1]
    return img


def main():
    print("\nInitializing video connection ...\n")
    init = False

    img = get_video().copy()

    split = 50
    shape = get_size(img)
    tracker = Tracking(split, shape)

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
