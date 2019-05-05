import cv2
import numpy as np
from scipy import spatial

import matplotlib.pyplot as plt
from data_structures import BBox


class Tracker:
    def __init__(self, split, img_size, hist_shape=75, keep_hist=20):
        # what will track the object
        self.active_region = BBox()

        # will be one big box over the whole image to store all the regions
        self.regions = self.construct_regions(split, img_size)

        # create histogram
        self.hist_shape = [hist_shape] * 3
        size = np.product(np.array(self.hist_shape)).astype('int')
        self.history = np.array([np.zeros(size, dtype='float32') for _ in range
                                (0, keep_hist)])

        self.frame_num = 0

        # viz
        fig = plt.figure()
        self.hist_plot = fig.add_subplot(1, 1, 1)

    def construct_regions(self, split, img_size):
        region_size = img_size // split
        region_size = np.array([8,8])

        # regions = []
        regions = BBox()
        for x in range(0, img_size[0], region_size[0]):
            for y in range(0, img_size[1], region_size[1]):
                regions.add_region((x, y), region_size)

        return regions

    def filter_img(self, img):
        # _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        # img[img<255-20] += 20

        img = cv2.GaussianBlur(img, (15, 15), 5)

        # cv2.imshow('sa', img)
        # cv2.waitKey(10)

        return img

    def set_reference(self, img, top_left, bottom_right):
        filtered = self.filter_img(img=img)
        ref_area = filtered[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # first time
        if self.active_region.num_regions == 0:
            for reg in self.regions:
                if all(reg.top_left >= top_left) and all(reg.bottom_right <= bottom_right):
                    self.active_region._add_region(reg)

        hist = self.calc_feature_hist(ref_area)
        self.iir_filter(hist)

    def calc_feature_hist(self, img):

        hist = cv2.calcHist([img], [0, 1, 2], None, self.hist_shape, [0, 255] * 3)

        # self.hist_plot.clear()
        # color = ('b', 'g', 'r')
        # for i, col in enumerate(color):
        #     print(np.max(hist[i]))
        #     histr = cv2.calcHist([img], [i], None, [75], [0, 255])
        #     histr = cv2.normalize(histr, histr)
        #     self.hist_plot.plot(histr, color=col)
        #     # plt.xlim([0, 256])
        #
        # plt.pause(0.01)
        # plt.draw()

        hist = cv2.normalize(hist, hist)
        hist = hist.flatten()
        return hist

    def iir_filter(self, histogram, weight=None):
        """
        Continuosly fliters the incoming histogram, optionally weighted
        :param histogram: The newest histogram to add to the history
        :param weight: How much to weight the new histogram
        :return:
        """
        if weight is not None:
            self.history *= (1 - weight)

        # add to the history
        self.history[self.frame_num % len(self.history)] = histogram

        if weight is not None:
            self.history[self.frame_num % len(self.history)] *= weight
        self.ref_hist = np.mean(self.history[np.sum(self.history, axis=1) != 0], axis=0)
        # print(len(self.ref_hist[self.ref_hist == 0]) / len(self.ref_hist))
        # self.hist_plot.clear()
        # self.hist_plot.plot(self.ref_hist)
        # plt.pause(0.01)
        # plt.draw()

    def compare(self, new_hist):
        for method in [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT,
                       cv2.HISTCMP_BHATTACHARYYA, cv2.HISTCMP_KL_DIV]:
            compare = cv2.compareHist(self.ref_hist, new_hist, method)
            print("%s:%s" % (method, compare))

        print("\n")

    def track(self, img):
        """
        Main function to update the active region, should be called every frame
        Computes the feature vector for potential new bounding boxes and then
        shifts if the correlation is high enough
        :param img: the image to track
        :return: a copy of the img with the drawn active region
        """
        self.frame_num += 1

        filtered = self.filter_img(img)
        draw = img.copy()

        search_regions = self.regions.get_search_regions(self.active_region,
                                                         threshold=1)

        features = []
        scores = []
        regions = []

        for i, regs in enumerate(search_regions):
            # usually corner of image
            if regs.num_regions == 0:
                scores.append(-1)
                features.append([0])
                continue

            # regs.draw_edges(draw)

            # get cropped region of potential region
            top_left, bottom_right = regs.get_corners()
            pixels = filtered[top_left[1]:bottom_right[1],
                              top_left[0]:bottom_right[0]]

            # calc and compare hist
            hist = np.array(self.calc_feature_hist(pixels))
            # score = cv2.compareHist(hist, self.ref_hist, cv2.HISTCMP_INTERSECT)
            score = spatial.distance.cosine(hist, self.ref_hist)
            features.append(hist)
            scores.append(score)

        scores = np.array(scores)

        # update
        if any(scores > 0):
            # best = np.argmax(scores)
            best = np.argmin(scores)
            if best != 0:
                self.active_region = search_regions[best]
                self.iir_filter(features[best], weight=None)

        # draw

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner = (0, draw.shape[0] - 10)
        font_scale = 0.6
        font_color = (0, 255, 0)
        line_type = 2

        cv2.putText(draw, 'Frame: %s' % self.frame_num,
                    bottom_left_corner,
                    font,
                    font_scale,
                    font_color,
                    line_type)

        self.active_region.draw_edges(draw)

        return draw

    def track_hog(self, img):
        """
        Will track the region using a method of extracting an HOG feature
        vector
        :param img: The image to track
        :return: a copy of img with the drawn active region
        """
        draw = img.copy()
        # draw = self.filter_img(draw)

        # filtered = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
        filtered = np.float32(draw) / 255.0
        sobelx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=1)
        # absx = cv2.convertScaleAbs(sobelx)
        absx = abs(sobelx)

        sobely = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=1)
        # absy = cv2.convertScaleAbs(sobely)
        absy = abs(sobely)

        add = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
        mag, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)

        # cv2.imshow('gradientx', absx)
        # cv2.imshow('gradienty', absy)
        # cv2.imshow('xy', mag)
        #
        # cv2.waitKey(0)

        test = cv2.resize(draw, (320, 640), interpolation=cv2.INTER_NEAREST)

        # hog = self.active_region.regions[3][1].hog(img)
        ref = np.linspace(0, 160, 9)
        for reg in self.active_region:
            hog = reg.hog(img)
            # print(hog)

            hog /= np.linalg.norm(hog)
            # print(hog)
            hog *= 12

            center = reg.center * 5
            for i, h in enumerate(hog):
                x1 = center[0] - h * np.cos((ref[i] * np.pi / 180) + np.pi / 2)
                y1 = center[1] - h * np.sin((ref[i] * np.pi / 180) + np.pi / 2)
                p1 = np.array((x1, y1)).astype('int')

                x2 = center[0] + h * np.cos((ref[i] * np.pi / 180) + np.pi / 2)
                y2 = center[1] + h * np.sin((ref[i] * np.pi / 180) + np.pi / 2)
                p2 = np.array((x2, y2)).astype('int')

                cv2.line(test, tuple(p1), tuple(p2), (0, 0, 255),
                thickness=2)

        # center = self.active_region.center_region().center

        # cv2.circle(draw, tuple(center), 5, (0, 255, 0), -1)
        #
        # for angle in ref:
        #
        #     x = center[0] + 10 * np.cos((angle * np.pi / 180) + np.pi / 2)
        #     y = center[1] + 10 * np.sin((angle * np.pi / 180) + np.pi / 2)
        #     p2 = np.array((x, y)).astype('int')
        #     cv2.line(draw, tuple(center), tuple(p2), (0, 0, 255), thickness=1)
        #

        cv2.imshow('sa', test)

        import matplotlib.pyplot as plt

        from skimage.feature import hog
        from skimage import data, exposure

        image = img

        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), visualize=True,
                            multichannel=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True,
                                       sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image,
                                                        in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()












        cv2.waitKey(0)

        # self.active_region.draw_edges(draw)



        #
        # for box in self.active_region.overlap_iter((2, 2), 1):
        #     box.draw_edges(draw, color=(255, 0, 0))
        #
        #     for reg in box:
        #         reg.hist(filtered)
        #
        #     test = cv2.resize(draw, (320, 640))
        #     cv2.imshow('sa', test)
        #     cv2.waitKey(100)

        return draw
