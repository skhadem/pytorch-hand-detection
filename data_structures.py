import numpy as np
import cv2


class BBox:
    def __init__(self, regions=None):
        self._init()
        if regions is not None:
            self._add_regions(regions)

    def _init(self):
        self.num_regions = 0
        self.center = None
        self.regions = [[]]

    def add_region(self, top_left, size):
        top_left = np.array(top_left)
        reg = Region(top_left, size)
        row = self._add_region(reg)
        self._sort_row(row)

    def add_area(self, area):
        rows = []
        for reg in area:
            rows.append(self._add_region(reg))

        for row in list(set(rows)):
            self._sort_row(row)

    def shift_area(self, area):
        max_y = self.get_global_max_y_reg()[1]
        new_max_y = area.get_global_max_y_reg()[1]
        max_x = self.regions[0][0].bottom_right[0]
        new_max_x = area.regions[0][0].bottom_right[0]

        if max_y == new_max_y:
            shift = len(area.regions)
            # left
            if max_x > new_max_x:
                self.regions = self.regions[:-shift]
            # right
            else:
                self.regions = self.regions[shift:]

        else:
            shift = len(area.regions[0])
            # top
            if max_y > new_max_y:
                for x in range(0, len(self.regions)):
                    self.regions[x] = self.regions[x][:-shift]
            # bottom
            else:
                for x in range(0, len(self.regions)):
                    self.regions[x] = self.regions[x][shift:]

        self.add_area(area)

    def potential(self, reg):
        # TODO: filter more
        row = self._add_region(reg)
        self._sort_row(row)

    def get_corners(self):
        top_left = self.regions[0][0].top_left
        bottom_right = self.regions[-1][-1].bottom_right

        return top_left, bottom_right

    def center_region(self):
        x_idx = len(self.regions)
        y_idx = 0
        for row in self.regions:
            y_idx += len(row)
        y_idx = y_idx // x_idx
        x_idx, y_idx = x_idx // 2, y_idx // 2

        return self.regions[x_idx-1][y_idx]

    def get_search_regions(self, active_area, threshold=1):
        """
        Takes the active region and returns the 9 new potential regions.


        Positions of the new top lefts are:

        | ----------------------- |
        | -----936--------------- |
        | -----714xxxx----------- |
        | -----825---x----------- |
        | ------x----x----------- |
        | ------xxxxxx----------- |
        | ----------------------- |
        | ----------------------- |


        :param active_area: The current active Regions
        :param threshold: The amount of regions to shift by
        :return: [Regions]
        """
        top_left_x, top_left_y = self.get_region_idx(active_area.regions[0][0])
        bottom_right_x, bottom_right_y = self.get_region_idx(active_area.regions[-1][-1])

        rv = [BBox() for _ in range(0, 9)]

        x_mult = [0, 1, -1]
        y_mult = [0, 1, -1]

        i = 0
        for xm in x_mult:
            for ym in y_mult:

                rows = []
                for x in range(top_left_x + (xm * threshold), bottom_right_x + 1 + (xm * threshold)):
                    if x < len(self.regions) and x >= 0:
                        for y in range(top_left_y + (ym * threshold), bottom_right_y + 1 + (ym * threshold)):
                            if y < len(self.regions[x]) and y >= 0:
                                rows.append(rv[i]._add_region(self.regions[x][y]))

                for row in rows:
                    rv[i]._sort_row(row)
                i += 1

        for regs in rv:
            if regs.num_regions != active_area.num_regions:
                regs.num_regions = 0

        return rv

    def overlap_iter(self, size, overlap):
        """
        Iterable that will yield overlapping boxes, meant to be used with HOG
        feature extraction
        :param size: The size of each sub box
        :param overlap: The amount to overlap each sub box by
        :return: Will yeld a box
        """

        for x in range(0, len(self.regions), overlap):
            for y in range(0, len(self.regions[x]), overlap):
                subbox = BBox()
                for xx in range(0, size[0]):
                    for yy in range(0, size[1]):
                        if (x + xx) < len(self.regions):
                            if (y + yy) < len(self.regions[x + xx]):
                                subbox._add_region(self.regions[x + xx][y + yy])

                if subbox.num_regions == (size[0] * size[1]):
                    yield subbox

        return

    def gradient_hist(self, img):
        pass

    def get_region_idx(self, region):
        """
        Gets the index of a region if it exits, returns None else
        TODO: make this raise something if the reg doesn't exist
        :param region:
        :return:
        """
        for x in range(0, len(self.regions)):
            for y in range(0, len(self.regions[x])):
                if all(region.center == self.regions[x][y].center):
                    return x, y

        return None

    def get_global_min_y_reg(self, region=None):
        if region is None:
            region = self
        min_y = 1e09
        reg = None
        for i, row in enumerate(region.regions):
            y = row[0].top_left[1]
            if y < min_y:
                min_y = y
                reg = row[0]
        return reg, min_y

    def get_global_max_y_reg(self, region=None):
        if region is None:
            region = self
        max_y = -1e09
        reg = None
        for i, row in enumerate(region.regions):
            y = row[-1].top_left[1]
            if y > max_y:
                max_y = y
                reg = row[-1]

        return reg, max_y

    def contains_reg(self, region):
        rv = False
        for reg in self:
            if all(region.center == reg.center):
                rv = True
                break

        return rv

    # %% drawing functions

    def draw_center_region(self, img):
        self.center_region().draw(img, color=(0, 0, 255))

    def draw_center_point(self, img, color=(0, 255, 0)):
        dims = self.regions[-1][-1].bottom_right - self.regions[0][0].top_left
        center_pt = (dims // 2) + self.regions[0][0].top_left
        cv2.circle(img, tuple(center_pt), 5, color, -1)

    def draw_edges(self, img, color=(0, 255, 0)):
        """
        Draws a box araound the edges of a bbox
        :param img:
        :param color:
        :return:
        """
        top_left, bottom_right = self.get_corners()
        cv2.rectangle(img, tuple(top_left), tuple(bottom_right), color, 2)
        # self.draw_center_point(img, color)

        for reg in self:
            reg.draw(img)

    # %% private funcitons %%

    def _add_regions(self, regs):
        for reg in regs:
            self._add_region(reg)

        self._sort_regions()

    def _sort_regions(self, row=None):
        if row is None:
            for row in range(0,len(self.regions)):
                self._sort_regions(row)
        else:
            self._sort_row(row)

    def _add_region(self, reg):
        row = self._new_reg(reg)
        self.regions[row].append(reg)
        self.num_regions += 1
        return row

    def _sort_row(self, row):
        self.regions[row] = sorted(self.regions[row],key= lambda reg: reg.center[1])

    def _new_reg(self, reg):
        # first time
        if self.num_regions == 0:
            return 0

        row_index = 0
        insert_index = -1
        for i, row in enumerate(self.regions):
            # only need to check the first one
            region = row[0]
            # newest has new min x
            if reg.center[0] < region.center[0]:
                if row_index == 0:
                    row_index = -1
                else:
                    insert_index = row_index
                break
            # found it
            elif reg.center[0] == region.center[0]:
                row_index = i
                break
            # keep looking
            else:
                row_index += 1

        # prepend
        if row_index == -1:
            self.regions = [[]] + self.regions
            row_index = 0
        # append
        if row_index == len(self.regions):
            self.regions.append([])

        if insert_index != -1:
            self.regions = self.regions[:insert_index] + [[]] + self.regions[insert_index:]

        return row_index

    # %% python funcitons
    def __str__(self):
        return_str = ""
        for i in range(0, len(self.regions)):
            for j in range(0, len(self.regions[i])):
                return_str += str(self.regions[i][j])
                if j != len(self.regions[i]) - 1:
                    return_str += ' -> '
            return_str += '\n'

        return return_str

    def __len__(self):
        return self.num_regions

    def __iter__(self):
        for row in self.regions:
            for reg in row:
                yield reg


class Region:
    def __init__(self, top_left, size):
        self.top_left = top_left
        self.size = size
        self.bottom_right = top_left + size
        self.active = False
        self.center = (top_left + self.bottom_right) // 2

    def check(self, pixels, reference):
        """ Deprc. """
        hist = cv2.calcHist([pixels], [0, 1, 2], None, [32]*3, [0, 256]*3)
        hist = cv2.normalize(hist, hist).flatten()

        score = cv2.compareHist(reference, hist, cv2.HISTCMP_CORREL)

        if score > 0.25:
            self.active = True
        else:
            self.active = False

    def draw(self, img, color=(255, 255, 255)):
        if self.active:
            color = (0, 255, 0)

        cv2.rectangle(img, tuple(self.top_left), tuple(self.top_left + self.size), color, thickness=1)

    def get_idxs(self):
        bottom_right = self.top_left + self.size
        return [self.top_left[1], bottom_right[1], self.top_left[0], bottom_right[0]]

    def hist(self, img):
        pixels = img[self.top_left[1]:self.bottom_right[1], self.top_left[0]:
                     self.bottom_right[0]]

        gradx = cv2.Sobel(pixels, cv2.CV_32F, 1, 0, ksize=1)
        grady = cv2.Sobel(pixels, cv2.CV_32F, 0, 1, ksize=1)

        mag, angle = cv2.cartToPolar(gradx, grady, angleInDegrees=True)

    def hog(self, img):
        pixels = img[self.top_left[1]:self.bottom_right[1], self.top_left[0]:
                     self.bottom_right[0]]

        sobelx = cv2.Sobel(pixels, cv2.CV_32F, 1, 0, ksize=1)

        sobely = cv2.Sobel(pixels, cv2.CV_32F, 0, 1, ksize=1)

        mag, angles = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
        mag = mag.flatten().reshape(mag.shape[0]*mag.shape[1], mag.shape[2])
        angles = angles.flatten().reshape(angles.shape[0]*angles.shape[1], angles.shape[2])

        maxes = np.argmax(mag, axis=1)

        mag = mag[np.arange(mag.shape[0]), maxes]
        angles = angles[np.arange(angles.shape[0]), maxes]
        angles[angles > 180] -= 180

        hog = np.zeros(9)

        for i, angle in enumerate(angles):
            ref = np.linspace(0, 160, 9)

            if angle > 160:
                ref[0] = 180

            idxs = abs(ref - angle).argsort()[:2]

            pcts = abs(ref[idxs] - angle) / np.sum(abs(ref[idxs] - angle))
            pcts = 1 - pcts

            ref[idxs] = -pcts
            ref[ref > 0] = 0
            ref = abs(ref)

            hog += ref*mag[i]

            # print(ref)
            # print(angle)


        #
        # draw = pixels.copy()
        # draw = cv2.resize(draw, (8*100,8*100), interpolation=cv2.INTER_NEAREST)
        # for x in range(0,8):
        #     for y in range(0,8):
        #         x = x *100
        #         y = y * 100
        #
            # cv2.imshow('split', pixels)
            # cv2.waitKey(0)

        return hog



    def __str__(self):
        return str(self.center)
