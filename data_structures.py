import numpy as np
import cv2

class Regions:
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


        if  max_y == new_max_y:
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
                for x in range(0,len(self.regions)):
                    self.regions[x] = self.regions[x][:-shift]
            # bottom
            else:
                for x in range(0,len(self.regions)):
                    self.regions[x] = self.regions[x][shift:]

        self.add_area(area)


    def potential(self, reg):
        # TODO: filter more
        row = self._add_region(reg)
        self._sort_row(row)

    def get_bbox(self):
        top_left = self.regions[0][0].top_left
        bottom_right = self.regions[-1][-1].bottom_right

        return top_left, bottom_right


    def center_region(self):
        y_idx = len(self.regions)
        x_idx = 0
        for row in self.regions:
            x_idx += len(row)
        x_idx = x_idx // y_idx
        x_idx, y_idx = x_idx // 2, y_idx // 2

        return self.regions[x_idx-1][y_idx]

    def get_search_region(self, active_area, threshold=1):
        top_left_x, top_left_y = self.get_region_idx(active_area.regions[0][0])
        bottom_right_x, bottom_right_y = self.get_region_idx(active_area.regions[-1][-1])

        regions = Regions()

        rows = []
        for x in range(top_left_x - threshold, top_left_x):
            if x >= 0:
                for y in range(top_left_y, bottom_right_y + 1):
                    if y < len(self.regions[x]):
                        rows.append(regions._add_region(self.regions[x][y]))

        for x in range(bottom_right_x + 1, bottom_right_x + 1 + threshold):
            if x < len(self.regions):
                for y in range(top_left_y, bottom_right_y + 1):
                    if y < len(self.regions[x]):
                        rows.append(regions._add_region(self.regions[x][y]))

        for x in range(top_left_x, bottom_right_x + 1):
            if x < len(self.regions):
                for y in range(top_left_y - threshold, top_left_y):
                    if y >=0:
                        rows.append(regions._add_region(self.regions[x][y]))

        for x in range(top_left_x, bottom_right_x + 1):
            if x < len(self.regions):
                for y in range(bottom_right_y + 1, bottom_right_y + 1 + threshold):
                    if y < len(self.regions[x]):
                        rows.append(regions._add_region(self.regions[x][y]))

        for x in range(top_left_x - threshold, bottom_right_x + threshold + 1):
            if x < len(self.regions) and x >= 0:
                for y in range(top_left_y - threshold, bottom_right_y + threshold + 1):
                    if y < len(self.regions[x]) and y >= 0:
                        if (x < top_left_x or x > bottom_right_x) and (y < top_left_y or y > bottom_right_y):
                            rows.append(regions._add_region(self.regions[x][y]))

        for row in list(set(rows)):
            regions._sort_row(row)

        return regions

    def get_search_regions(self, active_area, threshold=1):
        top_left_x, top_left_y = self.get_region_idx(active_area.regions[0][0])
        bottom_right_x, bottom_right_y = self.get_region_idx(active_area.regions[-1][-1])

        left_regions = Regions()
        right_regions = Regions()
        top_regions = Regions()
        bottom_regions = Regions()
        top_left_regions = Regions()
        top_right_regions = Regions()
        bottom_left_regions = Regions()
        bottom_right_regions = Regions()


        # left
        for x in range(top_left_x - threshold, top_left_x):
            if x >= 0:
                for y in range(top_left_y, bottom_right_y + 1):
                    if y < len(self.regions[x]):
                        left_regions._add_region(self.regions[x][y])

        # right
        for x in range(bottom_right_x + 1, bottom_right_x + 1 + threshold):
            if x < len(self.regions):
                for y in range(top_left_y, bottom_right_y + 1):
                    if y < len(self.regions[x]):
                        right_regions._add_region(self.regions[x][y])

        # top
        for x in range(top_left_x, bottom_right_x + 1):
            if x < len(self.regions):
                for y in range(top_left_y - threshold, top_left_y):
                    if y >=0:
                        top_regions._add_region(self.regions[x][y])
        # bottom
        for x in range(top_left_x, bottom_right_x + 1):
            if x < len(self.regions):
                for y in range(bottom_right_y + 1, bottom_right_y + 1 + threshold):
                    if y < len(self.regions[x]):
                        bottom_regions._add_region(self.regions[x][y])

        return [left_regions, right_regions, top_regions, bottom_regions]


    def get_search_regions_v2(self, active_area, threshold=1):
        top_left_x, top_left_y = self.get_region_idx(active_area.regions[0][0])
        bottom_right_x, bottom_right_y = self.get_region_idx(active_area.regions[-1][-1])

        left_regions = Regions()
        right_regions = Regions()
        top_regions = Regions()
        bottom_regions = Regions()
        top_left_regions = Regions()
        top_right_regions = Regions()
        bottom_left_regions = Regions()
        bottom_right_regions = Regions()

        rv = [left_regions, right_regions, top_regions, bottom_regions,
                top_left_regions, top_right_regions, bottom_left_regions,
                bottom_right_regions]

        rows = []
        # left
        for x in range(top_left_x - threshold, bottom_right_x + 1 - threshold):
            if x < len(self.regions) and x >= 0:
                for y in range(top_left_y, bottom_right_y + 1):
                    if y < len(self.regions[x]) and y >= 0:
                        rows.append(left_regions._add_region(self.regions[x][y]))

        for row in rows:
            left_regions._sort_row(row)
        rows = []

        # right
        for x in range(top_left_x + threshold, bottom_right_x + 1 + threshold):
            if x < len(self.regions) and x >= 0:
                for y in range(top_left_y, bottom_right_y + 1):
                    if y < len(self.regions[x]) and y >= 0:
                        rows.append(right_regions._add_region(self.regions[x][y]))

        for row in list(set(rows)):
            right_regions._sort_row(row)
        rows = []

        # top
        for x in range(top_left_x, bottom_right_x + 1):
            if x < len(self.regions) and x >= 0:
                for y in range(top_left_y - threshold, bottom_right_y + 1):
                    if y < len(self.regions[x]) and y >= 0:
                        rows.append(top_regions._add_region(self.regions[x][y]))

        for row in list(set(rows)):
            top_regions._sort_row(row)
        rows = []

        # bottom
        for x in range(top_left_x, bottom_right_x + 1):
            if x < len(self.regions) and x >= 0:
                for y in range(top_left_y, bottom_right_y + 1 + threshold):
                    if y < len(self.regions[x]) and y >= 0:
                        rows.append(bottom_regions._add_region(self.regions[x][y]))

        for row in list(set(rows)):
            bottom_regions._sort_row(row)
        rows = []

        # top_left
        for x in range(top_left_x - threshold, bottom_right_x + 1 - threshold):
            if x < len(self.regions) and x >= 0:
                for y in range(top_left_y - threshold, bottom_right_y + 1 - threshold):
                    if y < len(self.regions[x]) and y >= 0:
                        rows.append(top_left_regions._add_region(self.regions[x][y]))

        for row in list(set(rows)):
            top_left_regions._sort_row(row)
        rows = []

        # top_right
        for x in range(top_left_x + threshold, bottom_right_x + 1 + threshold):
            if x < len(self.regions) and x >= 0:
                for y in range(top_left_y - threshold, bottom_right_y + 1 - threshold):
                    if y < len(self.regions[x]) and y >= 0:
                        rows.append(top_right_regions._add_region(self.regions[x][y]))

        for row in list(set(rows)):
            top_right_regions._sort_row(row)
        rows = []

        # bottom_left
        for x in range(top_left_x + threshold, bottom_right_x + 1 + threshold):
            if x < len(self.regions) and x >= 0:
                for y in range(top_left_y + threshold, bottom_right_y + 1 + threshold):
                    if y < len(self.regions[x]) and y >= 0:
                        rows.append(bottom_left_regions._add_region(self.regions[x][y]))

        for row in list(set(rows)):
            bottom_left_regions._sort_row(row)
        rows = []



        # bottom_right
        for x in range(top_left_x - threshold, bottom_right_x + 1 - threshold):
            if x < len(self.regions) and x >= 0:
                for y in range(top_left_y + threshold, bottom_right_y + 1 + threshold):
                    if y < len(self.regions[x]) and y >= 0:
                        rows.append(bottom_right_regions._add_region(self.regions[x][y]))

        for row in list(set(rows)):
            bottom_right_regions._sort_row(row)
        rows = []

        # signs = [1,-1]
        # x_thresh_mults = [1,0,1,0]
        # y_thresh_mults = [1,0]
        #
        # i = 0
        # for sign in signs:
        #     for x_thresh_mult in x_thresh_mults:
        #         for y_thresh_mult in y_thresh_mults:
        #
        #             for x in range(top_left_x + (sign * x_thresh_mult * threshold),
        #                             bottom_right_x + (sign * x_thresh_mult * threshold) + 1):
        #                 if x < len(self.regions) and x >= 0:
        #                     for y in range(top_left_y + (sign * y_thresh_mult * threshold),
        #                                     bottom_right_y + (sign * y_thresh_mult * threshold) + 1):
        #                         if y < len(self.regions[x]) and y >=0:
        #                             rv[i]._add_region(self.regions[x][y])
        #         i += 1


        for regs in rv:
            if regs.num_regions != active_area.num_regions:
                regs.num_regions = 0

        return rv

    def get_region_idx(self, region):
        for x in range(0,len(self.regions)):
            for y in range(0,len(self.regions[x])):
                if all(region.center == self.regions[x][y].center):
                    return x, y

    def get_global_min_y_reg(self, region=None):
        if region == None:
            region = self
        min_y = 1e09
        reg = None
        for i,row in enumerate(region.regions):
            y = row[0].top_left[1]
            if y < min_y:
                min_y = y
                reg = row[0]
        return reg, min_y

    def get_global_max_y_reg(self, region=None):
        if region == None:
            region = self
        max_y = -1e09
        reg = None
        for i,row in enumerate(region.regions):
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
        self.center_region().draw(img, color=(0,0,255))

    def draw_center_point(self, img):
        dims = self.regions[-1][-1].bottom_right - self.regions[0][0].top_left
        center_pt = (dims // 2) + self.regions[0][0].top_left
        cv2.circle(img, tuple(center_pt), 5, (0,255,0), -1)

    def draw_edges(self, img):
        top_left, bottom_right = self.get_bbox()

        cv2.rectangle(img, tuple(top_left), tuple(bottom_right), (0,255,0), 2)

        self.draw_center_point(img)

        # for reg in self:
        #     reg.draw(img)

    # %% private funcitons

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
        for i,row in enumerate(self.regions):
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
        for i in range(0,len(self.regions)):
            for j in range(0,len(self.regions[i])):
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
        hist = cv2.calcHist([pixels], [0,1,2], None, [32,32,32], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist).flatten()

        score = cv2.compareHist(reference, hist, cv2.HISTCMP_CORREL)

        if score > 0.25:
            self.active = True
        else:
            self.active = False

    def draw(self, img, color=(255,255,255)):
        if self.active:
            color = (0, 255, 0)

        cv2.rectangle(img, tuple(self.top_left), tuple(self.top_left + self.size - 2), color, thickness=1)


    def get_idxs(self):
        bottom_right = self.top_left + self.size
        return [self.top_left[1], bottom_right[1], self.top_left[0], bottom_right[0]]

    def __str__(self):
        return str(self.center)

class Feature:
    def __init__(self):
        pass



# class ActiveRegion(Regions):
#     def __init__(self):
#         super().__init__()
#
#     def add_region(self, reg):
#         super().add_region(reg.top_left, reg.size)
#
#     def draw_edges(self, img):
#         top_left = self.regions[0][0].top_left
#         bottom_right = self.regions[-1][-1].bottom_right
#
#         cv2.rectangle(img, tuple(top_left), tuple(bottom_right), (0,255,0), 2)
#
#         self.draw_center_point(img)
#
#         for reg in self:
#             reg.draw(img)


class Tree:
    def __init__(self):
        self.root = None
        self.size = 0

    def leafs(self):
        pass

    def root(self):
        return self.root

    def __len__(self):
        return self.size

    def add_node(self, node):
        if self.root is None:
            self.root = node
            return

class Node:
    def __init__(self, center, left=None, right=None, top=None, bottom=None):
        self.center = center
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def is_root(self):
        return self.left is None or self.right is None \
         or self.top is None or self.bottom is None


class Graph:
    def __init__(self):
        self.center = None
        self.num_verts = 0
        self.verts = {}

    def add_vert(self, center):
        vert = Vertex(center)

    def __len__(self):
        return self.num_verts


class Vertex:
    def __init__(self, center):
        self.center = center
