# encoding: utf-8
'''
@Author: 刘琛
@Time: 2019/11/13 15:23
@Contact: victordefoe88@gmail.com

@File: utils.py
@Statement:

'''


import numpy as np
import collections
import pandas as pd
import os
import math
import cv2
import json
from tqdm import tqdm

# default dataset configuration
config_dict = {
    'dataset_name': 'collection_1',
    'dataset_path': r'Z:\research\datasets\GoogleEarth',
    'dataset_level': ['18', '17', '16']

}


def decimal2sexagesimal(raw_num):
    """
    This is for transforming the data into sexagesimal and keep parts of digits
    rule is:
        keep the integer part and transfer the fractional part
    :param raw_num:  float
    :return: sexagesimal number
    """
    digree = int(raw_num)
    frac = raw_num - digree
    m = int(frac * 60)
    s = round((frac * 60 - m) * 60, 2)

    return digree, m, s


def sexagesimal2decimal(raw_num_list):
    """
    Reversion process of decimal2sexagesimal
    :param raw_num: a list contain 3 numbers
    :return: float
    """
    return round(
        raw_num_list[0] +
        raw_num_list[1] /
        60 +
        raw_num_list[2] /
        3600,
        10)

#


def set_bond(dataset_conf=config_dict):
    """
    Finding the intersect maximum rectangle coordinates
    :param dataset_conf: parameters
    :return: bondary
    """
    Bondary = collections.namedtuple('Bondary', 'west, east, north, south')
    # bondary = Bondary(118.3247362807, 118.0864515186, 34.0708663191, 33.9221215026)
    dataset_name = dataset_conf['dataset_name']
    dataset_path = dataset_conf['dataset_path']
    dataset = os.path.join(dataset_path, dataset_name)
    assert os.path.exists(os.path.join(dataset, 'location.csv')
                          ), 'location.csv is not exists!! check it!'
    info = pd.read_csv(os.path.join(dataset, 'location.csv'), engine='python')

    info.drop(info.filter(regex='Unnamed'), axis=1, inplace=True)
    info.drop(['Elevation'], axis=1, inplace=True)
    # 在这个数据集中，经度为负的是西经，正的是东经，经度数值越大越东
    # 纬度负的是南维，正的是北纬，数值越大越北
    east = info.Longitude.loc[ info.Longitude > info.Longitude.mean()].min()
    west = info.Longitude.loc[ info.Longitude < info.Longitude.mean()].max()
    north = info.Latitude.loc[ info.Latitude > info.Latitude.mean()].min()
    south = info.Latitude.loc[info.Latitude < info.Latitude.mean()].max()

    # bondary = Bondary(
    #     info.Longitude.min(),
    #     info.Longitude.max(),
    #     info.Latitude.max(),
    #     info.Latitude.min())
    bondary = Bondary(west, east, north, south)
    bondary_se = Bondary([decimal2sexagesimal(west)],
                         [decimal2sexagesimal(east)],
                         [decimal2sexagesimal(north)],
                         [decimal2sexagesimal(south)])
    return bondary, bondary_se


bondary, bondary_se = set_bond()


##
def get_file_size(file, form='M'):
    """

    :param file: string
    :return:
    """
    size = os.path.getsize(file)
    try:
        bytes = float(size)
        kb = bytes / 1024
    except BaseException:
        print("Wrong format of bytes")
        return "Error"
    M = kb / 1024
    G = M / 1024
    ret_dict = {'K': kb, 'M': M, 'G': G, 'k': kb, 'm': M, 'g': G}
    return ret_dict[form]


# seperate the big pictures into small patchs for retrieval
class AMap():
    def __init__(self, dataset_conf=config_dict):
        """
        The class to deal with all type dataset
        :param dataset_conf: basic parameters for dataset
        """
        self.dataset_conf = dataset_conf
        self.level = '18'  # '16' | '17' | '18'
        self.name = dataset_conf['dataset_name']
        self.dataset_root = dataset_conf['dataset_path']
        self.amap_file = os.path.join(
            self.dataset_root, self.name, 'all', self.level, 'all_cut.tif')
        self.gt_file = os.path.join(
            self.dataset_root, self.name, 'patch', 'gt.csv')

        self.sep_output_path = os.path.join(
            self.dataset_root, self.name, 'seps', self.level)

        if not os.path.exists(self.sep_output_path):
            os.makedirs(self.sep_output_path)

        self.sep_ratio = None
        self.bonds = self.get_bonds()
        self.glob_info = {'lng_diff': abs(self.bonds.west - self.bonds.east),
                          'lat_diff': abs(self.bonds.north - self.bonds.south)}

    def get_bonds(self):
        """
        wrapper of returning bondaries func
        :return: set_bond
        """
        return set_bond(self.dataset_conf)

    def get_distance(self, lat1, lng1, lat2, lng2):
        """
        transfer to real distance, see https://zhuanlan.zhihu.com/p/42948839
        :param lat1: point 1 latitude
        :param lng1: point 1 longitude
        :param lat2: point 2 latitude
        :param lng2: point 2 longitude
        :return: distance in reality, unit is meter(M)
        """
        EARTH_RADIUS = 6378.137
        # need to transfer the latitude into 0~180 range (original one
        # is -90~90 range) .
        lat1, lat2 = 90 + lat1, 90 + lat2

        def HaverSin(theta):
            return math.pow(math.sin(theta / 2), 2)

        def rad(theta):
            """
            Transfer the latitude and longitude angle to radius
            :param theta: angle (the numeral value of lat and lng)
            :return: radius
            """
            return theta * math.pi / 180.0

        radLat1 = rad(lat1)
        radLat2 = rad(lat2)
        a = radLat1 - radLat2
        b = rad(lng1) - rad(lng2)

        s = 2 * math.asin(math.sqrt(HaverSin(a) + \
                          math.cos(radLat1) * math.cos(radLat2) * HaverSin(b)))
        # s = math.acos((math.cos(radLat1)*math.cos(radLat2)*math.cos(b) + math.sin(radLat1)*math.sin(radLat2)))

        s = s * EARTH_RADIUS
        s = (s * 10000) / 10
        return s

    def get_bond_dis(self):
        bonds = self.bonds
        # lng_diff = abs(bonds.west - bonds.east)
        # lat_diff = abs(bonds.north - bonds.south)
        lat_dis = self.get_distance(
            bonds.west, bonds.north, bonds.east, bonds.north)
        lng_dis = self.get_distance(
            bonds.west, bonds.north, bonds.west, bonds.south)
        return lat_dis, lng_dis

    def __cal_sep_loc(self, glob_size, pixel_step, patch_i, patch_j):
        """
        According to the pixel step and bondary,
        caiculate the sep_patch's center location.

        :param glob_size,: a tuple for the size
        :param pixel_step: a tuple for pixel step (x,y)
        :param patch_i: This is the i_th patch for rows dimension
        :param patch_j: j_th patch for cols dimension
        :return: exact center location for patch
        """

        loc_x = (patch_i * pixel_step[0] + int(0.5 * pixel_step[0])
                 ) * (self.glob_info['lat_diff'] / glob_size[0])
        loc_y = (patch_j * pixel_step[1] + int(0.5 * pixel_step[0])
                 ) * (self.glob_info['lng_diff'] / glob_size[1])

        return (loc_x, loc_y)

    def drop_black_edge(self, img, keep_the_preimg=True):
        """
        cut the edge , base on PyOpencv, only for rectangle, cut from the middle
        :param img: Opencv object
        :param keep_the_preimg: do you want to keep the final image
        :return: Opencv Mat opbject
        """
        image = cv2.medianBlur(img, 5)
        b = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
        binary_image = b[1]
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        x = binary_image.shape[0]
        y = binary_image.shape[1]
        print(x,y)
        edges_x = []
        edges_y = []
        for i in range(x):
            if binary_image[i][y//2] == 255:
                edges_x.append(i)
        for j in range(y):
            if binary_image[x//2][j] == 255:
                edges_y.append(j)

        left = min(edges_x)  # 左边界
        right = max(edges_x)  # 右边界
        width = right - left  # 宽度
        bottom = min(edges_y)  # 底部
        top = max(edges_y)  # 顶部
        height = top - bottom  # 高度
        pre_picture = img[left:left + width, bottom:bottom + height, :]
        print('Cut the black edge!! now the size is [width: %s AND height: %s] ' % (repr(width), repr(height)) )
        if keep_the_preimg:
            cv2.imwrite(os.path.join(
            self.dataset_root, self.name, 'all', self.level, 'all_dropbe.jpg'), pre_picture)
        return pre_picture

    def crop4minima(self, img, keep_the_preimg=True):
        """
        Assume the orignal image is not a rectangle shape, take the minimal rectangle crop area
        :param img: opencv object
        :param keep_the_preimg: do you want to keep the final image
        :return: opnecv object
        """
        image = cv2.medianBlur(img, 5)
        b = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
        binary_image = b[1]
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        x = binary_image.shape[0]
        y = binary_image.shape[1]
        print(x, y)

        # dst = cv2.cornerHarris(binary_image, 2, 3, 0.04)
        # a = dst > 0.01 * dst.max()
        # # b = pd.DataFrame(a)
        #
        # print(np.where(a==True)[0].shape)
        # dst

        # edges_x_left, edges_x_right = [], []
        # edges_y_up, edges_y_down = [], []
        # ## scan the center point
        # def scan_x():
        #     for j in range(y):
        #         for i in range(x):
        #             if binary_image[i][j] == 255:
        #                 edges_x_left.append((i,j))
        #             if binary_image[i][y-j-1] == 255:
        #                 edges_x_right.append((i,j))
        #             if len(edges_x_left) > 0 and len(edges_x_right) > 0:
        #                 return
        # def scan_y():
        #     for i in range(x):
        #         for j in range(y):
        #             if binary_image[i][j] == 255:
        #                 edges_y_up.append((i,j))
        #             if binary_image[x-i-1][j] == 255:
        #                 edges_y_down.append((i,j))
        #             if len(edges_y_up) > 0 and len(edges_y_down) > 0:
        #                 return
        # scan_x()
        # scan_y()
        # edges_x_left = pd.DataFrame(edges_x_left, columns=['x','y'])
        # edges_x_left.sort_values(by='x').loc[0].values
        #
        # left = max([min(edges_x_left), min(edges_x_right)])  # 左边界
        # right = max([max(edges_x_left), max(edges_x_right)])  # 右边界
        # print(edges_x_left, edges_x_right)
        # width = right - left  # 宽度
        # bottom = max([min(edges_y_up), min(edges_y_down)])  # 底部
        # top = min([max(edges_y_up), max(edges_y_down)])  # 顶部
        # height = top - bottom  # 高度
        pre_picture = img[left:left + width, bottom:bottom + height, :]
        print('Cut the black edge!! now the size is [width: %s AND height: %s] ' % (repr(width), repr(height)))
        if keep_the_preimg:
            cv2.imwrite(os.path.join(
                self.dataset_root, self.name, 'all', self.level, 'all_dropbe.jpg'), pre_picture)
        return pre_picture



    def sep(self, meter=None, sep_ratio=None):
        """
        Cutting all.tif picture into small pieces
        :param meter: distance unit in meter
        :param sep_ratio: or use ratio to seperate the picture
        :return: output path
        """
        loc_mark = pd.DataFrame(columns=['imgfname', 'imgid', 'lat', 'lng'])
        if meter is not None:
            # using the meter to measure the interval

            lat_dis, lng_dis = self.get_bond_dis()
            self.sep_ratio = (meter / lat_dis, meter / lng_dis)
            print(lat_dis, lng_dis)

        else:
            assert sep_ratio is not None, 'A ratio need to provide'
            self.sep_ratio = sep_ratio

        assert self.sep_ratio is not None, 'Information of meter or sep_ratio need to provide'

        if get_file_size(self.amap_file, 'M') > 500:
            print('Too big figure to handle, try to consider a second way')
        else:
            img = cv2.imread(self.amap_file)
            # img = self.drop_black_edge(img)  # prepare the true image
            # img = self.crop4minima(img)
            st = (int(self.sep_ratio[0] * img.shape[0]),
                  int(self.sep_ratio[1] * img.shape[1]))
            sp = (int(img.shape[0] / st[0]), int(img.shape[1] / st[1]))
            ## update the information
            self.glob_info['step_ratio'] = self.sep_ratio
            self.glob_info['pixel_step'] = st
            self.glob_info['imgs_num'] = sp
            self.glob_info['original img_size'] = img.shape

            for i in tqdm(range(sp[0])):
                for j in range(sp[1 ]):
                    simgno = str(int(i * sp[1] + j + 1))
                    simgn = os.path.join(
                        self.sep_output_path, '0' * (10 - len(simgno)) + simgno + '.jpg')
                    # print('\r' + simgn, end='', flush=True)
                    cv2.imwrite(
                        simgn, img[i * st[0]:(i + 1) * st[0], j * st[1]:(j + 1) * st[1], :])
                    # TODO: generate the seperated piece images' annotation
                    # record the conter pixel point's gps info, and record
                    # extra information into a txt file
                    loc = self.__cal_sep_loc(img.shape, st, i, j)
                    loc_mark.loc[i*sp[1] + j] = {'imgfname': simgno, 'imgid': int(
                        simgno), 'lat':self.bonds.north - loc[0], 'lng': self.bonds.west + loc[1]}
            ## Write down the marks
            loc_mark.to_csv(os.path.join(self.sep_output_path, 'loc_mark.csv'), index=0)
            with open(os.path.join(self.sep_output_path, 'info.txt'), 'w') as f:
                f.writelines(json.dumps(self.glob_info) + '\n')


    def create_exact_mapping(self):
        loc_mark = pd.DataFrame(columns=['imgfname', 'imgid', 'lat', 'lng'])
        standard_locs = pd.DataFrame(columns=['lat', 'lng'])
        standard = pd.read_csv(self.gt_file, header=None)
        standard_locs['lat'] = 0.5 * (standard[1] + standard[3])
        standard_locs['lng'] = 0.5 * (standard[2] + standard[4])

        pos_x = 0.5 * (standard[1] + standard[3])
        pos_y = 0.5 * (standard[2] + standard[4])


        # find out the exact pixel point
        lat_diff = abs(self.bonds.north - self.bonds.south)
        lng_diff = abs(self.bonds.east - self.bonds.west)
        standard_locs.lat = standard_locs.lat.apply(lambda x: abs(x - self.bonds.north)) / lat_diff
        standard_locs.lng = standard_locs.lng.apply(lambda x: abs(x - self.bonds.west)) / lng_diff

        print(standard_locs)

        # get amap file
        if get_file_size(self.amap_file, 'M') > 500:
            print('Too big figure to handle, try to consider a second way')
        else:
            img = cv2.imread(self.amap_file)
            print(img.shape)
            height, width, _ = img.shape

            pix_xs = standard_locs.lat * height
            pix_ys = standard_locs.lng * width

            pix_xs = pix_xs.apply(int)
            pix_ys = pix_ys.apply(int)
            print(pix_xs.describe())

            loc_idx = 0
            figure_size = (180,180)
            for i in tqdm(range(len(pix_xs))):
                sz1 = pix_xs.loc[i]
                sz2 = pix_ys.loc[i]
                a = int(sz1 - int(figure_size[0] / 2))  # x start
                b = int(sz1 + int(figure_size[0] / 2))  # x end
                c = int(sz2 - int(figure_size[1] / 2))  # y start
                d = int(sz2 + int(figure_size[1] / 2))  # y end

                try:
                    assert a>0 and b<height and c>0 and d<width
                    cropimg = img[a:b, c:d]

                    simgno = str(int(i))
                    simgn = os.path.join(
                        self.sep_output_path, '0' * (10 - len(simgno)) + simgno + '.jpg')
                    cv2.imwrite(simgn, cropimg)


                    loc_idx += 1
                    loc_mark.loc[loc_idx] = {'imgfname': simgno, 'imgid': int(
                        simgno), 'lat': pos_x[i], 'lng': pos_y[i]}
                except:
                    print('imgnore: {}'.format(simgno))

            loc_mark.to_csv(os.path.join(self.sep_output_path, 'loc_mark.csv'), index=0)
            with open(os.path.join(self.sep_output_path, 'info.txt'), 'w') as f:
                f.writelines(json.dumps(self.glob_info) + '\n')





if __name__ == '__main__':

    # print(decimal2sexagesimal(107.40563201904297))
    # print(decimal2sexagesimal(34.02503967285156))
    # print(sexagesimal2decimal([41, 54, 10.28]))
    print(bondary)
    # set_bond()
    a = AMap()
    #
    # # print('维度, 经度： ', a.get_bond_dis())
    # # print(a.get_distance(34.023244,-118.260944,  34.023961,-118.260944))
    # # print(a.get_distance(34.023675, -118.260944, 34.023675, -118.261817))
    # # print(pd.read_csv(a.gt_file).head())
    #
    # a.sep(meter=200, sep_ratio=None)
    a.create_exact_mapping()
