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
    bondary = Bondary(
        info.Longitude.min(),
        info.Longitude.max(),
        info.Latitude.max(),
        info.Latitude.min())
    return bondary


bondary = set_bond()


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
            self.dataset_root, self.name, 'all', self.level, 'all.tif')

        self.sep_output_path = os.path.join(
            self.dataset_root, self.name, 'seps', self.level)

        if not os.path.exists(self.sep_output_path):
            os.makedirs(self.sep_output_path)

        self.sep_ratio = None

    def get_bonds(self):
        """
        wrapper of returning bondaries func
        :return: set_bond
        """
        return set_bond(self.dataset_conf)

    def get_distance(self, lat1, lng1, lat2, lng2):
        EARTH_RADIUS = 6378.137

        def rad(d):
            """
            Transfer the latitude and longitude distance to radius
            :param d: distance
            :return: radius
            """
            return d * math.pi / 180.0

        radLat1 = rad(lat1)
        radLat2 = rad(lat2)
        a = radLat1 - radLat2
        b = rad(lng1) - rad(lng2)

        s = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) +
                                    math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b / 2), 2)))
        s = s * EARTH_RADIUS
        s = (s * 10000) / 10
        return s

    def sep(self, meter=None, sep_ratio=None):
        bonds = self.get_bonds()

        if meter is not None:
            # using the meter to measure the interval
            lng_diff = abs(bonds.west - bonds.east)
            lat_diff = abs(bonds.north - bonds.south)
            lat_dis = self.get_distance(
                bonds.west, bonds.north, bonds.east, bonds.north)
            lng_dis = self.get_distance(
                bonds.west, bonds.north, bonds.west, bonds.south)
            self.sep_ratio = (meter / lat_dis, meter / lng_dis)
            print(lat_dis, lng_dis)
            print(bonds.west - nor)

        else:
            assert sep_ratio is not None, 'A ratio need to provide'
            self.sep_ratio = sep_ratio

        assert self.sep_ratio is not None, 'Information of meter or sep_ratio need to provide'

        if get_file_size(self.amap_file, 'M') > 500:
            print('Too big figure to handle, try to consider a second way')
        else:
            img = cv2.imread(self.amap_file)
            st = (int(self.sep_ratio[0] * img.shape[0]),
                  int(self.sep_ratio[1] * img.shape[1]))
            sp = (int(img.shape[0] / st[0]), int(img.shape[1] / st[1]))
            print(
                'step_ratio:%s' % repr(self.sep_ratio),
                'pixel_step%s' % repr(st),
                'imgs_num %s' % repr(sp),
                'origninal image size %s' % repr(img.shape), sep='\n', end='')
            for i in range(sp[0]):
                for j in range(sp[1]):
                    simgno = str(int(i * sp[1] + j + 1))
                    simgn = os.path.join(
                        self.sep_output_path, '0' * (10 - len(simgno)) + simgno + '.jpg')
                    print('\r' + simgn, end='')
                    cv2.imwrite(
                        simgn, img[i * st[0]:(i + 1) * st[0], j * st[1]:(j + 1) * st[1], :])


# def crop(img, fn):
#     fn = fn.split('.')[0]
#     # img = np.array(img)
#     # 切割图像大小
#     sl = 2**(int(fn[-2:])-9)
#     sp = (int(img.shape[0]/sl), int(img.shape[1]/sl))
#     print(sp)
#     if not os.path.exists(fn):
#         os.makedirs(fn)
#     for i in range(sp[0]):
#         for j in range(sp[1]):
#             simgno = str(int(i * sp[1] + j + 1))
#             simgn = os.path.join(fn, '0'*(10-len(simgno)) + simgno+ '.jpg')
#             print('\r'+simgn, end='')
#             imsave(simgn,img[i*sl:(i+1)*sl,j*sl:(j+1)*sl,:])


if __name__ == '__main__':

    # print(decimal2sexagesimal(107.40563201904297))
    # print(decimal2sexagesimal(34.02503967285156))
    # print(sexagesimal2decimal([41, 54, 10.28]))
    print(bondary)
    # set_bond()
    a = AMap()
    a.sep(meter=None, sep_ratio=(0.005,0.005))
