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


def set_bond():
    Bondary = collections.namedtuple('Bondary', 'west, east, north, south')
    # bondary = Bondary(118.3247362807, 118.0864515186, 34.0708663191, 33.9221215026)
    dataset_name = 'collection_1'
    dataset_path = r'Z:\research\datasets\GoogleEarth'
    dataset = os.path.join(dataset_path, dataset_name)
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


if __name__ == '__main__':

    print(decimal2sexagesimal(118.3247362807))
    print(sexagesimal2decimal([41, 54, 10.28]))
    # print(bondary.right)
    set_bond()
