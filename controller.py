# encoding: utf-8
'''
@Author: 刘琛
@Time: 2019/11/8 17:25
@Contact: victordefoe88@gmail.com

@File: controller.py
@Statement:


'''

import tesserocr
from tesserocr import PyTessBaseAPI
import pytesseract
from thirdparty.deep_tr.utils import AttnLabelConverter
from thirdparty.deep_tr.model import Model
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import collections
from collections import OrderedDict
import torchsnooper

import win32api
import win32con
import os
import cv2
import win32ui
import PIL
import win32gui

import torch
from PIL import Image
import numpy as np
from pynput.mouse import Button
from pynput.keyboard import Key
import pynput
import time
import random
import matplotlib.pyplot as plt
import re
import pandas as pd

import tesserocr
from tesserocr import PyTessBaseAPI

from utils import bondary, sexagesimal2decimal

mouse = pynput.mouse.Controller()
keyboard = pynput.keyboard.Controller()


pwd = os.getcwd()
temp_table = pd.DataFrame(
    columns=[
        'filename',
        'init_coor_N',
        'init_coor_E',
        'end_coor_N',
        'end_coor_E'])

# 东西方速度，南北方向速度，正，则向东向北；数值控制移动跨步速度。
# 初速度为 向北速度为1
direction = {'EW': 0, 'NS': 1}
eop = False


def move_scene(direction):
    init_pos = (-1420, 3)
    if mouse.position[0] >= 0:
        mouse.position = init_pos
        mouse.click(Button.left, 1)
    else:
        mouse.click(Button.left, 1)
    act = {
        'left': Key.left,
        'right': Key.right,
        'up': Key.up,
        'down': Key.down}

    if direction['EW'] != 0:
        if direction['EW'] > 0:
            keyboard.press(act['right'])
            time.sleep(0.05 * abs(direction['EW']))
            keyboard.release(act['right'])
        elif direction['Ew'] < 0:
            keyboard.press(act['left'])
            time.sleep(0.05 * abs(direction['EW']))
            keyboard.release(act['left'])

    if direction['NS'] > 0:
        keyboard.press(act['up'])
        time.sleep(0.05 * abs(direction['NS']))
        keyboard.release(act['up'])
    elif direction['NS'] < 0:
        keyboard.press(act['down'])
        time.sleep(0.05 * abs(direction['NS']))
        keyboard.release(act['down'])


def crab_location(filename, mouse_pos, high_stage=False):

    mouse.position = mouse_pos
    # time.sleep(round(random.uniform(0.5, 1.0), 10))
    # print(init_crab)

    start_pos = (-721, 1057) if high_stage else (-681, 1057)

    mfcDC = win32ui.CreateDCFromHandle(
        win32gui.GetWindowDC(
            win32gui.GetDesktopWindow()))
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()

    cap_w, cap_h = 310, 25
    saveBitMap.CreateCompatibleBitmap(mfcDC, cap_w, cap_h)
    saveDC.SelectObject(saveBitMap)
    saveDC.BitBlt((0, 0), (cap_w, cap_h), mfcDC,
                  start_pos, win32con.SRCCOPY)
    # saveBitMap.SaveBitmapFile(saveDC, filename)

    bmparray = np.asarray(saveBitMap.GetBitmapBits(), dtype=np.uint8)
    bmpinfo = saveBitMap.GetInfo()
    pil_im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'],
         bmpinfo['bmHeight']),
        bmparray,
        'raw',
        'BGRX',
        0,
        1).convert('L')
    pil_im = pil_im.point(lambda x: 255 if x > 240 else 0).convert('1')
    pil_im.save(filename, quality=95)
    # pil_array = np.array(pil_im)
    # cv_im = cv2.cvtColor(pil_array, cv2.COLOR_RGB2BGR)
    # cv_im = cv2.cvtColor(cv_im, cv2.COLOR_BGR2GRAY)
    # ret, cv_th = cv2.threshold(cv_im, 240, 255, cv2.THRESH_BINARY)
    # # plt.imshow(cv_th, 'gray')
    # # plt.show()
    # cv2.imwrite(filename, cv_th)

    return pil_im


def window_capture(filename, windowname='Google Earth Pro'):

    # hwnd = win32gui.FindWindow(None, windowname) # 窗口的编号，0号表示当前活跃窗口

    hwnd = win32gui.GetDesktopWindow()
    # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    hwndDC = win32gui.GetWindowDC(hwnd)
    # 根据窗口的DC获取mfcDC
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    # mfcDC创建可兼容的DC
    saveDC = mfcDC.CreateCompatibleDC()
    # 创建bigmap准备保存图片
    saveBitMap = win32ui.CreateBitmap()
    # 获取监控器信息
    MoniterDev = win32api.EnumDisplayMonitors(None, None)
    # print(MoniterDev[1])
    print(hwnd)
    w = MoniterDev[0][2][2]
    h = MoniterDev[0][2][3]
    init_crab = (-w + 500, 400)

    cap_w, cap_h = 500, 500
    end_crab = (init_crab[0] + cap_w, init_crab[1] + cap_h)
    # print w,h　　　#图片大小
    # 为bitmap开辟空间
    saveBitMap.CreateCompatibleBitmap(mfcDC, cap_w, cap_h)
    # 高度saveDC，将截图保存到saveBitmap中
    saveDC.SelectObject(saveBitMap)
    # 截取从左上角（0，0）长宽为（w，h）的图片
    saveDC.BitBlt((0, 0), (cap_w, cap_h), mfcDC, init_crab, win32con.SRCCOPY)
    saveBitMap.SaveBitmapFile(saveDC, filename)
    return init_crab, end_crab


def ocr(img):

    pytesseract.pytesseract.tesseract_cmd = r"F://Program Files (x86)//Tesseract-OCR//tesseract.exe"
    raw_string = pytesseract.image_to_string(
        img,
        lang='chi_sim',
        config='--tessdata-dir "F://Program Files (x86)//Tesseract-OCR//tessdata" digits')

    # raw_string = tesserocr.image_to_text(img, lang='chi_sim', psm=7)
    print(raw_string)
    pattern = re.compile(r'[0-9]+')
    coors = pattern.findall(raw_string)
    coors = [float(x) for x in coors]
    coors = [coors[0], coors[1], coors[2] + coors[3]*0.01, coors[4], coors[5], coors[6]+coors[7]*0.01]
    notation = re.compile(r'[\u4e00-\u9fa5]+').findall(raw_string)
    print(coors, notation)
    sign_dict = {'西': -1, '东': 1, '南': -1, '北': 1}
    sign = (sign_dict[notation[0]], sign_dict[notation[1]])

    return coors, sign


def check(img_filename, gt_file, coors1, coors1sign, coors2, coors2sign, saved_flag=False):
    # 这个函数 一是合并转化ocr读取的信息成为标准的数值，二是根据这个数值获取下一步移动的指令
    # 向北向东为正
    assert len(coors1) == len(coors2) == 6, 'Error: OCR result error'

    y_a, x_a = sexagesimal2decimal(
        coors1[0:3]), sexagesimal2decimal(coors1[3:6])
    y_a *= coors1sign[0]
    x_a *= coors1sign[1]
    y_b, x_b = sexagesimal2decimal(
        coors2[0:3]), sexagesimal2decimal(coors2[3:6])
    y_b *= coors2sign[0]
    x_b *= coors2sign[1]
    print('当前位置坐标', x_a, y_a)

    # 写入缓变量 保存当前图像的 文件名和对应坐标位置
    global temp_table, eop
    temp_table = temp_table.append({
        'filename': img_filename,
        'init_coor_N': y_a,
        'init_coor_E': x_a,
        'end_coor_N': y_b,
        'end_coor_E': x_b}, ignore_index=True)

    # 设定飞行规则：分析当前位置
    # 撞南或者北墙，则向东移动一步并折返
    if y_a > bondary.north or y_b < bondary.south:
        direction['NS'] *= -1
        move_scene({'NS': direction['NS'] * 3, 'EW': 10})  # 右下角回弹

    if x_b >= bondary.east:
        eop = True

    move_scene(direction)

    if saved_flag:
        temp_table.to_csv(gt_file, index=False, mode='a', header=False)
        # clean the temp_variable
        temp_table = pd.DataFrame(
            columns=[
                'filename',
                'init_coor_N',
                'init_coor_E',
                'end_coor_N',
                'end_coor_E'])



def main():
    beg = time.time()
    save_interval = 50  # Save the csv files for every # times ops
    # 截图
    for i in range(10000):
        try:
            saved_imgfilename = '%d.bmp' % i
            saved_imgdir = r'Z:\research\datasets\GoogleEarth\collection_1\patch'
            gt_file = os.path.join(saved_imgdir, 'gt.csv')
            img_filepath = os.path.join(saved_imgdir, saved_imgfilename)
            init_crab, end_crab = window_capture(img_filepath)

            img = crab_location('loc.jpg', mouse_pos=init_crab)
            initcoors, initsign = ocr(img)
            img = crab_location('loc.jpg', mouse_pos=end_crab)
            endcoors, endsign = ocr(img)

            check(
                saved_imgfilename,
                gt_file,
                initcoors,
                initsign,
                endcoors,
                endsign,
                i % save_interval == 0)

            ## End of Program flag
            if eop is True:
                break
        except Exception as e:
            print(e)

    end = time.time()
    print(end - beg)


main()

# move_scene()
