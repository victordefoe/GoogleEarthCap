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

mouse = pynput.mouse.Controller()
keyboard = pynput.keyboard.Controller()


pwd = os.getcwd()


def move_scene(motion='left'):
    init_pos = (-1420, 500)
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

    for i in range(1):
        keyboard.press(act[motion])
        time.sleep(0.05)
        keyboard.release(act[motion])


def crab_location(filename, mouse_pos ,high_stage=False):

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
    saveBitMap.SaveBitmapFile(saveDC, filename)

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
    init_crab = (-w + 500, 500)

    cap_w, cap_h = 500, 500
    end_crab = (-w + 500 + cap_w, 500 + cap_h)
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
    pattern = re.compile('[0-9]+')
    coors = pattern.findall(raw_string)
    print(coors)
    return coors



def main():
    beg = time.time()

    # 截图
    for i in range(10):
        saved_imgfilename = 'test.jpg'
        init_crab, end_crab = window_capture(saved_imgfilename)


        img = crab_location('loc.jpg', mouse_pos=init_crab)
        ocr(img)
        img = crab_location('loc.jpg', mouse_pos=end_crab)
        ocr(img)


        move_scene()

    end = time.time()
    print(end - beg)


main()

# move_scene()
