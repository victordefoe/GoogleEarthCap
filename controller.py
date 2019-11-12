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
from pynput.mouse import Button, Controller
import time
import random
from pynput import mouse as ms
import matplotlib.pyplot as plt

mouse = Controller()

# set pointer positon
# mouse.position = (277, 645)
# print('now we have moved it to {0}'.format(mouse.position))
#
# #鼠标移动（x,y）个距离
# mouse.move(5, -5)
# print(mouse.position)
#
# mouse.press(Button.left)
# mouse.release(Button.left)
#
# #Double click
# mouse.click(Button.left, 1)
#
# #scroll two  steps down
# mouse.scroll(0, 500)


#
# def on_move(x, y ):
#     print('Pointer moved to ',x,y)
#
# def on_click(x, y , button, pressed):
#     print('{0} at {1}'.format('Pressed' if pressed else 'Released', (x, y)))
#     if not pressed:
#         return False
#
# def on_scroll(x, y ,dx, dy):
#     print('scrolled {0} at {1}'.format(
#         'down' if dy < 0 else 'up',
#         (x, y)))
#
# while True:
#     with ms.Listener(on_move = on_move,on_click = on_click,on_scroll = on_scroll) as listener:
#         listener.join()

# ---------------

pwd = os.getcwd()


def move_scene(motion='left'):
    act = {'left': (-63, 190)}
    # print(act[motion][0])

    mouse.position = act[motion]

    # mouse.click(Button.left, 1)
    mouse.press(Button.left)
    time.sleep(round(random.uniform(0.5, 1.0), 10))
    mouse.release(Button.left)


def crab_location(filename='loc.jpg'):

    low_stage = (-681, 1057)
    high_stage = (-721, 1057)

    mfcDC = win32ui.CreateDCFromHandle(
        win32gui.GetWindowDC(
            win32gui.GetDesktopWindow()))
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()

    cap_w, cap_h = 310, 25
    saveBitMap.CreateCompatibleBitmap(mfcDC, cap_w, cap_h)
    saveDC.SelectObject(saveBitMap)
    saveDC.BitBlt((0, 0), (cap_w, cap_h), mfcDC,
                  high_stage, win32con.SRCCOPY)
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
    # print w,h　　　#图片大小
    # 为bitmap开辟空间
    saveBitMap.CreateCompatibleBitmap(mfcDC, cap_w, cap_h)
    # 高度saveDC，将截图保存到saveBitmap中
    saveDC.SelectObject(saveBitMap)
    # 截取从左上角（0，0）长宽为（w，h）的图片
    saveDC.BitBlt((0, 0), (cap_w, cap_h), mfcDC, init_crab, win32con.SRCCOPY)
    saveBitMap.SaveBitmapFile(saveDC, filename)
    return init_crab


beg = time.time()

# 截图
init_crab = window_capture("haha.jpg")
mouse.position = init_crab
time.sleep(round(random.uniform(0.5, 1.0), 10))
print(init_crab)
img = crab_location('loc.jpg')


# img = torch.Tensor(img)


# ----  tring thirdparty chinese_ocr app , but it fails ----
# from thirdparty.chineseocr_app.crnn.network_torch import CRNN
# from thirdparty.chineseocr_app.crnn.keys import alphabetChinese, alphabetEnglish
# ocrModelWeight = os.path.join(pwd, "thirdparty","chineseocr_app", "models", "ocr-lstm.pth")
# alphabet = alphabetChinese
# nclass = len(alphabet)+1
# LSTMFLAG = True
# GPU = False
# OCRMODEL = CRNN( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=LSTMFLAG,GPU=GPU,alphabet=alphabet)
# OCRMODEL.load_weights(ocrModelWeight)


##

#
# class parmeters():
#     def __init__(self):
#         self.image_folder = 1
#         self.workers = 2
#         self.batch_size = 1
#         self.saved_model = 'D://UAV_location//google_earth//GooleEarth//thirdparty' \
#                            '//deep_tr//pretrained_models//TPS-ResNet-BiLSTM-Attn.pth'
#         self.batch_max_length = 25
#         self.imgH = 32
#         self.imgW = 100
#         self.rgb = False
#         self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
#         self.sensitive = True
#         self.PAD = True
#         self.Transformation = 'TPS'
#         self.FeatureExtraction = 'ResNet'
#         self.SequenceModeling = 'BiLSTM'
#         self.Prediction = 'Attn'
#         self.num_fiducial = 20
#         self.input_channel = 1
#         self.output_channel = 512
#         self.hidden_size = 256
#
#
# opt = parmeters()
# print(opt.FeatureExtraction)
#
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# converter = AttnLabelConverter(opt.character)
# opt.num_class = len(converter.character)
# if opt.rgb:
#     opt.input_channel = 3
# model = Model(opt).to(device)
# print(
#     'model input parameters',
#     opt.imgH,
#     opt.imgW,
#     opt.num_fiducial,
#     opt.input_channel,
#     opt.output_channel,
#     opt.hidden_size,
#     opt.num_class,
#     opt.batch_max_length,
#     opt.Transformation,
#     opt.FeatureExtraction,
#     opt.SequenceModeling,
#     opt.Prediction)
#
# # model = torch.nn.DataParallel(model).to(device)
# # load model
#
#
# print('loading pretrained model from %s' % opt.saved_model)
# state_dict = torch.load(opt.saved_model)
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# model.load_state_dict(new_state_dict)
#
# ###
#
# from thirdparty.deep_tr.dataset import ResizeNormalize
# transform = ResizeNormalize((opt.imgW, opt.imgH))
# image_tensor = transform(img).unsqueeze(0)
# ###
#
# # with torchsnooper.snoop():
# # image_tensor = img
# print('..........',image_tensor.size())
# image = image_tensor.to(device)
# text_for_pred = torch.LongTensor(
#     opt.batch_size,
#     opt.batch_max_length +
#     1).fill_(0).to(device)
#
#
# preds = model(image, text_for_pred, is_train=False)
# _, preds_index = preds.max(2)
# length_for_pred = torch.IntTensor([opt.batch_max_length] * opt.batch_size).to(device)
# preds_str = converter.decode(preds_index, length_for_pred)
#
# print(preds_str)



print(tesserocr.tesseract_version())  # print tesseract-ocr version
# prints tessdata path and list of available languages
print(tesserocr.get_languages())

# with PyTessBaseAPI() as api:
#     api.SetImageFile('loc.jpg')
#     print(api.GetUTF8Text())
#     print(api.AllWordConfidences())

# print(tesserocr.file_to_text('loc.jpg', lang='Armenian', psm=7 ))
def ocr(img):

    pytesseract.pytesseract.tesseract_cmd = r"F://Program Files (x86)//Tesseract-OCR//tesseract.exe"
    a = pytesseract.image_to_string(
        img,
        lang='chi_sim',
        config='--tessdata-dir "F://Program Files (x86)//Tesseract-OCR//tessdata" digits')
    print(a)


ocr(img)
end = time.time()
print(end - beg)

# move_scene()
