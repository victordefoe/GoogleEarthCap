# encoding: utf-8
'''
@Author: 刘琛
@Time: 2019/11/8 17:25
@Contact: victordefoe88@gmail.com

@File: controller.py
@Statement:


'''

from  pynput.mouse import Button, Controller
import time
import random
from pynput import mouse as ms

mouse = Controller()

#set pointer positon
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

## ---------------

import time
import win32gui, win32ui, win32con, win32api
import PIL, cv2
import numpy as np
from PIL import Image


def move_scene(motion='left'):
    act = {'left':(-63,190)}
    # print(act[motion][0])

    mouse.position = act[motion]


    # mouse.click(Button.left, 1)
    mouse.press(Button.left)
    time.sleep(round(random.uniform(0.5, 1.0), 10))
    mouse.release(Button.left)


def crab_location(filename = 'loc.jpg'):

  mfcDC = win32ui.CreateDCFromHandle(win32gui.GetWindowDC(win32gui.GetDesktopWindow()))
  saveDC = mfcDC.CreateCompatibleDC()
  saveBitMap = win32ui.CreateBitmap()

  cap_w, cap_h = 310,25
  saveBitMap.CreateCompatibleBitmap(mfcDC, cap_w, cap_h)
  saveDC.SelectObject(saveBitMap)
  saveDC.BitBlt((0, 0), (cap_w, cap_h), mfcDC, (-681,1057), win32con.SRCCOPY)
  saveBitMap.SaveBitmapFile(saveDC, filename)

  bmparray = np.asarray(saveBitMap.GetBitmapBits(), dtype=np.uint8)
  bmpinfo = saveBitMap.GetInfo()
  pil_im = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmparray, 'raw', 'BGRX', 0, 1)
  pil_array = np.array(pil_im)
  cv_im = cv2.cvtColor(pil_array, cv2.COLOR_RGB2BGR)




def window_capture(filename,windowname='Google Earth Pro'):

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
  init_crab = (-w+500, 500)

  cap_w, cap_h = 500,500
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

## 截图
# init_crab = window_capture("haha.jpg")
# mouse.position = init_crab
# time.sleep(round(random.uniform(0.5, 1.0), 10))
# print(init_crab)
# crab_location('loc.jpg')
from thirdparty.chineseocr_app.crnn.network_dnn import CRNN
from thirdparty.chineseocr_app.crnn.keys import alphabetChinese, alphabetEnglish
ocrModelOpencv = os.path.join(pwd, "thirdparty","chineseocr_app","models", "ocr.pb")
CRNN()

end = time.time()
print(end - beg)

# move_scene()



