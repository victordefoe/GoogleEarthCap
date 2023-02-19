
> this repo is archived

# GoogleEarthCap

capture the dataset from Goole Earth software

## principle
1. Screenshots: The main problem is the latitude and longitude coordinates. To confirm the screenshot of each frame, you need to know the latitude and longitude information corresponding to the coordinates of the four points in this area.

2. Accurately find the corresponding point on the global area image

These are two separate steps

For 1:
  In the low-altitude area, move along the designed route and traversal method, take a screenshot, and then use OCR to identify the coordinates

For 2:
  Get panoramas in high altitude areas
 
In this repo mainly involves step 1.

### requirements
System: Windows 7(I didn't try in another system); \
The running environment of this code requires the installation of tesseract 4.0.0;\
pytesseract \
pandas

### Notes
Pytesseract is the wrapper of tesseract-ocr development tool.
Direct tesseract API will be faster than using pyteseract

The recognition speed of chi_sim will be much slower, but the recognition number is more accurate.

## 原理
1. 截图：主要问题是经纬度坐标，区域的标注确认
每一帧截图，需要知道这一块区域的四个点的坐标对应的经纬度信息。

2. 在全局区域图像上面精准找到对应的点

这是两个分别的步骤

For 1:
 在低空区域沿着设计的路线和遍历方式，移动、截图，然后用ocr识别坐标

For 2:
 在高空区域获取全景图
 
在本工程中主要涉及步骤1.

### requirements
系统：Windows 7; 我只在该系统下试过 \
本代码运行环境需要安装tesseract 4.0.0; \
pytesseract \
pandas

### 附注
Pytesseract is the wrapper of tesseract-ocr development tool. 
直接的tesseract API会比使用pyteseract快一些些

chi_sim的识别速度会慢很多，但是识别数字比较准确。

