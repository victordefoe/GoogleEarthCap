# GooleEarth
capture the dataset from goole earth

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
本代码运行环境
需要安装tesseract 4.0.0; pytesseract \
系统：Windows 7; \
pandas

### 附注
Pytesseract is the wrapper of tesseract-ocr development tool. 
直接的tesseract API会比使用pyteseract快一些些

chi_sim的识别速度会慢很多，但是识别数字比较准确。

