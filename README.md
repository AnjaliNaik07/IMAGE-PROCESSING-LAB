# IMAGE-PROCESSING-LAB


program 1 : develpo a program to display the Gray scale image using read and write  operation

import cv2
img=cv2.imread('b1.jpg',0)
cv2.imshow('b1',img)
cv2.waitKey(0)
cv2.destroyAllwindows()

output:![p1 op](https://user-images.githubusercontent.com/99865210/173809761-c66e0bbd-5701-451a-8a3d-5b32f00b0b17.png)
******************************************************************************************************************************************************************


program  2: develop a program to display the image using MAtplot lib


import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mpimg.imread('img.jpg')
plt.imshow(img)

output:![image](https://user-images.githubusercontent.com/99865210/173810407-90c87e31-b982-447f-a650-7f18da4ece6c.png)
********************************************************************************************************************************************************************
program 3 :Develpo a program to perform a linear transformation rotation

from PIL import Image
img =Image.open("L1.jpg")
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.distroyAllwindows()

output:![image](https://user-images.githubusercontent.com/99865210/173816106-1bd26923-b5e7-4f13-86d6-0f058d75ccbb.png)

**********************************************************************************************************************************************************************
program 4: Develpo a program to convert color to RGB color values

from PIL import ImageColor
img1=ImageColor.getrgb("yellow")
print(img1)
img2=ImageColor.getrgb("red")
print(img2)
img3=ImageColor.getrgb("pink")
print(img3)
img4=ImageColor.getrgb("blue")
print(img4)

output:(255, 255, 0)
(255, 0, 0)
(255, 192, 203)
(0, 0, 255)
***********************************************************************************************************************************************************************
program 5: Write a pgm to create image using a color from PIL import image

from PIL import Image
img=Image.new("RGB",(200,400),(0,0,255))
img.show()

output:![image](https://user-images.githubusercontent.com/99865210/173816410-cb6c56e3-bcdc-4071-a367-0630c472420f.png)


*******************************************************************************************************************************************************************
program 6: Develop a program to visulate the image using various color spaces

import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('p2.jpg')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
plt.show()

output: ![image](https://user-images.githubusercontent.com/99865210/173813702-ca6d3deb-ab58-46c6-8c68-d7f03b330870.png)

output:  ![image](https://user-images.githubusercontent.com/99865210/173816669-5892b36c-39c6-41b3-9348-20722de0186d.png)

output:    ![image](https://user-images.githubusercontent.com/99865210/173816796-8dff3cb1-7126-4eab-bebf-ce0ec3d3d292.png)

***********************************************************************************************************************************************************************
program 7:Write  a program to display the image attributes 

from PIL import Image
image=Image.open('p1.jpg')
print("Filename:",image.filename)
print("Format:",image.format)
print("Mode:",image.mode)
print("Size:",image.size)
print("Width:",image.width)
print("Height",image.height)
image.close();

output:Filename: p1.jpg
Format: JPEG
Mode: RGB
Size: (275, 183)
Width: 275
Height 183

*******************************************************************************************************************************************************************
