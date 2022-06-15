# IMAGE-PROCESSING-LAB


program 1 : develpo a program to display the Gray scale image using read and write  operation

import cv2<br><br>
img=cv2.imread('b1.jpg',0)<br>
cv2.imshow('b1',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllwindows()<br>

output:![p1 op](https://user-images.githubusercontent.com/99865210/173809761-c66e0bbd-5701-451a-8a3d-5b32f00b0b17.png)
******************************************************************************************************************************************************************


program  2: develop a program to display the image using MAtplot lib<br>

<br>
import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mpimg.imread('img.jpg')<br>
plt.imshow(img)<br>

output:![image](https://user-images.githubusercontent.com/99865210/173810407-90c87e31-b982-447f-a650-7f18da4ece6c.png)<br>
********************************************************************************************************************************************************************
program 3 :Develpo a program to perform a linear transformation rotation<br>

from PIL import Image<br>
img =Image.open("L1.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.distroyAllwindows()<br>

output:![image](https://user-images.githubusercontent.com/99865210/173816106-1bd26923-b5e7-4f13-86d6-0f058d75ccbb.png)<br>

**********************************************************************************************************************************************************************
program 4: Develpo a program to convert color to RGB color values<br>

from PIL import ImageColor<br>
img1=ImageColor.getrgb("yel<br><br>low")<br>
print(img1)<br><br>
img2=ImageColor.getrgb("red")<br><br>
print(img2)<br><br>
img3=ImageColor.getrgb("pink")<br><br>
print(img3)<br><br>
img4=ImageColor.getrgb("blue")<br><br>
print(img4)<br><br>

output:(255, 255, 0)<br>
(255, 0, 0)<br>
(255, 192, 203)<br>
(0, 0, 255)<br>
***********************************************************************************************************************************************************************
program 5: Write a pgm to create image using a color from PIL import image<br>

from PIL import Image<br>
img=Image.new("RGB",(200,400),(0,0,255))<br>
img.show()<br>

output:![image](https://user-images.githubusercontent.com/99865210/173816410-cb6c56e3-bcdc-4071-a367-0630c472420f.png)<br>


*******************************************************************************************************************************************************************
program 6: Develop a program to visulate the image using various color spaces<br>
<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('p2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
plt.show()<br>

output: ![image](https://user-images.githubusercontent.com/99865210/173813702-ca6d3deb-ab58-46c6-8c68-d7f03b330870.png)<br>

output:  ![image](https://user-images.githubusercontent.com/99865210/173816669-5892b36c-39c6-41b3-9348-20722de0186d.png)<br>

output:    ![image](https://user-images.githubusercontent.com/99865210/173816796-8dff3cb1-7126-4eab-bebf-ce0ec3d3d292.png)<br><br>

***********************************************************************************************************************************************************************
program 7:Write  a program to display the image attributes <br>

from PIL import Image<br>
image=Image.open('p1.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height",image.height)<br>
image.close();<br>

output:Filename: p1.jpg<br>
Format: JPEG<br>
Mode: RGB<br>
Size: (275, 183)<br>
Width: 275<br>
Height 183<br>

*******************************************************************************************************************************************************************
