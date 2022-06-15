# IMAGE-PROCESSING-LAB


program 1 : develpo a program to display the Gray scale image using read and write  operation

import cv2
img=cv2.imread('b1.jpg',0)
cv2.imshow('b1',img)
cv2.waitKey(0)
cv2.destroyAllwindows()

output:![p1 op](https://user-images.githubusercontent.com/99865210/173809761-c66e0bbd-5701-451a-8a3d-5b32f00b0b17.png)



program  2: develop a program to display the image using MAtplot lib


import matplotlib.image as mping
import matplotlib.pyplot as plt
img=mpimg.imread('img.jpg')
plt.imshow(img)

output:![image](https://user-images.githubusercontent.com/99865210/173810407-90c87e31-b982-447f-a650-7f18da4ece6c.png)

program 3 :Develpo a program to perform a linear transformation rotation

from PIL import Image
img =Image.open("L1.jpg")
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.distroyAllwindows()

output:

program 4: Develpo a program to convert color to RGB color values
