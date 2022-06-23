# IMAGE-PROCESSING-LAB


program 1 : develop a program to display the Gray scale image using read and write  operation

import cv2<br>
img=cv2.imread('b1.jpg',0)<br>
cv2.imshow('b1',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllwindows()<br>

output:![p1 op](https://user-images.githubusercontent.com/99865210/173809761-c66e0bbd-5701-451a-8a3d-5b32f00b0b17.png)
******************************************************************************************************************************************************************


program  2: develop a program to display the image using MAtplot lib<br>

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
program 5:  Write a pgm to create image using a color from PIL import image<br>

from PIL import Image<br>
img=Image.new("RGB",(200,400),(0,0,255))<br>
img.show()<br>

output:![image](https://user-images.githubusercontent.com/99865210/173816410-cb6c56e3-bcdc-4071-a367-0630c472420f.png)<br>


***********************************************************************************************************************************************************************
program 6:  Develop a program to visulate the image using various color spaces<br>
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
program 7: Write  a program to display the image attributes <br>

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
program 8: Resize the original image<br>

import cv2<br>
img=cv2.imread('b1.jpg',0)<br>
cv2.imshow('b1',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllwindows()<br>

import matplotlib.image as mping<br>
import matplotlib.pyplot as plt<br>
img=mpimg.imread('img.jpg')<br>
plt.imshow(img)<br>


output :![8opt](https://user-images.githubusercontent.com/99865210/174047093-19c7a0a6-e5f7-4602-9844-ba870bc4fd68.png)<br>
original image length width (183, 275, 3)<br>

Resized image length width (160, 150, 3)<br>
<br>    ![8optt](https://user-images.githubusercontent.com/99865210/174047351-09465a4c-0408-438d-af55-d783426eab8b.png)<br>
 
 ******************************************************************************************************************************************************************
 
 
 program 9:Original image to gray scale  and to Binary<br>
 
import cv2<br>
#read the image file<br>
img=cv2.imread('F2.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>

#Gray scale<br>

img=cv2.imread('F2.jpg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>

#Binary image<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
CV2.waitKey(0)<br>
cv2.destroyAllwindows()<br>

output :![9opt](https://user-images.githubusercontent.com/99865210/174051559-50dec92f-cf41-474c-8d75-b0379d62b815.png)<br>
          ![9optt](https://user-images.githubusercontent.com/99865210/174051596-0a1e79a1-e1f7-4f85-8318-6202f2ccab6c.png)<br>
            ![9opttt](https://user-images.githubusercontent.com/99865210/174051691-9bbc480d-63a4-4891-a9ff-19109c4fed69.png)<br>

 
 *****************************************************************************************************************************************************************
 22/06/2022<br>
 
 program 1:To dread image using url<br><br>
        from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTvAz7oRpSnwGnBO2p64jeZKA6b0ULoNEII0w&usqp=CAU.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

output:![image](https://user-images.githubusercontent.com/99865210/175022163-97989496-a44d-47bc-9625-7d4becaf0a29.png)<br>

*****************************************
program 3: To perform arthemetic operations
import cv2
import matplotlib.image as mping
import matplotlib.pyplot as plt
img1=cv2.imread('flower2.jpg')
img2=cv2.imread('flower3.jpg')
fimg1=img1+img2
plt.imshow(fimg1)
plt.show()
cv2.imwrite('output.jpg',fimg1)
fimg2=img1-img2
plt.imshow(fimg2)
plt.show()
cv2.imwrite('output.jpg',fimg2)
fimg3=img1*img2
plt.imshow(fimg3)
plt.show()
cv2.imwrite('output.jpg',fimg3)
fimg4=img1/img2
plt.imshow(fimg4)
plt.show()
cv2.imwrite('output.jpg',fimg4)

output:![image](https://user-images.githubusercontent.com/99865210/175286745-cd75e48b-e145-46ee-95e8-45da54aaf083.png)

![image](https://user-images.githubusercontent.com/99865210/175287145-050e7629-d28d-4122-8487-71351e1c024c.png)

![image](https://user-images.githubusercontent.com/99865210/175287258-1721da91-bf87-4048-8144-879f34e2e9d7.png)

****************************************************************************************************************************************************

program 2:mask and blurr


import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img=mpimg.imread("flower3.jpg")
plt.imshow(img)
plt.show()
output:
![image](https://user-images.githubusercontent.com/99865210/175288164-eb54126e-8d48-40a2-84ab-49732c46e033.png)
*****************
hsv_img=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
light_orange=(1,190,200)
dark_orange=(18,255,255)
mask=cv2.inRange(hsv_img,light_orange,dark_orange)
result=cv2.bitwise_and(img,img,mask=mask) 
plt.subplot(1,2,1) 
plt.imshow(mask,cmap='gray') 
plt.subplot(1,2,2)
plt.imshow(result)
plt.show()
 output:
 


![image](https://user-images.githubusercontent.com/99865210/175288488-b88bba43-c045-41b5-843d-3657910c7df7.png)

![image](https://user-images.githubusercontent.com/99865210/175288507-d8a4faaf-f3dd-41da-9c9f-30e116c25e21.png)



*********************************************************************************************************************************************************

light_white=(0,0,200) 
dark_white=(145,60,255) 
mask_white=cv2.inRange(hsv_img,light_white,dark_white)
result_white=cv2.bitwise_and(img,img,mask=mask_white) 
plt.subplot(1,2,1)
plt.imshow(mask_white,cmap='gray') 
plt.subplot(1,2,2)
plt.imshow(result_white) 
plt.show()
output:![image](https://user-images.githubusercontent.com/99865210/175288679-5a29e88c-c407-4527-97a4-04f2a7b96ee3.png)
![image](https://user-images.githubusercontent.com/99865210/175288725-7d085d65-c1df-429c-ba28-0db33d724e38.png)
********************************************************************************************************************************************
final_mask=mask+mask_white
final_result=cv2.bitwise_and(img,img,mask=final_mask)
plt.subplot(1,2,1) 
plt.imshow(final_mask,cmap="gray")
plt.subplot(1,2,2) 
plt.imshow(final_result) 
plt.show() 
output:![image](https://user-images.githubusercontent.com/99865210/175288847-a89c5c65-0cc0-4db4-8d27-38535c6b3509.png)
![image](https://user-images.githubusercontent.com/99865210/175288874-0adae1f4-54f6-4622-b3f7-1dbedd72eb3f.png)
*************************************
blur=cv2.GaussianBlur(final_result,(7,7),0) 
plt.imshow(blur) 
plt.show()
output:![image](https://user-images.githubusercontent.com/99865210/175288992-3539dab6-8f9f-456d-baeb-accada3e9443.png)

**********************************************************************************************************************************************************
import cv2 
img=cv2.imread('D:\\rabbit.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow("GRAY image",gray)
cv2.imshow("HSV image",hsv)
cv2.imshow("LAB image",lab)
cv2.imshow("HLS image",hls)
cv2.imshow("YUV image",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()

output:




