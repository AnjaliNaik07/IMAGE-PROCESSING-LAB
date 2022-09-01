# IMAGE-PROCESSING-LAB


program 1 : develop a program to display the Gray scale image using read and write  operation
import cv2<br>
img=cv2.imread('b1.jpg',0)<br>
cv2.imshow('b1',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllwindows()<br>
output:![p1 op](https://user-images.githubusercontent.com/99865210/173809761-c66e0bbd-5701-451a-8a3d-5b32f00b0b17.png)
*****************************************************************************************************************************************************************
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
*********************************************************************************************************************************************************************
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
******************************************************************************************************************************************************************
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
![8optt](https://user-images.githubusercontent.com/99865210/174047351-09465a4c-0408-438d-af55-d783426eab8b.png)<br>
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
 program 10:To dread image using url<br><br>
 from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTvAz7oRpSnwGnBO2p64jeZKA6b0ULoNEII0w&usqp=CAU.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>
output:![image](https://user-images.githubusercontent.com/99865210/175022163-97989496-a44d-47bc-9625-7d4becaf0a29.png)<br>
******************************************************************************************************************************************************************
program 11: To perform arthemetic operations<br>
import cv2<br><br>
import matplot<br>lib.image as mping<br>
import matplotlib.pyplot as plt<br>
img1=cv2.imread('flower2.jpg')<br>
img2=cv2.imread('flower3.jpg')<br>
fimg1=img1+img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2=img1-img2<br>
plt.imshow(fimg2)<br><br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3=img1*img2<br>
plt.imshow(fimg3)<br>
plt.show()cv2.imwrite('output.jpg',fimg3)<br>
fimg4=img1/img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>
cv2.imwrite('output.jpg',fimg4)<br>
output:![image](https://user-images.githubusercontent.com/99865210/175286745-cd75e48b-e145-46ee-95e8-45da54aaf083.png)<br>
![image](https://user-images.githubusercontent.com/99865210/175287145-050e7629-d28d-4122-8487-71351e1c024c.png)<br>
![image](https://user-images.githubusercontent.com/99865210/175287258-1721da91-bf87-4048-8144-879f34e2e9d7.png)<br>
****************************************************************************************************************************************************
program 12:mask and blurr<br>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=mpimg.imread("flower3.jpg")<br>
plt.imshow(img)<br>
plt.show()<br>
output:<br>
![image](https://user-images.githubusercontent.com/99865210/175288164-eb54126e-8d48-40a2-84ab-49732c46e033.png)<br>
**************************************************************************************************************************************************
hsv_img=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(18,255,255)<br>
mask=cv2.inRange(hsv_img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask) <br>
plt.subplot(1,2,1) <br>
plt.imshow(mask,cmap='gray') <br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>
 output:
![image](https://user-images.githubusercontent.com/99865210/175288488-b88bba43-c045-41b5-843d-3657910c7df7.png)<br>
<br>
![image](https://user-images.githubusercontent.com/99865210/175288507-d8a4faaf-f3dd-41da-9c9f-30e116c25e21.png)<br>
*********************************************************************************************************************************************************
light_white=(0,0,200) <br>
dark_white=(145,60,255) <br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white) <br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap='gray') <br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white) <br>
plt.show()<br>
output:![image](https://user-images.githubusercontent.com/99865210/175288679-5a29e88c-c407-4527-97a4-04f2a7b96ee3.png)<br>
![image](https://user-images.githubusercontent.com/99865210/175288725-7d085d65-c1df-429c-ba28-0db33d724e38.png)<br>
********************************************************************************************************************************************
final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br><br>
plt.subplot(1,2,1) <br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2) <br>
plt.imshow(final_result) <br>
plt.show() <br>
output:![image](https://user-images.githubusercontent.com/99865210/175288847-a89c5c65-0cc0-4db4-8d27-38535c6b3509.png)<br>
![image](https://user-images.githubusercontent.com/99865210/175288874-0adae1f4-54f6-4622-b3f7-1dbedd72eb3f.png)<br>
*************************************
blur=cv2.GaussianBlur(final_result,(7,7),0) <br>
plt.imshow(blur) <br>
plt.show()<br>
output:![image](https://user-images.githubusercontent.com/99865210/175288992-3539dab6-8f9f-456d-baeb-accada3e9443.png)<br>
**********************************************************************************************************************************************************
program 13:Develop a pgm to change the image to different color spaces<br>
import cv2
img=cv2.imread('D:\STUDENT\DataSet\images.jpg')<br><br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br><br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br><br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br><br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br><br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br><br>
cv2.imshow("GRAY image",gray)<br><br>
cv2.imshow("HSV image",hsv)<br><br>
cv2.imshow("LAB image",lab)<br><br>
cv2.imshow("HLS image",hls)<br><br>
cv2.imshow("YUV image",yuv)<br><br>
cv2.waitKey(0)<br><br>
cv2.destroyAllWindows()<br><br>
<br><br><br>
output:<br>![image](https://user-images.githubusercontent.com/99865210/178957793-001684d5-f0f8-40ae-8438-f497f586b5e8.png)<br><br><br>
***************************************************************************************************************************************************************
program 14 :program to create an image using 2D array<br>
import cv2 as c <br>
import numpy as np<br>
from PIL import Image<br>
array=np.zeros([100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('flower1.png')<br>
img.show()<br>
c.waitKey(0)  <br>
output:<br>![image](https://user-images.githubusercontent.com/99865210/178955357-18c7c949-fee6-4c4d-9352-1abd6cb80043.png)<br>
******************************************************************************************************************************************************************
program 15:Bitwise operation<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('b1.jpg',1)<br>
image2=cv2.imread('b1.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br><br>
output:![image](https://user-images.githubusercontent.com/99865210/176403037-7ab37abb-b660-4e53-9323-2c7359302cc9.png)<br>
*****************************************************************************************************************************************************************
program 16: bilateral<br>
import cv2<br>
import numpy as np
image=cv2.imread('b3.jpg')<br>
cv2.imshow('Original Image',image)<br>
cv2.waitKey(0)<br>
Gaussian=cv2.GaussianBlur(image, (7, 7),0)<br>
cv2.imshow('Gaussian Blurring',Gaussian)<br>
cv2.waitKey(0)<br>
median=cv2.medianBlur(image, 5)<br>
cv2.imshow('Median blurring',median)<br>
cv2.waitKey(0)<br>
bilateral=cv2.bilateralFilter(image, 9, 75, 75)<br>
cv2.imshow('Bilateral Blurring', bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllwindows()<br>
output:![image](https://user-images.githubusercontent.com/99865210/176406914-372e7044-6f21-476a-a737-4567d203098c.png)<br>
![image](https://user-images.githubusercontent.com/99865210/176407028-976d1b0b-c44f-4cc4-b771-79ecfa644d82.png)<br>
![image](https://user-images.githubusercontent.com/99865210/176407141-b5ef89ca-0274-4b13-bddb-d72e6b9b998d.png)<br>
![image](https://user-images.githubusercontent.com/99865210/176407279-0899768d-869d-4ec1-9a92-c31992d61f21.png)<br>
******************************************************************************************************************************************************************
program 17:Image Enhancement<br>
from PIL import Image<br><br>
from PIL import ImageEnhance<br>
image=Image.open('a3.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br><br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contarsted=enh_con.enhance(contrast)<br>
image_contarsted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>
output:![image](https://user-images.githubusercontent.com/99865210/176414127-21ad80f3-cbc1-4e5f-893c-c013170f251e.png)<br>
![image](https://user-images.githubusercontent.com/99865210/176414274-46d16353-2b6f-4e27-a1a8-6e4562aa65c8.png)<br>
![image](https://user-images.githubusercontent.com/99865210/176414719-08ab5c1d-0811-4829-989f-cc0d61df4f18.png)<br>
![image](https://user-images.githubusercontent.com/99865210/176414817-fb9797e4-ca1d-4998-9706-3fa8b75a31e7.png)<br>
![image](https://user-images.githubusercontent.com/99865210/176414936-c6ca42d8-66f0-4479-beb0-f47b8afbb80d.png)<br>

***************************************************************************************************************************************************************

program:18<br>


import cv2<br>
import numpy as np<br>
from matplotlib  import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('appu1.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernal=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernal)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernal)<br>
erosion=cv2.erode(img,kernal,iterations=1)<br>
dilation=cv2.dilate(img,kernal,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernal)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>
output:![image](https://user-images.githubusercontent.com/99865210/176419670-4c7310aa-e37f-4131-9644-5403dd59b901.png)<br>

***********************************************************************************************************************************************************************
program 19: image with background<br>
from PIL import Image<br>
image_file = 'test.tiff'<br>
image = Image.open(image_file).convert('L')<br>
histo = image.histogram()<br>
histo_string = ''<br>
for i in histo:<br>
  histo_string += str(i) + "\n"<br>
print(histo_string)<br>
 output:![image](https://user-images.githubusercontent.com/99865210/178947956-db39bd02-368f-49a8-a074-2cc95f95dd7c.png)<br>
 
 *****************************************************************************************************************************************************************
 program 20: image without background<br>
 import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('b4.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
  for j in range(0,y):<br>
     if(image[i][j]>50 and image[i][j]<150):<br>
       z[i][j]=255<br>
     else:<br>
        z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing w/o background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br><br>

output:![image](https://user-images.githubusercontent.com/99865210/178948588-e0860174-f9f1-4e9e-83ba-a3ea84490626.png)
****************************************************************************************************************************************************************

program :21<br>
import cv2<br>
OriginalImg=cv2.imread('b2.jpg')<br>
GrayImg=cv2.imread('b2.jpg',0)<br>
isSaved=cv2.imwrite('D:\A\i.jpg',GrayImg)<br>
cv2.imshow("display Original Image",OriginalImg)<br>
cv2.imshow("display Grayscale Image",GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print("the image is succesfully saved.")<br>
    
    
 ![image](https://user-images.githubusercontent.com/99865210/178970338-3cf80440-42e2-4ba5-a8ec-c414d2d0edcc.png)


******************************************************************************************************************************************************************
 program :22 histogram pgm<br>
 
 from skimage import io<br>
import matplotlib.pyplot as plt<br>
image = io.imread('a3.jpg')<br>
ax = plt.hist(image.ravel(), bins = 256)<br>
plt.show()<br>
output: <br>
from skimage import io<br><br>
import matplotlib.pyplot as plt<br><br>
image = io.imread('a3.jpg')<br><br>
ax = plt.hist(image.ravel(), bins = 256)<br><br>
plt.show()<br><br>
output:<br><br>

    ![image](https://user-images.githubusercontent.com/99865210/178966495-34f9e055-5a17-4bee-a178-24a32d26664a.png)<br>
    ![image](https://user-images.githubusercontent.com/99865210/178966495-34f9e055-5a17-4bee-a178-24a32d26664a.png)
********************************************************************************************************************************************************************
   
from skimage import io<br>
import matplotlib.pyplot as plt<br>
image = io.imread('a3.jpg')<br>
ax = plt.hist(image.ravel(), bins = 256)<br>
_ = plt.xlabel('Intensity Value')<br>
_ = plt.ylabel('Count') <br>
plt.show()<br>
<br>
output:![image](https://use<br>r-images.githubusercontent.com/99865210/178966669-d87c06af-3e6b-4be1-8792-cf9f01244280.png)

**********************************    
import cv2<br>
import numpy as np<br>
img  = cv2.imread('a3.jpg',0)<br>
hist = cv2.calcHist([img],[0],None,[256],[0,256])<br>
plt.hist(img.ravel(),256,[0,256])<br>
plt.show()<br>

output:![image](https://user-images.githubusercontent.com/99865210/178967163-01e5fdea-4940-40da-a2b4-d01766e81918.png)
***************************************
import numpy as np<br>
import cv2 as cv<br>
from matplotlib import pyplot as plt<br>
img = cv.imread('a3.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img = cv.imread('a3.jpg',0)<br>
plt.hist(img.ravel(),256,[0,256]);<br>
plt.show()<br>

output:![image](https://user-images.githubusercontent.com/99865210/178967295-f1c70083-3b77-4518-a9c1-31fdb00ef395.png)<br>

![image](https://user-images.githubusercontent.com/99865210/178967331-d7c62680-f109-458a-895f-cad6259be765.png)<br>

********************************************************************************************************************************************************************
Program 23:to perform basic image data analysis using intensity transformation:<br>
a) Image negative<br>
b) Log transformation<br>
c) Gamma correction<br>


%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('b1.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>


output:![image](https://user-images.githubusercontent.com/99865210/179951461-b0e17290-f853-4de5-83b0-151e50bc2c79.png)<br>

****************************************************
<br>
negative=255-pic <br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>

output:  ![image](https://user-images.githubusercontent.com/99865210/179951636-9f4e79f3-53f6-471e-be12-33c757e05d82.png)<br>
***************************************************
<br>
%matplotlib inline<br>
import imageio<br><br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('b1.jpg')<br>
gray=lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>
max_=np.max(gray)<br>
def log_transform():<br>
    return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>
    
   output:![image](https://user-images.githubusercontent.com/99865210/179951714-733564cf-0052-40d1-a6f1-9dfb4eb91ab7.png)<br>
   **************************************************
 
import imageio<br> <br>
import matplotlib.pyplot as plt<br>
pic=imageio.imread('b1.jpg')<br>
gamma=2.2<br>
gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>

output:![image](https://user-images.githubusercontent.com/99865210/179954778-bbd2a9ce-eed3-44c2-80db-311ed25372a7.png)<br>


*****************************************************************
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
my_image=Image.open('a3.jpg')<br>
sharp =my_image.filter(ImageFilter.SHARPEN)<br>
sharp.save('D:/image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>
<br>
output:![image](https://user-images.githubusercontent.com/99865210/179955996-e8150ad0-508f-43aa-a12b-5c2765212a96.png)<br>

********************************************************************
import matplotlib.pyplot as plt<br>
img=Image.open('a3.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>
flip.save('D:/image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>

output:![image](https://user-images.githubusercontent.com/99865210/179956169-7cb7cdaa-d709-4eb9-9d60-b9bd3a3c52c3.png)<br>
![image](https://user-images.githubusercontent.com/99865210/179956223-16573f6c-17c8-4ed0-b671-80beb7ed199e.png)<br>

******************************************************************
<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
im = Image.open('a3.jpg')<br>
width,height = im.size<br>
im1 = im.crop((280,200,800,700))<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>

output:![image](https://user-images.githubusercontent.com/99865210/179956342-7617a524-1d2f-4326-9e8d-69d66ec11691.png)<br>


**************************************************************************
program 24:matrix to image<br>
from PIL import Image<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
w, h = 600, 600<br><br>
data = np.zeros((h, w, 3), dtype=np.uint8<br>
data[0:100, 0:100] = [255, 0, 0]<br>
data[100:200, 100:200] = [0,255, 0]<br><br>
data[200:300, 200:300] = [0, 0, 255]<br><br>
data[300:400, 300:400] = [255, 70, 0]<br><br>
data[400:500, 400:500] = [255,120, 0]<br><br>
data[500:600, 500:600] = [ 255, 255, 0]<br><br>
      #len     width
img = Image.fromarray(data, 'RGB')<br>
plt.imshow(img)<br>
plt.show()<br>

output:![image](https://user-images.githubusercontent.com/99865210/180202389-a1473ec4-7cb2-4198-8d60-ac0addda21c9.png)<br>

![image](https://user-images.githubusercontent.com/99865210/183861603-a78f9c2e-320a-419b-9a45-4e9cfe33925d.png)<br>

********************************************************************************************
program 25: central pixel expamding by increasing its values<br>

import numpy as np<br>
import matplotlib.pyplot as plt<br>
arr = np.zeros((256,256,3), dtype=np.uint8)<br>
imgsize = arr.shape[:2]<br>
innerColor = (255, 255, 255)<br>
outerColor = (255, 0, 0)<br>
for y in range(imgsize[1]):<br>
    for x in range(imgsize[0]):<br>
        #Find the distance to the center<br>
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)<br>

        #Make it on a scale from 0 to 1innerColor<br>
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)<br>

        #Calculate r, g, and b values<br>
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
        #print r, g, b<br>
        arr[y, x] = (int(r), int(g), int(b))<br>
plt.imshow(arr, cmap='gray')<br>
plt.show()<br>
output:![image](https://user-images.githubusercontent.com/99865210/180202693-b0f8cbca-3980-468b-a992-040da35d8458.png)<br>


program 26:  to perform matrix operation on pixels<br>

import numpy as np<br>
#Create matrix<br>
matrix = np.array([[1, 2, 3],<br>
                   [4, 5, 6],<br>
                   [7, 8, 9]])<br>
#Return maximum element<br>
np.max(matrix)<br>

output: 9<br>

**************************************************
import numpy as np<br>
#Create matrix<br>
matrix = np.array([[1, 2, 3],<br>
                   [4, 5, 6],<br>
                   [7, 8, 9]])<br>
#Return maximum element<br>
np.min(matrix)<br>

output:1<br>

***************************************************


#example of pixel normalization<br>
from numpy import asarray<br>
from PIL import Image<br>
#load image<br>
image = Image.open('b3.jpg')<br>
pixels = asarray(image)<br>
#confirm pixel range is 0-255<br>
#print('Data Type: %s' % pixels.dtype)<br>
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))<br>
#convert from integers to floats<br>
pixels = pixels.astype('float32')<br>
#normalize to the range 0-1<br>
pixels /= 255.0<br>
#confirm the normalization<br>
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))<br>

output:Min: 0.000, Max: 255.000<br>
Min: 0.000, Max: 1.000<br>


*************************************************************
import cv2<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread("b3.jpg",0)<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
np.average(img)<br>


output:149.67725119422806<br>
![image](https://user-images.githubusercontent.com/99865210/181223879-6758a0b3-7704-4035-a98c-f0b44e629393.png)<br>
**************************************************************
from PIL import Image,ImageStat<br>
import matplotlib.pyplot as plt<br>
im=Image.open('b3.jpg')<br>
plt.imshow(im)<br>
plt.show()<br>
stat=ImageStat.Stat(im)<br>
print(stat.stddev)<br>
output:![image](https://user-images.githubusercontent.com/99865210/181224057-1080980e-c515-4b1c-93a9-7469ce74dbd9.png)<br>
[66.13053575068778, 68.6403333951999, 70.05475170427705]<br>
***********************************************************
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=imageio.imread('a3.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
max_channels = np.amax([np.amax(img[:,:,0]), np.amax(img[:,:,1]),np.amax(img[:,:,2])])<br>
print(max_channels)<br>

output:![image](https://user-images.githubusercontent.com/99865210/181224248-790d3226-e189-42ca-9685-40b22abdcd79.png)<br>
************************************************************
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
img=imageio.imread('a3.jpg' )<br>
plt.imshow(img)<br>
plt.show()<br>
min_channels = np.amin([np.amin(img[:,:,0]), np.amin(img[:,:,1]),np.amin(img[:,:,2])])<br>
print(min_channels)<br>

output:![image](https://user-images.githubusercontent.com/99865210/181224479-5e6f0399-0be8-4f4b-ae54-c5c24559a70d.png)<br>

*****************************************************************
import numpy as np<br>
x = np.ones((3, 3))<br>
x[1:-1, 1:-1] = 0<br>
x = np.pad(x, pad_width=1, mode='constant', constant_values=2)<br>
print(x)<br>
output:[[2. 2. 2. 2. 2.]<br>
 [2. 1. 1. 1. 2.]<br>
 [2. 1. 0. 1. 2.]<br>
 [2. 1. 1. 1. 2.]<br>
 [2. 2. 2. 2. 2.]]<br>
***********************************************************************************
#Python3 program for printing the rectangular pattern<br>
#Function to print the pattern<br>
def printPattern(n):<br>
 
    arraySize = n * 2 - 1;<br>
    result = [[0 for x in range(arraySize)]<br>
                 for y in range(arraySize)];<br>
         
    #Fill the values<br>
    for i in range(arraySize):
        for j in range(arraySize):<br>
            if(abs(i - (arraySize // 2)) ><br>
               abs(j - (arraySize // 2))):<br>
                result[i][j] = abs(i - (arraySize // 2));<br>
            else:<br>
                result[i][j] = abs(j - (arraySize // 2));<br>
             
    #Print the array<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            print(result[i][j], end = " ");<br>
        print("");<br>
 
#Driver Code<br><br><br>
n = 4;<br>
 
printPattern(n);<br><br>
output:<br><br>
3 3 3 3 3 3 3 <br><br>
3 2 2 2 2 2 3 <br><br>
3 2 1 1 1 2 3 <br><br>
3 2 1 0 1 2 3 <br><br>
3 2 1 1 1 2 3 <br><br>
3 2 2 2 2 2 3<br><br> 
3 3 3 3 3 3 3 <br><br>

***************************************************************************
import numpy as np<br>
import matplotlib.pyplot as plt<br>


array_colors = np.array([[[255, 0, 0], <br>
                         [0, 255, 0],<br>
                         [0, 0, 255]],<br>
                         [[255, 168, 0],<br> 
                    [255, 255, 0],<br>
                    [128, 128, 128]],<br>
                    [[255, 212, 0], <br>
                    [255, 0, 255],<br>
                    [240, 152, 255]],<br>
                    ])<br>
plt.imshow(array_colors);<br>
np.min(array_colors)<br>
<br>
output:![image](https://user-images.githubusercontent.com/99865210/181448313-8ce56177-c099-4156-bcf9-470d37376908.png)<br>

******************************************************************************************************************************************************************
program  :Edge detection using open CV<br>
import cv2<br>
#Read the original image<br>
img = cv2.imread('lion2.jpg') <br>
#Display original image<br>
cv2.imshow('Original', img)<br>
cv2.waitKey(0)<br>

#Convert to graycsale<br>
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)<br>
#Blur the image for better edge detection<br>
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) <br>

#Sobel Edge Detection<br>
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis<br>
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis<br>
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection<br>
#Display Sobel Edge Detection Images<br>
cv2.imshow('Sobel X', sobelx)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel Y', sobely)<br>
cv2.waitKey(0)<br>
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)<br>
cv2.waitKey(0)<br>
#Canny Edge Detection<br>
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
#Display Canny Edge Detection Image<br>
cv2.imshow('Canny Edge Detection', edges)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

output:![image](https://user-images.githubusercontent.com/99865210/186385183-691715f3-d55e-472d-bbce-50b07e772916.png)<br>
![image](https://user-images.githubusercontent.com/99865210/186385369-5c07ec3c-3337-4eec-b8e5-9ce4c98eefd4.png)<br>
![image](https://user-images.githubusercontent.com/99865210/186385495-1b2a83f4-b7d8-4a91-a6fa-e0a72e7e107e.png)<br>
![image](https://user-images.githubusercontent.com/99865210/186385640-2d00aac0-2f3e-4285-aa8e-ba44b8254425.png)<br>
![image](https://user-images.githubusercontent.com/99865210/186385787-2fe3cf67-b299-4952-b273-5b4585c33d6d.png)<br>





**********************************************************************************************************************************
USING PILLOW FUNCTIONS<br>


from PIL import Image, ImageChops, ImageFilter
from matplotlib import pyplot as plt
#Create a PIL Image objects
x = Image.open("x.png")
o = Image.open("o.png")
#Find out attributes of Im<br>age Objects
print('size of the image: ', x.size, ' colour mode:', x.mode)
print('size of the image: ', o.size, ' colour mode:', o.mode)
#plot 2 images one besides the other<br>
plt.subplot(121), plt.imshow(x)<br>
plt.axis('off')<br>
plt.subplot(122), plt.i<br>mshow(o)
plt.axis('off')<br>
#multiply images<br>
merged = ImageChops.multiply(x,o)<br>
#adding 2 images<br>
add = ImageChops.add(x,o)<br>
#convert colour mode
greyscale = mer<br>ged.convert('L')
greyscale<br>
OUTPUT<br>
![image](https:/<br>/user-images.githubusercontent.com/99865210/187867136-f3fc6fc5-cfad-49cb-bb4f-f935ab674061.png)
<br>



image = merged<br>
print('image size: ', image.size,<br>
'\ncolor mode: ', image.mode,<br>
'\nimage width: ', image.width, '| also represented by: ',image.size[0],<br>
'\nimage height: ',image.height, '| also represented by: ',image.size[1],)<br>
OUTPUT<br>
![img2](https://user-images.githubusercontent.com/99865210/187881108-e6ab46ac-1661-4e0f-9365-b743051817ee.png)<br>



#mapping the pixels of the image so we can use them as coordinates<br>
pixel = greyscale.load()<br>

#a nested Loop to parse through all the pixels in the image<br>
for row in range (greyscale.size[0]):<br>
for column in range(greyscale.size[1]):<br>
if pixel[row, column] != (255):
pixel[row, column] = (0)<br>
greyscale<br>
OUTPUT<br>
![img3](https://user-images.githubusercontent.com/99865210/187881210-cacb31ca-b6c9-4bd0-8aab-bccd36770fb9.png)<br>


<br>
invert = ImageChops.invert(greyscale)<br>

#2.invert by subtraction<br>
bg = Image.new('L', (256, 256), color=(255)) #create a new image with a solid white background<br>
subt = ImageChops.subtract(bg, greyscale) #subtract image from background<br>
<br>
#3. rotate<br>
rotate =subt.rotate(45)<br>
rotate<br>
OUTPUT<br>
![img4](https://user-images.githubusercontent.com/99865210/187881377-12fdebb5-b622-4570-b3b1-26b5642c0268.png)<br>



#gaussian blur<br>
blur = greyscale.filter(ImageFilter.GaussianBlur (radius=1))<br>
<br>
#edge detection<br>
edge = blur.filter(ImageFilter.FIND_EDGES)<br>
edge<br>
OUTPUT<br>
![img5](https://user-images.githubusercontent.com/99865210/187881505-fefce4a9-183f-414a-b631-7627205c4181.png)<br>


<br><br>
#change edge colours<br><br>
edge = edge.convert('RGB')<br><br>
bg_red = Image.new('RGB', (256,256), color=(255,0,0))<br><br>

filled_edge = ImageChops.darker(bg_red, edge)<br><br>
filled_edge<br><br>
OUTPUT<br><br>
![img6](https://user-images.githubusercontent.com/99865210/187881537-dbfe58ed-dbc8-4ec3-96c7-4166e36f01a5.png)<br><br>



edge.save('processed.png')<br><br>
***************************************************************************************************************************************************

IMAGE RESTORATION<br><br>
1(restore a damaged image )<br><br>

import numpy as np<br><br>
import cv2<br><br>
import matplotlib.pyplot as plt<br><br>
#Open the image.<br><br>
img = cv2.imread('dimage_damaged.png')<br><br>
plt.imshow(img)<br><br>
plt.show()<br><br>
#Load the mask.<br><br>
mask= cv2.imread('dimage_mask.png', 0)<br><br>
plt.imshow(mask)<br><br>
plt.show()<br>
#Inpaint<br>.
dst = cv2.inpaint (img, mask, 3, cv2.INPAINT_TELEA)<br>
#write the output.<br>
cv2.imwrite('dimage_inpainted.png', dst)<br>
plt.imshow(dst)<br>
plt.show()<br>
OUTPUT<br>
![img7](https://user-images.githubusercontent.com/99865210/187881790-add906f5-e7fd-4a8a-b680-611963be51bd.png)<br>

********************************************************************************************************************************
2.Removing logos:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)


def show_image(image, title='
Image', cmap_type='gray'):
plt.imshow(image, cmap=cmap_type)
plt.title(title)
plt.axis('off')
def plot_comparison (img_original, img_filtered, img_title_filtered):
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
ax1.imshow(img_original, cmap=plt.cm.gray)
ax1.set_title('Original')
ax1.axis('off')
ax2.imshow(img_filtered, cmap=plt.cm.gray)
ax2.set_title(img_title_filtered)
ax2.axis('off')


from skimage.restoration import inpaint
from skimage.transform import resize
from skimage import color


image_with_logo= plt.imread('imlogo.png')
#Initialize the mask
mask= np.zeros(image_with_logo.shape[:-1])
#Set the pixels where the Logo is to 1
mask [210:272, 360:425] = 1
#Apply inpainting to remove the Logo
image_logo_removed =inpaint.inpaint_biharmonic (image_with_logo,
mask, multichannel=True)
#Show the original and Logo removed images
plot_comparison (image_with_logo, image_logo_removed, 'Image with logo removed')
OUTPUT
![image](https://user-images.githubusercontent.com/99865210/187881963-623abfcb-20b8-49fe-aa31-a5efe2af6b38.png)
![image](https://user-images.githubusercontent.com/99865210/187882004-84ff6aae-ca95-4e39-a6b0-febc879ec13c.png)


********************************************************************************************************************************************
2.NOISE
(ADDING A NOISE)

from skimage.util import random_noise
fruit_image = plt.imread('fruitts.jpeg')
#Add noise to the image
noisy_image = random_noise (fruit_image)
#Show th original and resulting image
plot_comparison (fruit_image, noisy_image, 'Noisy image')
OUTPUT
![img8](https://user-images.githubusercontent.com/99865210/187882699-b5e7c0a8-ac3d-4dd9-9c6c-7d23099eda9b.png)
**************************************
(REDUSING NOISE)
from skimage.restoration import denoise_tv_chambolle
noisy_image = plt.imread('noisy.jpg')

#Apply total variation filter denoising
denoised_image = denoise_tv_chambolle (noisy_image, multichannel=True)

#Show the noisy and denoised image
plot_comparison (noisy_image, denoised_image, 'Denoised Image')
OUTPUT
![img09](https://user-images.githubusercontent.com/99865210/187882934-97b3f585-c76b-43c9-9f67-f7bf776b33ae.png)
************************************
(REDUCING A NOISE WHILE PRESERVING EDGES)

from skimage.restoration import denoise_bilateral
landscape_image = plt.imread('noisy.jpg')
#Apply bilateral filter denoising
denoised_image = denoise_bilateral (landscape_image, multichannel=True)
#Show original and resulting images
plot_comparison (landscape_image, denoised_image, 'Denoised Image')
OUTPUT
![img10](https://user-images.githubusercontent.com/99865210/187883123-bf39431a-c838-4919-b8e8-11f4e7f7b817.png)
*********************************************************************************************************************************************

3 SEGMENTATION:
(SUPERPIXEL SEGMENTATION)

from skimage.segmentation import slic
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import numpy as np
face_image = plt.imread('face.jpg')
segments = slic(face_image, n_segments=400)
segmented_image=label2rgb(segments,face_image,kind='avg')
plt.imshow(face_image)
plt.show()
plt.imshow((segmented_image * 1).astype(np.uint8))
plt.show()

![image](https://user-images.githubusercontent.com/99865210/187901367-67fed0fe-80ab-4452-a22c-7b1b641fcbea.png)




********************************************************************************************************************************************
4 CONTOURS
(CONTOURING  SHAPES)


def show_image_contour (image, contours):
plt.figure()
for n, contour in enumerate (contours):
plt.plot(contour[:, 1], contour[:,0], linewidth=3)
plt.imshow(image, interpolation='nearest', cmap='gray_r')
plt.title('Contours')
plt.axis('off')
from skimage import measure, data<br>
#Obtain the horse image
horse_image = data.horse()
#Find the contours with a constant Level value of 0.8
contours = measure.find_contours (horse_image, level=0.8)
#Shows the image with contours found
show_image_contour (horse_image, contours)
OUTPUT
![img12](https://user-images.githubusercontent.com/99865210/187883788-f4044501-0bb8-41fc-9494-3ceb330a0c5b.png)
*****************************************************
2.FIND CONTOURS OF AN IMAGE THAT  IS NOT BINARY

from skimage.io import imread
from skimage.filters import threshold_otsu
image_dices = imread('diceimg.png')
#Make the image grayscale
image_dices = color.rgb2gray(image_dices)
#Obtain the optimal thresh value
thresh = threshold_otsu (image_dices)
#Apply thresholding
binary = image_dices > thresh
#Find contours at a constant value of 0.8
contours = measure.find_contours(binary, level=0.8)
#Show the image
show_image_contour(image_dices, contours)
OUTPUT
![img13](https://user-images.githubusercontent.com/99865210/187884016-6f064f20-69ec-4203-94f7-5af1ef9a8bb4.png)

*****************************************************************************************************************************************
3.COUNT THE DOTS IN THE DICE IMAGE

#Create List with the shape of each contour
shape_contours = [cnt.shape[0] for cnt in contours]
#set se as the maximum size of the dots shape
max_dots_shape = 50
#Count dots in contours excluding bigger than dots size
dots_contours = [cnt for cnt in contours if np.shape (cnt) [0] < max_dots_shape]
#Shows all contours found
show_image_contour (binary, contours)
#Print the dice's number
print('Dices dots number: {}'.format(len(dots_contours)))
OUTPUT
![img14](https://user-images.githubusercontent.com/99865210/187884268-5987ea3e-1bd1-4ace-88cd-9ea23b6ddff0.png)

********************************************************************************************************************************************
VARIOUS EDGE DETECTIOM

1.CANNY EDGE DETECTION
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

loaded_image = cv2.imread("angry bird.png")
loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)

gray_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY)

edged_image = cv2.Canny(gray_image, threshold1=30, threshold2=100)

plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(loaded_image,cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(gray_image,cmap="gray")
plt.axis("off")
plt.title("Grayscale Image")
plt.subplot(1,3,3)
plt.imshow(edged_image,cmap="gray")
plt.axis("off")
plt.title("Canny Edge Detected Image")
plt.show()

OUTPUT:![image](https://user-images.githubusercontent.com/99865210/187899206-6d0ece9d-aa3e-440b-8c29-a2870063ba51.png)

***************************************************
2.EDGE DETECTING  SCHEMAS -THE GRADIENT(SOBEL 1ST ORDER DERIVATIVES(BASED EDGE DETECTION AND THE  LAPLICATION(2ND ORDER DERIVATIVE IT IS EXTREMELY SENSITIVE TO NOISE)BASED EDGE DETECTOR

#Laplacian and Sobel Edge detecting methods
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Loading image
#imge = cv2.imread('SanFrancisco.jpg',) 
imge0 = cv2.imread('angry bird.png',)
#converting to gray scale
gray = cv2.cvtColor(imge0, cv2.COLOR_BGR2GRAY)
# remove noise
img = cv2.GaussianBlur(gray,(3,3),0)
#convolute with proper kernels
laplacian = cv2.Laplacian (img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) #y

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([]) 
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3), plt.imshow(sobelx, cmap = 'gray')
plt.title('Sobel x'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
          
plt.show()

OUTPUT:![image](https://user-images.githubusercontent.com/99865210/187899838-528abf46-68e2-4a25-81fa-cf05434a92a9.png)
*****************************************************************
3.EDGE DETECTION USING PREWITT OPERATOR

import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('angry bird.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

cv2.imshow("Original Image", img)
cv2.imshow("Prewitt x", img_prewittx)
cv2.imshow("Prewitt y", img_prewitty)
cv2.imshow("Prewitt", img_prewittx + img_prewitty)
cv2.waitKey()
cv2.destroyAllWindows()

OUTPUT:![image](https://user-images.githubusercontent.com/99865210/187900196-f9b14ba8-d7ae-4029-b76c-35d1d977aa9a.png)
**************************************************************************
4.ROBERT'S EDGE DETECTION-ROBERTS CROSS OPERATOR

#Roberts Edge Detection- Roberts cross operator import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
roberts_cross_v = np.array([[1, 0 ],
                            [0,-1]])

roberts_cross_h = np.array([[0, 1],
                            [-1, 0 ]] )
img = cv2.imread("angry bird.png",0).astype('float64')
img/=255.0
vertical = ndimage.convolve( img, roberts_cross_v ) 
horizontal=ndimage.convolve( img, roberts_cross_h)

edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
edged_img*=255
cv2.imwrite("output.jpg", edged_img)
cv2.imshow("OutputImage", edged_img)
cv2.waitKey()
cv2.destroyAllWindows()

OUTPUT:![b1](https://user-images.githubusercontent.com/99865210/187900494-a590c5df-9f5a-40f8-bf97-62861f5bf5ec.png)



