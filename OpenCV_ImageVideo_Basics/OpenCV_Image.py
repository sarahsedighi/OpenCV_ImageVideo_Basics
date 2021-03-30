'''
@author: Sara

OpenCV _ Image Processing _ Basics

'''

## PART 1: read, write and display and image

import cv2

# 1-1: read an image

img_rgb = cv2.imread('img.jpg', 1) # 1: 3 channels, RGB
img_gray1 = cv2.imread('img.jpg', 0) # 0: 1 channels, Gray
# or:
img_gray2 = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img_rgbalpha = cv2.imread('img.jpg', -1) # -1: 4 channels, RGBalpha

print(img_gray2)

# 1-2: display an image

cv2.imshow('RGB image', img_rgb)
cv2.waitKey(0)
cv2.imshow('Gray image', img_gray1)
cv2.waitKey(0)
cv2.imshow('Gray image', img_gray2)
cv2.waitKey(0)
cv2.imshow('RGBalpha image', img_rgbalpha)
cv2.waitKey(0)

# 1-3: write (save) an image

cv2.imwrite('img_1.jpg', img_gray1)

cv2.destroyAllWindows()

# 1-4: open one image on matplotlib

from matplotlib import pyplot as plt

img = cv2.imread('img.jpg', -1)
cv2.imshow('Image', img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)

# plt.xticks([]), plt.yticks([])

plt.show()

# 1-5: open multiple images on matplotlib

img = cv2.imread('img.jpg')
_, th0 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
_, th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
_, th2 = cv2.threshold(img, 50, 255, cv2.THRESH_TRUNC)
_, th3 = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)
_, th4 = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO_INV)

titles = ['Main Image', 'BINARY', 'BINARY-INV', 'TRUNC', 'TOZERO', 'TOZERO-INV']
images = [img, th0, th1, th2, th3, th4]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

#%%
## PART 2: drawing functions

import cv2
import numpy as np

# BGR color picker: https://wamingo.net/rgbbgr/

# 2-1: insert shapes
# 2-1-1: draw line/ arrowed line/ rectangle/ circle

img = cv2.imread('img.jpg', 1)

img = cv2.line(img, (0, 0), (400, 100), (255, 0 , 0), 5) # img = cv2.line(img, (P1), (P2), (B, G, R), int)
img = cv2.arrowedLine(img, (0, 0), (700, 500), (0, 255, 0), 5)
img = cv2.rectangle(img, (300, 300), (500, 500), (0, 0, 255), 5)
img = cv2.circle(img, (100, 200), 50, (50, 50, 0), 5) # img = cv2.circle(img, (center), (radius), (B, G, R), int)

cv2.imshow('with shapes', img)
cv2.waitKey(0)

# 2-1-2: filled rectangle and circle

img = cv2.rectangle(img, (300, 300), (500, 500), (0, 0, 255), -1)
img = cv2.circle(img, (100, 200), 50, (50, 50, 0), -1)

cv2.imshow('with filled shapes', img)
cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# 2-2: insert polyline

import cv2
import numpy as np

img = cv2.imread('img.jpg', 1)

points = np.array([[100, 400], [200, 500], [250, 600], [150, 600]], np.int32)
points = points.reshape((-1, 1, 2))

points1 = np.array([[300, 400], [400, 500], [450, 600], [350, 600]], np.int32)
points1 = points1.reshape((-1, 1, 2))

points2 = np.array([[700, 400], [800, 500], [850, 600], [750, 600]], np.int32)
points2 = points2.reshape((-1, 1, 2))

img = cv2.polylines(img, points, True, (0, 255, 0), 10)
img = cv2.polylines(img, [points1], True, (0, 255, 0), 10) # automatically closed polylines
img = cv2.polylines(img, [points2], False, (0, 255, 0), 10)

cv2.imshow('with polylines', img)
cv2.waitKey(0)

#%%
# 2-3: put text

img = cv2.imread('img.jpg', 1)

# img = cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
img = cv2.putText(img, 'Hello', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)

cv2.imshow('with text', img)
cv2.waitKey(0)

#%%
# 2-4: generate an image

# Black Background
img1 = np.zeros((700, 500, 3), np.uint8) 

# White-Gray Background
img2 = np.zeros((700, 500, 3), np.uint8) 
img2.fill(255)

# Random Background
img3 = np.random.rand(700, 500, 3)


img1 = cv2.rectangle(img1, (300, 300), (400, 400), (0, 0, 255), -1)
img1 = cv2.circle(img1, (100, 200), 50, (50, 50, 0), -1)
cv2.imwrite('image1.jpg', img1)

img2 = cv2.rectangle(img2, (300, 300), (400, 400), (0, 0, 255), -1)
img2 = cv2.circle(img2, (100, 200), 50, (50, 50, 0), -1)
cv2.imwrite('image2.jpg', img2)

img3 = cv2.rectangle(img3, (300, 300), (400, 400), (0, 0, 255), -1)
img3 = cv2.circle(img3, (100, 200), 50, (50, 50, 0), -1)
cv2.imwrite('image3.jpg', img3)

cv2.imshow('Black Background', img1)
cv2.waitKey(0)
cv2.imshow('White-Gray Background', img2)
cv2.waitKey(0)
cv2.imshow('Random Background', img3)
cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# PART 3: understanding an image

# 3-1: size, resize
# size
img = cv2.imread('img.jpg', 1)
print(img.shape)

rows, cols, chs = img.shape
print('H:', rows)
print('W:', cols)
print('C:', chs)

# resize 1
img_resize1 = cv2.resize(img, (100, 150))
print(img1.shape)

# resize 2
img_resize2 = cv2.resize(img, None, fx = 0.5, fy = 1) # same size: fy = 1
print(img_resize2.shape)

cv2.imshow('org image', img)
cv2.imshow('resize type 1', img_resize1)
cv2.imshow('resize type 2', img_resize2)
cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# 3-2: transformation: shift and rotation

import cv2
import numpy as np

img = cv2.imread('img.jpg', 1)

# 3-2-1: shift
M_shift = np.float32([[1,0,50], [0,1,100]])
img_shift = cv2.warpAffine(img, M_shift, (1024, 684))

# 3-2-2: rotation
M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), 90, .5)
img_rotation = cv2.warpAffine(img, M_rot, (1024, 684))

cv2.imshow('shift', img_shift)
cv2.imshow('rotaion', img_rotation)
cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# 3-3: get and set pixel values

img = cv2.imread('img.jpg', 1)

img[150:160, 350:360] = (0, 0, 0)

cv2.imshow('change pixel values', img)
cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# 3-4: ROI (region of interes)

img = cv2.imread('img.jpg', 1)
img_roi = img[150:350, 450:580]

cv2.imwrite('ROI.jpg', roi_img)
cv2.imshow('ROI image', roi_img)
cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# 3-5: copy and paste roi

img = cv2.imread('img.jpg', 1)
img_roi = img[150:350, 450:580]
img[50:250, 50:180] = img_roi

# or: img[50:250, 50:180] = img[150:350, 450:580]

img2 = cv2.imread('lena.tif', 1)
img2[50:250, 50:180] = img[150:350, 450:580]

cv2.imshow('copy roi', img)
cv2.imshow('copy roi in new image', img2)
cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# 3-6: sum two images

img1 = cv2.imread('img.jpg', 1)
img2 = cv2.imread('scene.png', 1)

rows1, cols1, chs1 = img1.shape
rows2, cols2, chs2 = img2.shape

img2 = cv2.resize(img2, (cols1, rows1))

sum_img = cv2.add(img1, img2)
weighted_sum_img = cv2.addWeighted(img1, 1, img2, 0.5, 0)

# like watermark: 
# weighted_sum_img = cv2.addWeighted(img1, 1, img2, 0.05, 0)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('sum_img', sum_img)
cv2.imshow('weighted_sum_img', weighted_sum_img)
cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# 3-7: subtract two images

img1 = cv2.imread('img.jpg', 1)
img2 = cv2.imread('scene.png', 1)

rows1, cols1, chs1 = img1.shape
rows2, cols2, chs2 = img2.shape

img2 = cv2.resize(img2, (cols1, rows1))

diff = cv2.absdiff(img1, img2)

cv2.imshow('diff', diff)
cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# 3-8: mouse click position (x,y) - right click

def click_event(event, x, y,  flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ',', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ',' + str(y)
        cv2.putText(img, strXY, (x,y), font, 1, (0, 0, 255), 2)
        cv2.imshow('Image', img)

img = cv2.imread('img.jpg')
cv2.imshow('Image', img)

cv2.setMouseCallback('Image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()   

#%%
# 3-9: mouse click BGR channels - right click
    
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        Blue = img[y, x, 0]
        Green = img[y, x, 1]
        Red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(Blue) + ',' + str(Green) + ',' + str(Red)
        cv2.putText(img, strBGR, (x,y), font, 1, (0, 255, 255), 1)
        cv2.imshow('Image', img)

img = cv2.imread('img.jpg')
cv2.imshow('Image', img)

cv2.setMouseCallback('Image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
# 3-10: draw connecting points by right click

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
        points.append((x,y))
        if len(points) >= 2:
            cv2.line(img, points[-1], points[-2], (255, 0, 0), 5)
        cv2.imshow('Image', img)

img = np.zeros((500, 500, 3), np.uint8)
cv2.imshow('Image', img)

points = [ ]

cv2.setMouseCallback('Image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
# 3-11: tracking HSV values    

def display():
    pass

cv2.namedWindow('HSV Tracker')
cv2.createTrackbar('LH', 'HSV Tracker', 0, 255, display)
cv2.createTrackbar('LS', 'HSV Tracker', 0, 255, display)
cv2.createTrackbar('LV', 'HSV Tracker', 0, 255, display)
cv2.createTrackbar('UH', 'HSV Tracker', 255, 255, display)
cv2.createTrackbar('US', 'HSV Tracker', 255, 255, display)
cv2.createTrackbar('UV', 'HSV Tracker', 255, 255, display)


img = cv2.imread('img.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

l_h = cv2.getTrackbarPos('LH', 'HSV Tracker')
l_s = cv2.getTrackbarPos('LS', 'HSV Tracker')
l_v = cv2.getTrackbarPos('LV', 'HSV Tracker')

u_h = cv2.getTrackbarPos('UH', 'HSV Tracker')
u_s = cv2.getTrackbarPos('US', 'HSV Tracker')
u_v = cv2.getTrackbarPos('UV', 'HSV Tracker')

lower_hsv = np.array([l_h, l_s, l_v])
upper_hsv = np.array([u_h, u_s, u_v])

mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

output_mask = cv2.bitwise_and(img, img, mask=mask_hsv)

cv2.imshow('Image', img)
cv2.imshow('Mask', mask_hsv)
cv2.imshow('Output', output_mask)

key = cv2.waitKey(1)
if key == 27:
    break

cv.destroyAllWindows()

#%%
# PART 4: thresholding

import cv2

gray = cv2.imread('img.jpg', 0)
gray = cv2.resize(gray, (500,500))

ret, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
ret, bw_inv = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
ret, bw_trunc = cv2.threshold(gray, 100, 255, cv2.THRESH_TRUNC)   # truncate
ret, bw_tr_0 = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO)   # threshold to zero
ret, bw_tr_inv = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO_INV)   # threshold to zero, inverted

cv2.imshow('original', gray)
cv2.imshow('threshold', bw)
cv2.imshow('threshold_inv', bw_inv)
cv2.imshow('threshold_inv_trunc', bw_trunc)
cv2.imshow('threshold_tr_0', bw_tr_0)
cv2.imshow('threshold_tr_inv', bw_tr_inv)
cv2.waitKey(0) 

cv2.destroyAllWindows()

#%% 
# PART 5:image histogram

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg')
# b, g, r = cv2.split(img)

plt.hist(img.ravel(), 256, [0, 255])

plt.hist(b.ravel(), 256, [0, 256])
plt.hist(g.ravel(), 256, [0, 256])
plt.hist(r.ravel(), 256, [0, 256])

plt.show()

#%%
# PART 6: smoothing image and noise removal

import cv2
import numpy as np

img = cv2.imread('img.jpg')

img_blurred = cv2.cv2.blur(img, (5,5))
img_gaussian = cv2.GaussianBlur(img, (5,5), 0)
img_median = cv2.medianBlur(img, 5)

cv2.imshow('img', img)
cv2.imshow('blurred', img_blurred)
cv2.imshow('gaussian', img_gaussian)
cv2.imshow('median', img_median)
cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# PART 7: edge detection: canny, sobel and laplacian filters

import cv2
import numpy as np

img = cv2.imread('img.jpg')

img_gaussian = cv2.GaussianBlur(img, (9,9), 0)

img_canny = cv2.Canny(img, 100, 150)
img_gaussian_canny = cv2.Canny(img_gaussian, 100, 150)

img_sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
img_sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 3)

img_lab = cv2.Laplacian(img, cv2.CV_64F, ksize = 3)

cv2.imshow('img', img)
cv2.imshow('gaussian', img_gaussian)
cv2.imshow('canny', img_canny)
cv2.imshow('canny_gaussian', img_gaussian_canny)
cv2.imshow('sobelx', img_sobelx)
cv2.imshow('sobely', img_sobely)
cv2.imshow('lab', img_lab)

cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# PART 8: contours

import cv2
import numpy as np

img = cv2.imread('img.jpg')
img1 = cv2.imread('img.jpg')
img2 = cv2.imread('img.jpg')
img3= cv2.imread('img.jpg')
img4= cv2.imread('img.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 100, 255, 0)

conts, heir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# -1: draw all contours
cv2.drawContours(img, conts, -1, (255, 0, 0))

# contour #1 (first) or #2 (second) or #3 (third) ...
cv2.drawContours(img1, conts, 1, (255, 0, 0))
cv2.drawContours(img2, conts, 2, (255, 0, 0))
cv2.drawContours(img3, conts, 3, (255, 0, 0))

for cont in conts:
    if cv2.contourArea(cont) > 1000:
        cv2.drawContours(img4, [cont], 0, (0, 255, 255), 3)

cv2.imshow('all contours', img)
cv2.imshow('1st contour', img1)
cv2.imshow('second contour', img2)
cv2.imshow('third contours', img3)
cv2.imshow('large contours', img4)

cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# PART 9: morphological transformation
# 9-1: erosion and dilation

import cv2
import numpy as np

img = cv2.imread('img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

kernal = np.ones((5,5), np.uint8)

dilation = cv2.dilate(thresh, kernal)
erosion = cv2.erode(thresh, kernal)

cv2.imshow('dilation', dilation)
cv2.imshow('erosion', erosion)

cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# 9-2: opening and closing

# manual opening: 1-erosion 2-dilate
# manual closing: 1-dilate 2-erosion

kernal1 = np.ones((3,3), np.uint8)
kernal2 = np.ones((9,9), np.uint8)

erosion = cv2.erode(thresh, kernal1)
dilation = cv2.dilate(erosion, kernal2)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal2)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernal1)

cv2.imshow('opening', opening)
cv2.imshow('closing', closing)

cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# PART 10: template matching

import cv2
import numpy as np

# img = cv2.imread('img.jpg')
# template = cv2.imread('template.jpg')

img = cv2.imread('img.jpg', 1)
template = img[150:350, 450:580]

cv2.imwrite('roi.jpg', template)

template = cv2.imread('roi.jpg')
# w, h = template.shape[::-1]
w, h, c =template.shape

result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
minval, maxval, minLoc, maxLoc = cv2.minMaxLoc(result)

# cv2.TM_SQDIFF and cv2.TM_SQDIFF_NORMED => topLeft = minLoc

topLeft = maxLoc

cv2.rectangle(img, topLeft, (topLeft[0]+h, topLeft[1]+w), (255,0,0), 3)

cv2.imshow('img', img)
cv2.imshow('template', template)
cv2.waitKey(0)

cv2.destroyAllWindows()

#%%
# PART 11: geometric shape detection

import cv2

img = cv2.imread('image1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

conts, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(img, conts, -1, (255, 255, 0), 3)

for cont in conts:
    approx = cv2.approxPolyDP(cont, 0.01* cv2.arcLength(cont, True), True)
    print(len(approx))
    if len(approx) == 4: # 3 for triangles and 4 for squares
        cv2.drawContours(img, [cont], -1, (255,255,0), 3)

cv2.imshow('img', img)
cv2.imshow('bw', bw)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% 
# PART 12: line detection

import cv2
import numpy as np

img = cv2.imread('lines.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 100, 150)

lines = cv2.HoughLinesP(canny, 1, np.pi/180, 25, maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)

cv2.imshow('img', img)
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
# PART 13: fourier transform

import cv2
import numpy as np

img = cv2.imread('img.jpg',0)

ft = np.fft.fft2(img)
fs = np.fft.fftshift(ft)
mag = 20*np.log(np.abs(fs))
mag = np.asanyarray(mag, dtype=np.uint8)
mag_shift = 20*np.log(np.abs(fs))
mag_shift = np.asanyarray(mag_shift, dtype=np.uint8)

img_mag = np.concatenate((img, mag_shift), axis=1) # show images in one window

cv2.imshow('img', img)
cv2.imshow('mag', mag)
cv2.imshow('mag_shift', mag_shift)
cv2.imshow('img_mag', img_mag)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
# PART 14: face and eye detection

import os
import cv2

img = cv2.imread('img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
face_cascade = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
eye_cascade = os.path.join(cv2_base_dir, 'data/haarcascade_eye_tree_eyeglasses.xml')

# or download and use xml
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

faces = face_cascade.detectMultiScale(gray, 1.1, 4) # it can detect more than one face
eyes = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 3)

for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 3)
    
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2 

# 14-1: face detection using Haar Cascade classifier

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('img.jpg')
gray_img = cv2.cvtColor(img, cv.COLOR_BGR2GRAY)
face_detect = face_cascade.detectMultiScale(gray_img, 1.1, 3)

for (x, y, w, h) in face_detect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=5)

cv2.imshow('Image', img)
cv2.waitKey(0)

# 14-2: eye detection using Haar Cascade 

eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

img = cv2.imread('img.jpg')
gray_img = cv2.cvtColor(img, cv.COLOR_BGR2GRAY)

eye_detect = eye_cascade.detectMultiScale(gray_img)

for (x, y, w, h) in eye_detect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=4)

cv2.imshow('Image', img)
cv2.waitKey(0)

# 14-3: face and eye detection

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

img = cv2.imread('img.jpg')
gray_img = cv2.cvtColor(img, cv.COLOR_BGR2GRAY)
face_detect = face_cascade.detectMultiScale(gray_img, 1.1, 4)
eye_detect = eye_cascade.detectMultiScale(gray_img)

for (x, y, w, h) in face_detect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=3)

    for (ex, ey, ew, eh) in eye_detect:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), thickness=3)

cv.imshow('Image', img)
cv.waitKey(0)

#%%
# PART 15: feature detection and description

import cv2
import numpy as np

img = cv2.imread('img.jpg', 1)
# img = cv2.imread('img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('template.jpg', 0)

# cv2.matchTemplate => feature matching
# SIFT, SURF, ORB

orb = cv2.ORB_create(nfeatures=500)

kf1, des1 = orb.detectAndCompute(gray, None)
kf2, des2 = orb.detectAndCompute(template, None)

gray = cv2.drawKeypoints(gray, kf1, None)
template = cv2.drawKeypoints(template, kf2, None)

cv2.imshow('img', gray)
cv2.imshow('template', template)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% 
# PART 16: feature matching and object detection

import cv2
import numpy as np

template = cv2.imread('template.jpg', 0)

# cv2.matchTemplate => feature matching
# (SIFT, SURF), ORB

# detect features
orb = cv2.ORB_create(nfeatures=1000)

kf1, des1 = orb.detectAndCompute(gray, None)
kf2, des2 = orb.detectAndCompute(template, None)

# matching Features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matched = bf.match(des1, des2)
matched = sorted(matched, key = lambda x:x.distance)

# drawing
gray = cv2.drawKeypoints(gray, kf1, None)
template = cv2.drawKeypoints(template, kf2, None)
matchImg = cv2.drawMatches(gray, kf1, template, kf2, matched[:50], None, flags = 2)

cv2.imshow('template', template)
cv2.imshow('matchImg', matchImg)
cv2.waitKey(0)
cv2.destroyAllWindows()














































