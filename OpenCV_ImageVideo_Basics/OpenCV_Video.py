"""
@author: Sara

OpenCV _ Video Processing _ Basics

"""

# PART 1: display a video or camera

import cv2
import numpy as np

# run a camera
# numbers: number of using camera: 0, 1, 2...
# cap = cv2.VideoCapture(0)

# run a saved video
# path = stream
cap = cv2.VideoCapture('stream.avi')

while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('stream', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
    # use this if it doesn't break
    # if cv2.waitKey(1) & 0xFF == ord('q'):
     #    break
        
        
cap.release()
cv2.destroyAllWindows()

# ascii table: escape27

#%%

import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('streat.avi')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

# https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

#%%
# PART 2: understanding a video

cap = cv2.VideoCapture('stream.avi')
print(cap.isOpened())
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('stream', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()

#%% 
# PART 3: flipped

import cv2
import numpy as np

cap = cv2.VideoCapture('stream.avi')

while (cap.isOpened()):
    ret, frame = cap.read()
    flipped = cv2.flip(frame, 1)
    cv2.imshow('stream', frame)
    cv2.imshow('flipped', flipped)
    key = cv2.waitKey(10)
    if key == 27:
        break;
    
cap.release()
cv2.destroyAllWindows()

#%%
# PART 4: gray scale

while (cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        flipped = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('stream', frame)
        cv2.imshow('flipped', flipped)
        cv2.imshow('gray', gray)
        key = cv2.waitKey(10)
        if key == 27:
            break;
    else:
        break;
    
cap.release()
cv2.destroyAllWindows()

#%%
# PART 5: write and create a video

import cv2
import numpy as np
cap = cv2.VideoCapture('stream.avi')

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
outVideo = cv2.VideoWriter('Out_stream.avi', fourcc, 20, (800, 600))

while (cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        newImg = cv2.merge((gray, gray, gray))
        
        outVideo.write(newImg)
        
        cv2.imshow('gray', gray)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()

#%% 
# PART 6: color tracking, color filtering

# blueball.mp4
import cv2
import numpy as np
cap = cv2.VideoCapture('blueball.mp4')

# google: color picker to find range
while (cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # better to present colors than RGB
        
        low_range = np.array([358, 90, 90])
        high_range = np.array([0, 90, 90])
        
        mask = cv2.inRange(hsv, low_range, high_range)
        
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        key = cv2.waitKey(10)
        if key == 27:
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()

#%% 

# PART 7: add and sum video frames

import cv2
import numpy as np

cap = cv2.VideoCapture('stream.avi')
ret1, frame1 = cap.read()
frame2 =frame1

while (cap.isOpened()):
    ret1, frame1 = cap.read()
    diff = cv2.absdiff(frame1, frame2)

#     cv2.imshow('frame1', frame1)
#     cv2.imshow('frame2', frame2)
    cv2.imshow('diff', diff)
    cv2.waitKey(10)
    frame2 =frame1
cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture('stream.avi')
ret1, frame1 = cap.read()
frame2 = frame1

while (cap.isOpened()):
    ret1, frame1 = cap.read()
    if ret1 == True:
        diff = cv2.absdiff(frame1, frame2)
#         cv2.imshow('frame1', frame1)
#         cv2.imshow('frame2', frame2)
        cv2.imshow('diff', diff)
        cv2.waitKey(10)
        frame2 = frame1
    else:
        break
cap.release()
cv2.destroyAllWindows()

#%% 
# PART 8: motion detection and tracking

import cv2
import numpy as np

cap = cv2.VideoCapture('stream.avi')
ret1, frame1 = cap.read()
frame2 = frame1

while (cap.isOpened()):
    ret1, frame1 = cap.read()
    frameCPY = frame1.copy()
    if ret1 == True:
        diff = cv2.absdiff(frame1, frame2)
        diffGray = cv2.cvtColor(diff, cv2.COLOR_BGR2BGRA)
        blur = cv2.GaussianBlur(diffGray, (9,9), 0)
        _, thre = cv2.threshold(diffGray, 50, 255, cv2.THRESH_BINARY)
        
        conts, _ = cv2.findContours(thre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frameCPY, conts, -1, (255, 0, 0), 3)
        
#         cv2.imshow('frame1', frame1)
#         cv2.imshow('frame2', frame2)
#         cv2.imshow('diff', diff)
        cv2.imshow('frameCPY', frameCPY)
        cv2.imshow('thre', thre)
        cv2.waitKey(10)
        frame2 = frame1
    else:
        break
cap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture('stream.avi')
ret1, frame1 = cap.read()
frame2 = frame1

while (cap.isOpened()):
    ret1, frame1 = cap.read()
    frameCPY = frame1.copy()
    if ret1 == True:
        diff = cv2.absdiff(frame1, frame2)
        diffGray = cv2.cvtColor(diff, cv2.COLOR_BGR2BGRA)
        blur = cv2.GaussianBlur(diffGray, (9,9), 0)
        _, thre = cv2.threshold(diffGray, 50, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thre, None, iteration = 3)
        conts, _ = cv2.findContours(thre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(frameCPY, conts, -1, (255, 0, 0), 3)
        
        for cont in conts:
            (x,y,w,h) = cv2.boundingRect(cont)
            if w > 100:
                cv2.rectangle(frameCPY, (x,y), (x+w, y+h), (255,0,0), 3)
        
#         cv2.imshow('frame1', frame1)
#         cv2.imshow('frame2', frame2)
#         cv2.imshow('diff', diff)
        cv2.imshow('frameCPY', frameCPY)
        cv2.imshow('thre', thre)
        cv2.waitKey(10)
        frame2 = frame1
    else:
        break
cap.release()
cv2.destroyAllWindows()


































 


















