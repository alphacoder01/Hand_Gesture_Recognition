import cv2
import numpy as np

cap = cv2.VideoCapture(0)
count = 0
delay= 0
while True:
    ret,frame = cap.read()
    delay += 1
    cv2.rectangle(frame,(0,416),(224,224),(0,255,0),2)
    crop_img = frame[192:416,0:224]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3,), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    cv2.imshow('sure_bg',sure_bg)
    ret, final = cv2.threshold(sure_bg,0,255,cv2.THRESH_BINARY_INV)    
    cv2.imshow('frame',frame)
    # cv2.imshow('needed',crop_img)
    final = cv2.GaussianBlur(final,(5,5),0)
    cv2.imshow('un',final)
    if delay%5 ==0:
                # create the directory containing sub-dir for that person's image
            cv2.imwrite('G:/open cv/Hand_Gesture/Hand_data/Test_data/%s.jpg'%str(count),final)
            count += 1



    if cv2.waitKey(5) & 0xff == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()