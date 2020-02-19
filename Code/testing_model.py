from tensorflow.keras.models import load_model
import webbrowser
import numpy as np
import cv2
import tensorflow as tf
import os
from PIL import ImageGrab

CATEGORIES = ['Video_Player','Browser','Youtube','Close','Vol_Up','Vol_Down']

model = load_model('G:/open cv/Hand_Gesture/Artifacts/my_model_more_epochs_50_new_metrics.h5')
print('Model loaded')

def prepare(filepath):
    IMG_SIZE = 224
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img_array.reshape(-1, IMG_SIZE,IMG_SIZE,1)

def prepare_frame(img):
    IMG_SIZE = 224
    # img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    return img.reshape(-1, IMG_SIZE,IMG_SIZE,1)

C = 0
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vid = cv2.VideoWriter('output.avi',fourcc,6,(640,480))
while True:
    ret, frame = cap.read()
    cv2.rectangle(frame,(0,416),(224,224),(0,255,0),2)
    crop_img = frame[192:416,0:224]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3,), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # cv2.imshow('sure_bg',sure_bg)
    ret, final = cv2.threshold(sure_bg,0,255,cv2.THRESH_BINARY_INV)    
   
    # cv2.imshow('needed',crop_img)
    final = cv2.GaussianBlur(final,(5,5),0)
    # cv2.imshow('un',final)
    prediction = model.predict([prepare_frame(final)])
    text = str((CATEGORIES[int(np.argmax(prediction[0]))]))
    print(text)
    
    if text == 'Youtube':
        C +=1
        if C >=20:
            webbrowser.open("https://www.youtube.com/")
            C = 0
    if text =='Browser':
        C +=1
        if C >=20:
            webbrowser.open('https:/google.com/')
            time.sleep(5)
            C = 0
    if text == 'Close':
        C +=1
        if C >=50:
            browserExe = "chrome.exe"
            os.system("taskkill /f /im "+browserExe)
            C =0

    cv2.putText(frame,text,(200,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    
    vid.write(frame)
    if cv2.waitKey(5) & 0xff == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

# for file in os.listdir('G:/open cv/Hand_Gesture/Hand_data/Test_data/'):
#     print(file)
#     new_path = 'G:/open cv/hand_Gesture/Hand_data/Test_data/'+file
#     prediction = model.predict([prepare(new_path)])
#     print(CATEGORIES[int(np.argmax(prediction[0]))])
