import numpy as np
import matplotlib.pyplot as pyt
import os
import cv2

DATADIR='G:/open cv/Hand_Gesture/Hand_data'
CATEGORIES = ['Video_Player','Browser','Youtube','Close','Vol_Up','Vol_Down']
training_data = []
def create_training_data():
    for cat in CATEGORIES:
        path = os.path.join(DATADIR,cat)
        class_num = CATEGORIES.index(cat)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            training_data.append([img_array,class_num])

create_training_data()
print(len(training_data))


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X= np.array(X).reshape(-1,224,224,1)

import pickle
pickle_out = open('X_more.pickle','wb')
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open('y_more.pickle','wb')
pickle.dump(y,pickle_out)
pickle_out.close()

print('Done!')