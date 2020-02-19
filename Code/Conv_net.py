# import the necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,MaxPool2D
import pickle
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# load the generate data from pickle files.
X = pickle.load(open('X_more.pickle','rb'))
y = pickle.load(open('y_more.pickle','rb'))
# Covert the labels to one-hot encoding 
y = to_categorical(y,num_classes=6)
X = X/255.0

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

# Define the Keras model.
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size=(5,5), padding = 'Same',activation = 'relu',input_shape = X.shape[1:]))
model.add(Conv2D(filters = 32, kernel_size=(5,5), padding = 'Same',activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation='relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(6,activation = 'softmax'))
optimizer = RMSprop(lr=0.001,rho = 0.9,epsilon=1e-08,decay=0.0)
model.compile(optimizer=optimizer, loss = 'categorical_crossentropy',metrics=['accuracy','categorical_accuracy'])

epochs= 50
batch_size = 32
# Use ImageDataGenerator for augmenting the train data
datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,
featurewise_std_normalization=False,samplewise_std_normalization=False,zca_whitening=False,rotation_range=10,zoom_range=0.1,width_shift_range=0.1, height_shift_range=0.1,horizontal_flip=False,vertical_flip=False)

datagen.fit(X_train)

history = model.fit(datagen.flow(X_train,y_train,batch_size= batch_size),epochs = epochs,
                   validation_data=(X_test,y_test),verbose = 2, steps_per_epoch=X_train.shape[0]//batch_size)

# Plot the loss and accuray graphs.
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.legend(['Train','Test'],loc = 'upper left')
plt.savefig('Accuracy.png')


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.legend(['Train','Test'],loc='upper left')
plt.savefig('Loss.png')

# Save the model.
model.save('my_model_more_epochs_50_new_metrics.h5')
