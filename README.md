# Hand_Gesture_Recognition
Contains reproducible code for Hand Gesture recognizer.

*Dependencies:
  1. Tensorflow-gpu=1.14
  2. Keras=2.2.4
  3. Open cv
  4. Python=3.6
  5. PIL
  
*Code Directory contains the file as follows:
  1. **Conv_net.py** : Contains the code to train the model on the processed data.
  2. **getting_data.py** : Using open cv, and numpy capture the frames of the webcam and generate the dataset.
  3. **process_data.py** : Helps in processing the images to specified shape, and stores them as numpy array in  pickle file.
  4.**testing_model.py** : Can be used for realtime prediction as well as inference on static images of specified format.
  
*Results Directory contains the accuracy and loss plots of the trained model, achieving over **99% accuracy on both train and test sets**.

*Pretrained model file as well as pickle file to be uploaded soon.
