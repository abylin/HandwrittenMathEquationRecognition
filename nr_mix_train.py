import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
import os
import cv2
from sklearn import preprocessing
from pathlib import Path
from PIL import Image

symbols_list = ['zero','one','two','three','four','five','six','seven','eight','nine','minus','plus','equal','div','decimal','times', 'left', 'right']

# Get the training dataset. Data set is too large to be placed on GitHub
train_path = "./dataset/train"
train_image = []
train_label = []
for symbols_dir in os.listdir(train_path):
    if symbols_dir.split()[0] in symbols_list:
        for image in os.listdir(train_path + "/" + symbols_dir):
            train_label.append(symbols_dir.split()[0])
            train_image.append(train_path + "/" + symbols_dir + "/" + image)
# Since Keras is split first and then randomized, the training data set needs to be scrambled first
index = np.arange(len(train_image))
np.random.shuffle(index)
train_image=np.array(train_image)[index]
train_label=np.array(train_label)[index]

# Get the eval dataset. Data set is too large to be placed on GitHub
eval_path = "./dataset/eval"
test_image = []
test_label = []
for symbols_dir in os.listdir(eval_path):
    if symbols_dir.split()[0] in symbols_list:
        for image in os.listdir(eval_path + "/" + symbols_dir):
            label1 = symbols_dir.split()[0]
            path1 = eval_path + "/" + symbols_dir + "/" + image
            test_label.append(symbols_dir.split()[0])
            test_image.append(eval_path + "/" + symbols_dir + "/" + image)

X_train = []
X_test = []

# loading the images
for path in train_image:    
    img = cv2.imread(path, 0) # Load the image with grayscale
    img = cv2.resize(img, (100, 100))
    img = np.array(img)
    img = img.reshape(100, 100, 1)
    X_train.append(img)
for path in test_image:    
    img = cv2.imread(path, 0) # Load the image with grayscale
    img = cv2.resize(img, (100, 100))
    img = np.array(img)
    img = img.reshape(100, 100, 1)   
    X_test.append(img)

X_train = np.array(X_train) # (7557, 100, 100, 3)
X_test = np.array(X_test) # (1010, 100, 100, 3)
print(X_test.shape)

label_encoder = preprocessing.LabelEncoder()
y_train_temp = label_encoder.fit_transform(train_label) 
y_test_temp = label_encoder.fit_transform(test_label)

print("y_train_temp shape: ", y_train_temp.shape) #(7557,)
print("y_test_temp shape: ", y_test_temp.shape) #  (1010,)

y_train = keras.utils.to_categorical(y_train_temp, 18)
y_test = keras.utils.to_categorical(y_test_temp, 18)
print("y_train shape: ", y_train.shape) # (7557, 16)
print("y_test shape: ", y_test.shape)  # (1010, 16)

# using sequential model for training
model = Sequential()

# each grayscale image is 100x100x1
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(100, 100, 1), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# prevent overfitting
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))

# last layer predicts 16 labels
model.add(Dense(18, activation="softmax"))

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)
model.summary()

# save the image of model
keras.utils.plot_model(model, "model.png", show_shapes=True)

# training the model 
history = model.fit(
    X_train,
    y_train,
    batch_size = 64,
    epochs = 50,
    validation_split = 0.15,
    shuffle=True
)
model.save('mix2.h5')
print("Saved the model as mix2.h5")

# displaying the model accuracy
plt.plot(history.history['accuracy'], label='train', color="red")
plt.plot(history.history['val_accuracy'], label='validation', color="blue")
plt.title('Model accuracy')
plt.legend(loc='upper left')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# displaying the model loss
plt.plot(history.history['loss'], label='train', color="red")
plt.plot(history.history['val_loss'], label='validation', color="blue")
plt.title('Model loss')
plt.legend(loc='upper left')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

score, acc = model.evaluate(X_test, y_test)
print('Test score:', score)
print('Test accuracy:', acc)

score, acc = model.evaluate(X_train, y_train)
print('Train score:', score)
print('Train accuracy:', acc)

pred = model.predict(X_test)

# Use some examples to view the predictions
fig, axs= plt.subplots(2,5, figsize=[24,12])
count=0
for i in range(2):    
    for j in range(5):  
        image = cv2.imread(test_image[count + count*76], 0) 
        img = cv2.resize(image, (100, 100)) 
        img = np.array(img)
        img = img.reshape(100, 100, 1)       
        img = np.expand_dims(img, axis=0)
          
        pred = model.predict(img)  
        result = np.argsort(pred)   
        result = result[0][::-1] 

        final_label = label_encoder.inverse_transform(np.array(result))      
        axs[i][j].imshow(image, cmap='gray')
        axs[i][j].set_title(str("Prediction: " + final_label[0]), fontsize = 14)
        
        count += 1
        
plt.suptitle("the prediction exsamples", fontsize = 18)        
plt.show()