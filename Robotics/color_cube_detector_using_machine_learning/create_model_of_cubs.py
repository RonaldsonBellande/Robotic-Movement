#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cairo


# In[23]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
from livelossplot.keras import PlotLossesCallback
import efficientnet.keras as efn
import keras
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import pickle
import sys
import numpy as np
from keras import optimizers
import os
from mlxtend.plotting import plot_confusion_matrix
import random
from PIL import Image
from contextlib import redirect_stdout
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


# In[24]:


image_width, image_height = 200, 200
epochs = 0
cur_epoch = 200
batch_size = 8
test_size = 30
input_shape = (image_width, image_height, 3)
file_path_train = r"/home/ronaldsonbellande/Desktop/mobile_robotics/personal_repository/color_cube_detector_using_machine_learning/training_image_cubes/"
file_path_validation = r"/home/ronaldsonbellande/Desktop/mobile_robotics/personal_repository/color_cube_detector_using_machine_learning/validation/"
file_path_test = r"/home/ronaldsonbellande/Desktop/mobile_robotics/personal_repository/color_cube_detector_using_machine_learning/testing/"


# In[25]:


model = Sequential()
model = Sequential()

model.add(Conv2D(32, 3, 3, padding='same', input_shape=input_shape, activation='relu'))
model.add(Conv2D(32, 3, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, padding='same', activation='relu'))
model.add(Conv2D(64, 3, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, 3, padding='same', activation='relu'))
model.add(Conv2D(128, 3, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.0001),
            metrics=['accuracy'])

# model.compile(loss='binary_crossentropy',
#             optimizer=RMSprop(lr=0.0001),
#             metrics=['accuracy'])


# In[26]:


model_summary = model.summary()
with open('model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()


# In[27]:


scale = 1./255
training_data_image_classification = ImageDataGenerator(rescale= scale, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
validation_data_image_classification = ImageDataGenerator(rescale= scale)
test_data_image_classification = ImageDataGenerator(rescale= scale)


# In[28]:


training_generator = training_data_image_classification.flow_from_directory(file_path_train,
    target_size=(image_width, image_height),
    batch_size=batch_size, class_mode="binary")


# In[41]:


print(training_generator)


# In[29]:


validation_generator = validation_data_image_classification.flow_from_directory(file_path_validation,
    target_size=(image_width, image_height),
    batch_size=batch_size, class_mode="binary")


# In[30]:


test_generator = test_data_image_classification.flow_from_directory(file_path_test,
    target_size=(image_width, image_height),
    batch_size=1,
    class_mode="binary", 
    shuffle=False)


# In[31]:


for data_batch, labels_batch in training_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


# In[32]:


class_names = training_generator.class_indices
print(class_names)


# In[33]:


checkpoint = keras.callbacks.EarlyStopping(monitor='val_acc', patience=4, verbose=1)


# In[34]:


trained_model = model.fit(
    training_generator,
    steps_per_epoch=len(training_generator.filenames)//training_generator.batch_size,
    initial_epoch=epochs,
    epochs=epochs + cur_epoch,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames)//validation_generator.batch_size,
    callbacks=[checkpoint],
    shuffle = True, verbose=1)


# In[35]:


model.save_weights("CNN_cube_detector.h5")


# In[36]:


xlabel = 'Epoch'
legends = ['Training', 'Validation']
ylim_pad = [0.01, 0.1]
plt.figure(figsize=(15, 5))
y1 = trained_model.history['accuracy']
y2 = trained_model.history['val_accuracy']

min_y = min(min(y1), min(y2))-ylim_pad[0]
max_y = max(max(y1), max(y2))+ylim_pad[0]


plt.subplot(121)
plt.plot(y1)
plt.plot(y2)
plt.title('Model Accuracy', fontsize=17)
plt.xlabel(xlabel, fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.ylim(min_y, max_y)
plt.legend(legends, loc='upper left')
plt.grid()
    
y1 = trained_model.history['loss']
y2 = trained_model.history['val_loss']
min_y = min(min(y1), min(y2))-ylim_pad[1]
max_y = max(max(y1), max(y2))+ylim_pad[1]
    
plt.subplot(122)
plt.plot(y1)
plt.plot(y2)

plt.title('Model Loss', fontsize=17)
plt.xlabel(xlabel, fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.ylim(min_y, max_y)
plt.legend(legends, loc='upper left')
plt.grid()                      
plt.show()


# In[37]:


print("results")
result  = model.evaluate(test_generator, steps=len(test_generator), verbose=2)

print("%s%.2f  "% ("Loss     : ", result[0]))
print("%s%.2f%s"% ("Accuracy : ", result[1]*100, "%"))


# In[38]:


classes = test_generator.class_indices
classes


# In[40]:


y_pred = model.predict(test_generator, steps=len(test_generator), verbose=1)
# y_pred = y_pred.argmax(axis=-1)
y_pred = y_pred.reshape(1,9)
y_pred = y_pred[0]
y_true=test_generator.classes
print(y_pred)


# In[19]:


#metrics.f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))


# In[20]:


image_file_name = "/home/ronaldsonbellande/Desktop/mobile_robotics/personal_repository/color_cube_detector_using_machine_learning/"

file_path = 'CNN_cube_detector.h5'
title = file_path.split("/")
model_title = "/".join([i for i in title[3:]])

precision = precision_score(y_true, y_pred, average="weighted") 
recall = recall_score(y_true, y_pred, average="weighted") 
f1 = f1_score(y_true, y_pred, average="weighted") 

print("-"*90)
print("Derived Report")
print("-"*90)
print("%s%.2f%s"% ("Precision     : ", precision*100, "%"))
print("%s%.2f%s"% ("Recall        : ", recall*100,    "%"))
print("%s%.2f%s"% ("F1-Score      : ", f1*100,        "%"))
print("-"*90)
print("\n\n")

CM = confusion_matrix(y_true*10, y_pred*10)
fig, ax = plot_confusion_matrix(conf_mat=CM , figsize=(10,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(len(classes)), classes, fontsize=10)
plt.yticks(range(len(classes)), classes, fontsize=10)
plt.title("Confusion Matrix for Model File (Test Dataset): \n"+model_title, fontsize=10)
fig.savefig(image_file_name, dpi=100)
plt.show()
    

cls_report_print = classification_report(y_true, y_pred, target_names=classes)

cls_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

print("\n\n")
print("-"*90)
print("Report for Model File: ", model_title)
print("-"*90)
print(cls_report_print)
print("-"*90)


# In[ ]:





# In[ ]:





# In[ ]:




