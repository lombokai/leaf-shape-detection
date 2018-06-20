import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import datetime

import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, Reshape, GlobalAveragePooling2D
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard

import skimage
from skimage import io, color, transform

# Mendefinisikan Fungsi
def set_gpu_mem_alloc(mem_use):
    avail  = 2004
    percent = mem_use / avail
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = percent
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

set_gpu_mem_alloc(1500)

# print("Success import all library!")

batch_size = 16
h,w,d = 224,224,3
epochs = 10

date_str = datetime.datetime.now().strftime("%d%m%y_%H%M")


train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(h, w),
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(h, w),
        class_mode='categorical')

x_train, y_train = train_generator.next()
x_test, y_test = test_generator.next()


num_class = len(train_generator.class_indices)
train_step = len(train_generator.classes)
valid_step = len(test_generator.classes)

print('NUM Classes => ', num_class)
print('Train Step  => ',  train_step)
print('Valid Step  => ',  valid_step)

ksize = (3,3)
psize = (2,2)

input_shape = (h,w,d)

model = Sequential()
model.add(Conv2D(32,kernel_size=ksize, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=psize))
# model.add(Flatten())
model.add(Conv2D(32,kernel_size=ksize, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=psize))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(num_class, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc','mse'])

# untuk liat log history
tb1_path = 'log/'+date_str
pathlib.Path(tb1_path).mkdir(parents=True, exist_ok=True)
#training
tensorboard = TensorBoard( log_dir=tb1_path, histogram_freq=0,
                            write_graph=True, write_images=True)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_step // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=valid_step // batch_size,
    callbacks=[tensorboard],
    verbose=2)


# serialize model to JSON
model_json = model.to_json()
with open("saved/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("saved/model.h5")
print("Saved model to disk")

# x, y = test_generator.next()
# img = x[0] * 255
# img = img.astype("uint8")
# plt.imshow(img, cmap = 'gray')
# plt.show()

# for yy in y:
#     print(np.argmax(yy)+1)

# pred = model.predict(x)
# for pr in pred:
#     print(np.argmax(pr)+1)