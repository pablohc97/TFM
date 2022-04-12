# Load libreries, set the seed and select the GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

seed = 123

import tensorflow as tf
tf.random.set_seed(seed)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)

from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Softmax, ReLU
from tensorflow.keras import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageEnhance, Image
from tensorflow.keras.optimizers import Adam
from GDN import *
import matplotlib
matplotlib.use('Agg')

# Load data
(X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = seed)

# Define modification functions
# Change luminance
def change_lum_smooth(image):
    return_image = image.copy()
    limit = random.uniform(2, 4)
    vals = 0.1 + 1/(1 + np.exp(-np.linspace(-limit, limit, image.shape[0])))
    orientation = random.random()
    direction = random.random()
    if direction < 0.5:
        vals = np.array(list(vals)[::-1])
    if orientation < 0.5:
        for i in range(image.shape[2]):
            for j in range(image.shape[0]):
                return_image[:,j,i] = return_image[:,j,i]*vals
    if orientation >= 0.5:
        for i in range(image.shape[2]):
            for j in range(image.shape[0]):
                return_image[j,:,i] = return_image[j,:,i]*vals
    return tf.clip_by_value(return_image, 0, 255)

# Change contrast
def change_contrast_smooth(image):
    return_image = image.copy()
    limit = random.uniform(2, 4)
    vals = 0.5 + 1/(1 + np.exp(-np.linspace(-limit, limit, image.shape[0])))
    orientation = random.random()
    direction = random.random()
    if direction < 0.5:
        vals = np.array(list(vals)[::-1])
    if orientation < 0.5:
        for i in range(image.shape[2]):
            for j in range(image.shape[0]):
                enhancer = ImageEnhance.Contrast(Image.fromarray(return_image[:,j,i]))
                return_image[:,j,i] = np.array(enhancer.enhance(vals[j])).reshape((32))
    if orientation >= 0.5:
        for i in range(image.shape[2]):
            for j in range(image.shape[0]):
                enhancer = ImageEnhance.Contrast(Image.fromarray(return_image[j,:,i]))
                return_image[j,:,i] = np.array(enhancer.enhance(vals[j])).reshape((32))
    return tf.clip_by_value(return_image, 0, 255)

# Change luminance, contrast or leave as original
def random_lum_contrast_smooth(image):
    random_val = random.random()
    if random_val < 1/3:
        return change_lum_smooth(image)
    elif random_val >= 1/3 and random_val < 2/3:
        return change_contrast_smooth(image)
    else:
        return image

X_train_new, X_val_new, X_test_new = [], [], []

for i in range(X_train.shape[0]):
    X_train_new.append(random_lum_contrast_smooth(X_train[i]))

for i in range(X_val.shape[0]):
    X_val_new.append(random_lum_contrast_smooth(X_val[i]))

for i in range(X_test.shape[0]):
    X_test_new.append(random_lum_contrast_smooth(X_test[i]))

# Modify and normalize the data
X_train_modified = np.array(X_train_new)/255.0
X_val_modified = np.array(X_val_new)/255.0
X_test_modified = np.array(X_test_new)/255.0
X_train_original, X_val_original, X_test_original = X_train/255.0, X_val/255.0, X_test/255.0

# One hot encoding for the labels
Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)
Y_test = to_categorical(Y_test)

# Check data shapes
print(X_train_modified.shape, Y_train.shape, X_val_modified.shape, Y_val.shape, X_test_modified.shape, Y_test.shape)
print(X_train_original.shape, Y_train.shape, X_val_original.shape, Y_val.shape, X_test_original.shape, Y_test.shape)

# Build the different models
# Model with 3 GDN layers
print('Model with 3 GDN layers')
model_3_GDN = Sequential()
model_3_GDN.add(GDN(input_shape = X_train.shape[1:4], filter_shape = (7,7)))
model_3_GDN.add(Conv2D(8, 3, padding  = 'same', activation = 'relu'))
model_3_GDN.add(AveragePooling2D())
model_3_GDN.add(GDN(filter_shape = (5,5)))
model_3_GDN.add(Conv2D(16, 3, padding  = 'same', activation = 'relu'))
model_3_GDN.add(AveragePooling2D())
model_3_GDN.add(GDN(filter_shape = (3,3)))
model_3_GDN.add(Conv2D(32, 3, padding  = 'same', activation = 'relu'))
model_3_GDN.add(AveragePooling2D())
model_3_GDN.add(Flatten())
model_3_GDN.add(Dense(10))
model_3_GDN.add(Softmax())
model_3_GDN.summary()

# Model with 1 GDN layers
print('Model with 1 GDN layers')
model_1_GDN = Sequential()
model_1_GDN.add(GDN(input_shape = X_train.shape[1:4], filter_shape = (7,7)))
model_1_GDN.add(Conv2D(8, 3, padding  = 'same', activation = 'relu'))
model_1_GDN.add(AveragePooling2D())
model_1_GDN.add(Conv2D(16, 3, padding  = 'same', activation = 'relu'))
model_1_GDN.add(AveragePooling2D())
model_1_GDN.add(Conv2D(32, 3, padding  = 'same', activation = 'relu'))
model_1_GDN.add(AveragePooling2D())
model_1_GDN.add(Flatten())
model_1_GDN.add(Dense(10))
model_1_GDN.add(Softmax())
model_1_GDN.summary()

# Model with NO GDN layers
print('Model with no GDN layers')
model_no_GDN = Sequential()
model_no_GDN.add(Conv2D(8, 3, padding  = 'same', activation = 'relu', input_shape = X_train.shape[1:4]))
model_no_GDN.add(AveragePooling2D())
model_no_GDN.add(Conv2D(16, 3, padding  = 'same', activation = 'relu'))
model_no_GDN.add(AveragePooling2D())
model_no_GDN.add(Conv2D(32, 3, padding  = 'same', activation = 'relu'))
model_no_GDN.add(AveragePooling2D())
model_no_GDN.add(Flatten())
model_no_GDN.add(Dense(10))
model_no_GDN.add(Softmax())
model_no_GDN.summary()

# Hyperparameters for training the models and data generator
batch_size = 32
epochs = 2
datagen = ImageDataGenerator(width_shift_range = 0.15,  
                             height_shift_range = 0.15, 
                             horizontal_flip = True)
train_dataset = datagen.flow(X_train_modified, Y_train, batch_size = batch_size)
steps = int(X_train_modified.shape[0]/batch_size)

lr_3_gdn = 1e-3
lr_1_gdn = 1e-3
lr_no_gdn = 1e-3

# Reduce learning rate callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy', 
                                                 factor = 0.5, 
                                                 patience = 100, 
                                                 min_lr = 1e-5, 
                                                 verbose = 1)

# Save weights callback
checkpoint_path_3_gdn = './Classification/3_gdn.hdf5'
checkpoint_path_1_gdn = './Classification/1_gdn.hdf5'
checkpoint_path_no_gdn = './Classification/No_gdn.hdf5'
checkpoint_model_3_gdn = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path_3_gdn,
                                                            save_weights_only = True,
                                                            monitor = 'val_accuracy',
                                                            mode = 'max',
                                                            save_best_only = True,
                                                            verbose = 0)
checkpoint_model_1_gdn = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path_1_gdn,
                                                            save_weights_only = True,
                                                            monitor = 'val_accuracy',
                                                            mode = 'max',
                                                            save_best_only = True,
                                                            verbose = 0)
checkpoint_model_no_gdn = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path_no_gdn,
                                                             save_weights_only = True,
                                                             monitor = 'val_accuracy',
                                                             mode = 'max',
                                                             save_best_only = True,
                                                             verbose = 0)

# Compile and train the models
model_3_GDN.compile(loss = 'categorical_crossentropy', 
                    optimizer = Adam(learning_rate = lr_3_gdn), 
                    metrics = ['accuracy'])
model_1_GDN.compile(loss = 'categorical_crossentropy', 
                    optimizer = Adam(learning_rate = lr_1_gdn), 
                    metrics = ['accuracy'])
model_no_GDN.compile(loss = 'categorical_crossentropy', 
                     optimizer = Adam(learning_rate = lr_no_gdn), 
                     metrics = ['accuracy'])

history_3_gdn = model_3_GDN.fit(train_dataset, 
                                epochs = epochs, 
                                batch_size = batch_size, 
                                verbose = 2,
                                callbacks = [reduce_lr, checkpoint_model_3_gdn],
                                validation_data = (X_val, Y_val))
history_1_gdn = model_1_GDN.fit(train_dataset, 
                                epochs = epochs, 
                                batch_size = batch_size, 
                                verbose = 2,
                                callbacks = [reduce_lr, checkpoint_model_1_gdn],
                                validation_data = (X_val, Y_val))
history_no_gdn = model_no_GDN.fit(train_dataset, 
                                  epochs = epochs, 
                                  verbose = 2,
                                  batch_size = batch_size, 
                                  callbacks = [reduce_lr, checkpoint_model_no_gdn],
                                  validation_data = (X_val, Y_val))                                                                                  

# Save training histories
np.save('./Classification/history_3_gdn.npy', history_3_gdn.history)
np.save('./Classification/history_1_gdn.npy', history_1_gdn.history)
np.save('./Classification/history_no_gdn.npy', history_no_gdn.history)

# Plot and save training curves
plt.plot(history_3_gdn.history['accuracy'])
plt.plot(history_1_gdn.history['accuracy'])
plt.plot(history_no_gdn.history['accuracy'])
plt.title('Model accuracy on train')
plt.ylabel('Train accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(['3 GDN layers', '1 GDN layer', 'No GDN layers'], loc = 'lower right')
plt.savefig('./Classification/Train_accuracy.png')
plt.show()
plt.clf()   

plt.plot(history_3_gdn.history['val_accuracy'])
plt.plot(history_1_gdn.history['val_accuracy'])
plt.plot(history_no_gdn.history['val_accuracy'])
plt.title('Model accuracy on validation')
plt.ylabel('Val accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(['3 GDN layers', '1 GDN layer', 'No GDN layers'], loc = 'lower right')
plt.savefig('./Classification/Validation_accuracy.png')
plt.show()
plt.clf() 

plt.plot(history_3_gdn.history['loss'])
plt.plot(history_1_gdn.history['loss'])
plt.plot(history_no_gdn.history['loss'])
plt.title('Model loss on train')
plt.ylabel('Train loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(['3 GDN layers', '1 GDN layer', 'No GDN layers'], loc = 'upper right')
plt.savefig('./Classification/Train_loss.png')
plt.show()
plt.clf()   

plt.plot(history_3_gdn.history['val_loss'])
plt.plot(history_1_gdn.history['val_loss'])
plt.plot(history_no_gdn.history['val_loss'])
plt.title('Model loss on validation')
plt.ylabel('Val loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend(['3 GDN layers', '1 GDN layer', 'No GDN layers'], loc = 'upper right')
plt.savefig('./Classification/Validation_loss.png')
plt.show()
plt.clf() 