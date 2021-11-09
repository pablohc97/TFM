#!/usr/bin/env python
# coding: utf-8

# # Libreries

# In[1]:


# Libreries
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
#import pandas as pd
from PIL import ImageOps, Image
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


# In[2]:


# Check tf version
print(tf.__version__)


# # Color information

# In[3]:


# From color_info.txt -> https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L52-L99

# Class names
names = ['unlabeled', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall',
         'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
         'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']

# Class colors
colors = np.array([(0, 0, 0), (111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), 
          (102, 102, 156), (190, 153, 153), (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), 
          (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), ( 0, 0, 142), ( 0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), 
          (0, 0, 142)], dtype = np.int32)


# # Load images

# In[4]:


# Define paths
train_images_folder_path = "./Data/Train_data/Images"  #"/content/drive/My Drive/TFM/Train_data/Images"
train_mask_folder_path = "./Data/Train_data/Labels"  #"/content/drive/My Drive/TFM/Train_data/Labels"
val_images_folder_path = "./Data/Validation_data/Images"  #"/content/drive/My Drive/TFM/Validation_data/Images"
val_mask_folder_path = "./Data/Validation_data/Labels"  #"/content/drive/My Drive/TFM/Validation_data/Labels"

# Get images and masks names
train_images_names = sorted([img for img in os.listdir(train_images_folder_path) if img.endswith('.png')])
train_mask_names = sorted([img for img in os.listdir(train_mask_folder_path) if img.endswith('.png')])
val_images_names = sorted([img for img in os.listdir(val_images_folder_path) if img.endswith('.png')])
val_mask_names = sorted([img for img in os.listdir(val_mask_folder_path) if img.endswith('.png')])

# Count check
print(f'Train images: {len(train_images_names)}')
print(f'Train labels: {len(train_mask_names)}')
print(f'Validation images: {len(val_images_names)}')
print(f'Validation labels: {len(val_mask_names)}')


# In[5]:


# Define image shape. Images are 1024 x 2048 (height x width) -> (896) 768 x 2048 after the crop, so to maintain the ratio: 
img_height, img_width = 96, 256  #112, 256
batch_size = 32

# Load functions:

def one_hot_mask(y):
    ''' Do the one hot encoding for the masks.
  
    Arguments:
        - y (tf tensor): Mask of shape (height, width, 3)

    Returns:
        - mask (tf tensor): Mask after do the one hot. Shape (height, width, num_classes) '''

    one_hot_map = []
    for color in colors:
        class_map = tf.reduce_all(tf.equal(y, color), axis = -1)
        one_hot_map.append(class_map)
    mask = tf.cast(tf.stack(one_hot_map, axis = -1), tf.int32)
    return mask



def load_image(folder, file, height = 96, width = 256, crop = True):
    ''' Load and preprocess a train image by:
        - Crop the image to not have the Mercedes-Benz star
        - Resize the image to (height, width)
        - Normalize the image to [0, 1]
  
    Arguments:
        - folder (string): Path to the folder
        - file (string): Name of the file to load
        - height (int): Height to resize -- 96
        - width (int): Width to resize -- 256
        - crop (bool): Crpo the image or not -- True

    Returns:
        - image (tf tensor): Preprocessed image '''

    # Load the image (png)
    image = tf.io.read_file(folder + '/' + file)
    image = tf.cast(tf.image.decode_png(image, channels = 3), tf.float32)

    # Crop the image
    if crop:
        image = tf.image.crop_to_bounding_box(image, 0, 0, 768, 2048)

    # Resize the image
    image = tf.image.resize(image, (height, width))

    # Normalize the image
    image = tf.cast(image, tf.float32)/255.0
    return image



def load_mask(folder, file, height = 96, width = 256, one_hot = True, crop = True):
    ''' Load and preprocess a train mask by:
        - Crop the image to not have the Mercedes-Benz star
        - Resize the image to (height, width)
        - Reshaping the mask from (height, width, 3) to (height, width, 30): One hot encoding
  
    Arguments:
        - folder (string): Path to the folder
        - file (string): Name of the file to load
        - height (int): Height to resize -- 96
        - width (int): Width to resize -- 256
        - one_hot (bool): Do one hot encoding or not -- True
        - crop (bool): Crpo the image or not -- True

    Returns:
        - image (tf tensor): Preprocessed mask '''

    # Load the mask (png)
    image = tf.io.read_file(folder + '/' + file)
    image = tf.cast(tf.image.decode_png(image, channels = 3), tf.int32)

    # Crop the mask
    if crop:
        image = tf.image.crop_to_bounding_box(image, 0, 0, 768, 2048)

    # Resize the mask
    image = tf.image.resize(image, (height, width), method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # One hot encoding
    if one_hot:
        image = one_hot_mask(image)
    return image



def load_train(image_name, mask_name):
    ''' Load and preprocess a train image and its mask
  
    Arguments:
        - image_name (string): Name of the image to load
        - mask_name (string): Name of the mask to load

    Returns:
        - image (tf tensor): Preprocessed image
        - mask (tf tensor): Preprocessed mask '''

    image = load_image(train_images_folder_path, image_name, img_height, img_width)
    mask = load_mask(train_mask_folder_path, mask_name, img_height, img_width)
    return image, mask



def load_val(image_name, mask_name):
    ''' Load and preprocess a validation image and its mask
  
    Arguments:
        - image_name (string): Name of the image to load
        - mask_name (string): Name of the mask to load

    Returns:
        - image (tf tensor): Preprocessed image
        - mask (tf tensor): Preprocessed mask '''

    image = load_image(val_images_folder_path, image_name, img_height, img_width)
    mask = load_mask(val_mask_folder_path, mask_name, img_height, img_width)
    return image, mask


# In[6]:


# Check mask shape after one hot encoding
print(load_mask(train_mask_folder_path, train_mask_names[0], img_height, img_width).shape)


# # Datasets creation

# In[7]:


# Create train tensorflow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images_names, train_mask_names))
train_dataset = train_dataset.map(load_train, num_parallel_calls = tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size)
print(train_dataset)

# Create validation tensorflow dataset
val_dataset = tf.data.Dataset.from_tensor_slices((val_images_names, val_mask_names))
val_dataset = val_dataset.map(load_val, num_parallel_calls = tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size)
print(val_dataset)

# Load the validation mask, one hot and get the correct value (96, 256, 1)
y_real_oh = np.argmax([load_mask(val_mask_folder_path, val_name, img_height, img_width, one_hot = True) for val_name in val_mask_names], axis = 3)


# # U-Net GDN model

# In[9]:


from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import NonNeg


# In[10]:


class GDN(Layer):
    def __init__(self, 
                 filter_shape = (3,3), 
                 **kwargs):
      
        self.filter_shape = filter_shape

        super(GDN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(name = 'beta', 
                                    shape = (input_shape.as_list()[-1]),
                                    initializer = tf.keras.initializers.constant(0.001),
                                    trainable = True,
                                    constraint = lambda x: tf.clip_by_value(x, 1e-15, np.inf))
        
        self.alpha = self.add_weight(name = 'alpha', 
                                     shape = (input_shape.as_list()[-1]),
                                     initializer = tf.keras.initializers.constant(2.0),
                                     trainable = True,
                                     constraint = NonNeg())

        self.epsilon = self.add_weight(name = 'epsilon', 
                                     shape = (input_shape.as_list()[-1]),
                                     initializer = tf.keras.initializers.constant(0.5),
                                     trainable = True,
                                     constraint = NonNeg())
        
        self.gamma = self.add_weight(name = 'gamma', 
                                     shape = (self.filter_shape[0], self.filter_shape[1], input_shape.as_list()[-1], input_shape.as_list()[-1]),
                                     initializer = tf.keras.initializers.Ones,
                                     trainable = True,
                                     constraint = NonNeg())
        
        
        super(GDN, self).build(input_shape)

    def call(self, x):
        norm_conv2 = tf.nn.convolution(tf.abs(x)**self.alpha,
                                      self.gamma,
                                      strides = (1, 1),
                                      padding = "SAME",
                                      data_format = "NHWC")

        norm_conv = self.beta + norm_conv2
        norm_conv = norm_conv**self.epsilon
        return x / norm_conv
        
    def compute_output_shape(self, input_shape):
        return (input_shape, self.output_dim)


# In[25]:


# Convolutional block function:

def conv_block(inputs, filters, pool = True):
    ''' Define a convolutional block of the encoder.
  
    Arguments:
        - inputs (tf tensor): Input tensor
        - filters (int): Number of filters
        - pool (bool): Do the MaxPooling or not -- True

    Returns:
        - x (tf tensor): Output tensor before the MaxPooling
        - p (tf tensor): Output tensor after the MaxPooling '''

    x = Conv2D(filters, 3, padding = "same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    if pool == True:
        p = MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x


# Model cration function:

def build_unet(shape, num_classes):
    ''' Build an UNET form model with GDN.
  
    Arguments:
        - shape ((int, int, int)): Shape of the input tensor
        - num_classes (int): Number of classes

    Returns:
        - model (tf model): Model built '''

    inputs = Input(shape)

    # Encoder
    g1 = GDN()(inputs)
    x1, p1 = conv_block(g1, 16, pool = True)
    g2 = GDN()(p1)
    x2, p2 = conv_block(g2, 32, pool = True)
    g3 = GDN()(p2)
    x3, p3 = conv_block(g3, 64, pool = True)
    g4 = GDN()(p3)
    x4, p4 = conv_block(g4, 128, pool = True)

    # Bridge
    b1 = conv_block(p4, 256, pool = False)

    # Decoder
    #u1 = UpSampling2D((2, 2), interpolation = "bilinear")(b1)
    u1 = Conv2DTranspose(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 128, pool = False)

    #u2 = UpSampling2D((2, 2), interpolation = "bilinear")(x5)
    u2 = Conv2DTranspose(128, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 64, pool = False)

    #u3 = UpSampling2D((2, 2), interpolation = "bilinear")(x6)
    u3 = Conv2DTranspose(64, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool = False)

    #u4 = UpSampling2D((2, 2), interpolation = "bilinear")(x7)
    u4 = Conv2DTranspose(32, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool = False)

    # Output
    output = Conv2D(num_classes, 1, padding = 'same', activation = 'softmax')(x8)

    return Model(inputs, output)


# In[26]:


# Build the model
model1 = build_unet((img_height, img_width, 3), 30)
print(model1.summary())


# # Train check

# In[27]:


# Train the model for a small number of epochs to check it works well
lr = 1e-4
epochs = 2

model1.compile(loss = 'categorical_crossentropy', 
               optimizer = tf.keras.optimizers.Adam(learning_rate = lr), 
               metrics = ['accuracy'])

histroy_model1 = model1.fit(train_dataset, 
                            epochs = epochs,
                            verbose = 1)


# In[28]:


plt.plot(histroy_model1.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(histroy_model1.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


# # Helpfull functions

# In[ ]:


# Prediction images functions:

def color_to_one_hot_mask(mask, colors, img_height = 96, img_width = 256):
    ''' Convert from the mask from the classes with highest probablities to the correct color. From (96, 256, 1) to (96, 256, 3).
  
    Arguments:
        - mask (tf tensor): Mask with the classes with highest probabilities
        - colors (list): List with the class colors
        - img_height (int): Height of the images -- 96
        - img_width (int): Width of the images -- 256

    Returns:
        - color_mask (tf tensor): Color mask '''

    color_mask = np.zeros((img_height, img_width, 3)).astype('float')
    for c in range(len(colors)):
        color_true = mask == c
        for i in range(3):
            color_mask[:,:,i] += color_true*colors[c][i]

    color_mask = tf.cast(color_mask, dtype = tf.int32)

    return color_mask


def visualize_image_mask(image_paths, mask_paths, preds, colors, img_height = 96, img_width = 256):
    ''' Display 5 random images, their masks and their predictions.
  
    Arguments:
        - image_paths (string list): Paths to images
        - mask_paths (string list): Paths to masks
        - preds (tf tensor): Model predictions
        - colors (list): Colors list
        - img_height (int): Height to reshape -- 96
        - img_width (int): Width to reshape -- 256  '''

    idx = np.random.choice(len(image_paths), 5)
    print('Val images: ', idx)

    images = [load_image(val_images_folder_path, image_paths[i], img_height, img_width) for i in idx]
    masks = [load_mask(val_mask_folder_path, mask_paths[i], img_height, img_width, one_hot = False) for i in idx]
    preds = preds[idx]

    plt.figure(figsize = (20, 5))
    plt.subplots_adjust(hspace = 0.05)

    for i in range(5):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[i])
        plt.subplot(3, 5, i + 6)
        plt.imshow(masks[i])
        plt.subplot(3, 5, i + 11)
        plt.imshow(color_to_one_hot_mask(preds[i], colors))

    plt.show()


# In[ ]:


# IOU function:

def iou_metrics(y_true, y_pred):
    ''' Calculate the intersection over union metric for each class for one image.
  
    Arguments:
        - y_true (tf tensor): Real mask the classes with highest probabilities (height, width, 1)
        - y_pred (tf tensor): Predicted mask the classes with highest probabilities (height, width, 1)

    Returns:
        - class_iou (list): Iou for each class '''

    class_iou = []
    smoothening_factor = 0.00001

    for i in range(len(names)):
        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area

        iou = (intersection + smoothening_factor) / (combined_area - intersection + smoothening_factor)
        if iou < 10**(-5):
            iou = 0
        class_iou.append(iou)

    return class_iou


# # Prediction check

# In[ ]:


# Predictions
preds = model1.predict(val_dataset, batch_size = 16)
preds = np.argmax(preds, axis = 3)

# Visualize predictions
visualize_image_mask(val_images_names, val_mask_names, preds, colors, img_height = 96, img_width = 256)


# In[ ]:


# IOU per class
iou = []
for i in range(len(preds)):
    iou.append(iou_metrics(y_real_oh[i], preds[i]))
    
print(pd.DataFrame({'Names': names, 'IOU': [iou for iou in np.mean(iou, axis = 0)]}))

# Mean IOU
print('Mean IOU: ', np.mean(iou))


# # Hyperparameter turning

# In[ ]:


# Dataset
train_dataset_hpt = tf.data.Dataset.from_tensor_slices((train_images_names[:1000], train_mask_names[:1000]))
train_dataset_hpt = train_dataset_hpt.map(load_train, num_parallel_calls = tf.data.experimental.AUTOTUNE)
train_dataset_hpt = train_dataset_hpt.batch(batch_size)
print(train_dataset_hpt)


# In[ ]:


# Models
model_hpt_1 = build_unet((img_height, img_width, 3), 30)
model_hpt_2 = build_unet((img_height, img_width, 3), 30)
model_hpt_3 = build_unet((img_height, img_width, 3), 30)

lr_1, lr_2, lr_3 = 1e-5, 1e-4, 1e-3
epochs = 20

model_hpt_1.compile(loss = 'categorical_crossentropy', 
                    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_1), 
                    metrics = ['accuracy'])

model_hpt_2.compile(loss = 'categorical_crossentropy', 
                    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_2), 
                    metrics = ['accuracy'])

model_hpt_3.compile(loss = 'categorical_crossentropy', 
                    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_3), 
                    metrics = ['accuracy'])


# In[ ]:


# Train different learning rates

histroy_model_hpt_1 = model_hpt_1.fit(train_dataset_hpt, 
                                      epochs = epochs,
                                      verbose = 1)

histroy_model_hpt_2 = model_hpt_2.fit(train_dataset_hpt, 
                                      epochs = epochs,
                                      verbose = 1)

histroy_model_hpt_3 = model_hpt_3.fit(train_dataset_hpt, 
                                      epochs = epochs,
                                      verbose = 1)

# Training plots

plt.plot(histroy_model_hpt_1.history['accuracy'])
plt.plot(histroy_model_hpt_2.history['accuracy'])
plt.plot(histroy_model_hpt_3.history['accuracy'])
plt.legend(['-5', '-4', '-3'], loc = 'upper left')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(histroy_model_hpt_1.history['loss'])
plt.plot(histroy_model_hpt_2.history['loss'])
plt.plot(histroy_model_hpt_3.history['loss'])
plt.legend(['-5', '-4', '-3'], loc = 'upper left')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


# # Train

# In[ ]:


lr = 1e-4
epochs = 200

checkpoint_filepath = './tmp_gdn/weights.{epoch:02d}-{accuracy:.2f}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath, 
                                                               save_weights_only = True, 
                                                               monitor = 'accuracy',
                                                               mode = 'max',
                                                               save_best_only = True,
                                                               verbose = 0)

reduce_lr = ReduceLROnPlateau(monitor = 'loss', 
                              factor = 0.1,
                              patience = 5, 
                              min_lr = 1e-6,
                              verbose = 1)

model1.compile(loss = 'categorical_crossentropy', 
               optimizer = tf.keras.optimizers.Adam(learning_rate = lr), 
               metrics = ['accuracy'])

history_model1 = model1.fit(train_dataset, 
                            epochs = epochs,
                            verbose = 1,
                            callbacks = [model_checkpoint_callback, reduce_lr])


# In[ ]:


# Training plots
plt.plot(history_model1.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(history_model1.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


# In[ ]:


# Restore best weights
model1.load_weights(checkpoint_filepath)

# Predictions
preds = model1.predict(val_dataset)
preds = np.argmax(preds, axis = 3)

# Visualize predictions
visualize_image_mask(val_images_names, val_mask_names, preds, colors, img_height = 96, img_width = 256)


# In[ ]:


# IoU
iou = []
for i in range(len(preds)):
    iou.append(iou_metrics(y_real_oh[i], preds[i]))
    
# Iou per class
print(pd.DataFrame({'Names': names, 'IOU': [iou for iou in np.mean(iou, axis = 0)]}))

# Mean IoU
print('Mean IOU: ', np.mean(iou))

