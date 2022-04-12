# Load libreries and select GPU
import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from PIL import ImageOps, Image
from sklearn.metrics import jaccard_score
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import keras
from GDN import *
import matplotlib
matplotlib.use('Agg')
print(tf.__version__)
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define paths
train_images_folder_path = "./Data/Train_data/Images" 
train_mask_folder_path = "./Data/Train_data/Labels" 
test_images_folder_path = "./Data/Validation_data/Images" 
test_mask_folder_path = "./Data/Validation_data/Labels" 

# Get image and mask names
train_images_names_original = sorted([img for img in os.listdir(train_images_folder_path) if img.endswith('.png')])
train_mask_names_original = sorted([img for img in os.listdir(train_mask_folder_path) if img.endswith('.png')])
test_images_names = sorted([img for img in os.listdir(test_images_folder_path) if img.endswith('.png')])
test_mask_names = sorted([img for img in os.listdir(test_mask_folder_path) if img.endswith('.png')])

# Load and preprocess functions
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

    image = load_image(test_images_folder_path, image_name, img_height, img_width)
    mask = load_mask(test_mask_folder_path, mask_name, img_height, img_width)
    return image, mask
    
# Functions to build the model 
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
    x = Dropout(0.2)(x)

    x = Conv2D(filters, 3, padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    if pool == True:
        p = MaxPool2D((2, 2))(x)
        return x, p
    else:
        return x
        
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
    u1 = Conv2DTranspose(256, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 128, pool = False)

    u2 = Conv2DTranspose(128, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 64, pool = False)

    u3 = Conv2DTranspose(64, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool = False)

    u4 = Conv2DTranspose(32, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool = False)

    # Output
    output = Conv2D(num_classes, 1, padding = 'same', activation = 'softmax')(x8)

    return Model(inputs, output)
    

# Function to calculate the mean IoU (metric)
def iou_metrics(y_true, y_pred):
    ''' Calculate the intersection over union metric for each class for one image.
  
    Arguments:
        - y_true (array): Real mask with one hot (height, width, 30)
        - y_pred (array): Predicted mask the classes with highest probabilities (height, width, 1)

    Returns:
        - class_iou (float): Mean weighted IoU '''
    
    iou = []
    for pred in range(y_pred.shape[0]):
        one_hot_map = []
        for clas in range(len(colors)):
            class_map = tf.reduce_all(tf.equal(y_pred[pred].reshape(img_height, img_width, 1), clas), axis = -1)
            one_hot_map.append(class_map)
        pred_i = np.array(tf.cast(tf.stack(one_hot_map, axis = -1), tf.int32))
        iou.append(jaccard_score(y_true[pred].reshape(img_height*img_width, 30), pred_i.reshape(img_height*img_width,30), average = 'samples'))

    return np.mean(iou)
    

# Visualization function      
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


# Seeds
seeds = [0, 11, 25, 333, 41, 55, 666, 70, 8, 123]

# Training loop
for i in range(len(seeds)):

    # Select one seed
    print('STARTS TRAINING NUMBER ' + str(i))
    os.environ['PYTHONHASHSEED'] = str(seeds[i])
    tf.random.set_seed(seeds[i])
    np.random.seed(seeds[i])
    random.seed(seeds[i])


    names = ['unlabeled', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall',
             'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
             'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']

    colors = np.array([(0, 0, 0), (111, 74, 0), (81, 0, 81), (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), 
            (102, 102, 156), (190, 153, 153), (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), 
            (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), ( 0, 0, 142), ( 0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), 
            (119, 11, 32), (0, 0, 142)], dtype = np.int32)


    # Divide into train and validation and reorder
    values = random.sample(range(len(train_images_names_original)), 300)
    train_values = random.sample(list(set(range(len(train_images_names_original))) - set(values)), len(train_images_names_original) - 300)
    print(len(set(values).intersection(set(train_values))))

    train_images_names, train_mask_names, val_images_names, val_mask_names = [], [], [], []

    train_images_names = [train_images_names_original[j] for j in train_values]
    train_mask_names = [train_mask_names_original[j] for j in train_values]
    val_images_names = [train_images_names_original[j] for j in values]
    val_mask_names = [train_mask_names_original[j] for j in values]

    # Count check
    print(f'Train images: {len(train_images_names)}')
    print(f'Train labels: {len(train_mask_names)}')
    print(f'Val images: {len(val_images_names)}')
    print(f'Val labels: {len(val_mask_names)}')
    print(f'Test images: {len(test_images_names)}')
    print(f'Test labels: {len(test_mask_names)}')


    # Define image shape. Images are 1024 x 2048 (height x width) -> (896) 768 x 2048 after the crop, so to maintain the ratio: 
    img_height, img_width = 96, 256  #112, 256
    batch_size = 32

    # Check mask shape after one hot encoding
    print(load_mask(train_mask_folder_path, train_mask_names[0], img_height, img_width).shape)

    # Create train tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images_names, train_mask_names))
    train_dataset = train_dataset.map(load_train, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    print(train_dataset)

    # Create validation tensorflow dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images_names, val_mask_names))
    val_dataset = val_dataset.map(load_train, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size)
    print(val_dataset)

    # Create test tensorflow dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images_names, test_mask_names))
    test_dataset = test_dataset.map(load_val, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)
    print(test_dataset)

    # Load the validation mask, one hot and get the correct value (96, 256, 1)
    y_real = np.array([load_mask(test_mask_folder_path, test_name, img_height, img_width, one_hot = True) for test_name in test_mask_names])
    y_real_oh = np.argmax(y_real, axis = 3)
    y_val = np.array([load_mask(train_mask_folder_path, val_name, img_height, img_width, one_hot = True) for val_name in val_mask_names])


    # Build the model
    model1 = build_unet((img_height, img_width, 3), 30)
    print(model1.summary())

    # IoU callback: to calculate after each epoch and save the weights of the epoch with higher IoU on validation
    ious = []
    class JaccardScoreCallback(keras.callbacks.Callback):
        """Computes the Jaccard score and logs the results to TensorBoard."""
    
        def __init__(self, model, y_val, val_dataset):
            self.model = model
            self.y_validation = y_val
            self.validation_dataset = val_dataset
            self.keras_metric = tf.keras.metrics.Mean("jaccard_score")
            self.epoch = 0
            self.best_iou = 0.0
    
        def on_epoch_end(self, batch, logs=None):
            self.epoch += 1
            self.keras_metric.reset_state()
            predictions = self.model.predict(self.validation_dataset)
            predictions = np.argmax(predictions, axis = 3)
            jaccard_value = iou_metrics(self.y_validation, predictions)
            ious.append(np.array(jaccard_value))
            if jaccard_value > self.best_iou:
              self.best_iou = jaccard_value
              self.model.save_weights('./Good_train/4_gdn/Train_'+str(i)+'/best_model.h5', overwrite = True)
            print('IoU on validation: ', jaccard_value)

    # Function to visualize and save the predictions
    def visualize_image_mask(image_paths, mask_paths, preds, colors, img_height = 96, img_width = 256):
        ''' Display 5 random images, their masks and their predictions.
      
        Arguments:
            - image_paths (string list): Paths to images
            - mask_paths (string list): Paths to masks
            - preds (tf tensor): Model predictions
            - colors (list): Colors list
            - img_height (int): Height to reshape -- 96
            - img_width (int): Width to reshape -- 256  '''
    
        idx = [3, 80, 194, 281, 406]
        print('Test images: ', idx)
    
        images = [load_image(test_images_folder_path, image_paths[i], img_height, img_width) for i in idx]
        masks = [load_mask(test_mask_folder_path, mask_paths[i], img_height, img_width, one_hot = False) for i in idx]
        preds = preds[idx]
    
        plt.figure(figsize = (20, 5))
        plt.subplots_adjust(hspace = 0.05)
    
        for k in range(5):
            plt.subplot(3, 5, k + 1)
            plt.imshow(images[k])
            plt.subplot(3, 5, k + 6)
            plt.imshow(masks[k])
            plt.subplot(3, 5, k + 11)
            plt.imshow(color_to_one_hot_mask(preds[k], colors))
        
        plt.savefig('./Good_train/4_gdn/Train_'+str(i)+'/Predictions.png')
        #plt.show()
        plt.clf()

    # Training hyperparameters
    lr = 1e-3
    epochs = 500

    # Reduce learning rate callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy', 
                                                     factor = 0.5,
                                                     patience = 15,
                                                     min_delta = 0.001, 
                                                     min_lr = 1e-12,
                                                     verbose = 1)

    # IoU callback
    callback_iou = JaccardScoreCallback(model1, y_val, val_dataset)

    # Compile and train
    model1.compile(loss = 'mae',    
                   optimizer = tf.keras.optimizers.Adam(learning_rate = lr), 
                   metrics = ['accuracy'])

    history_model1 = model1.fit(train_dataset, 
                                epochs = epochs,
                                verbose = 1,
                                callbacks = [callback_iou, reduce_lr],
                                validation_data = val_dataset)

    # Save history
    np.save('./Good_train/4_gdn/Train_'+str(i)+'/history_train_'+str(i)+'.npy', history_model1.history)
    np.save('./Good_train/4_gdn/Train_'+str(i)+'/ious_train_'+str(i)+'.npy', np.asarray(ious))


    # Plot and save the training curves
    plt.plot(np.log10(history_model1.history['lr']))
    plt.grid()
    plt.title('Model learning rate')
    plt.ylabel('Log10(lr)')
    plt.xlabel('Epoch')
    plt.savefig('./Good_train/4_gdn/Train_'+str(i)+'/Lr.png')
    #plt.show()
    plt.clf()

    plt.plot(history_model1.history['accuracy'], label = 'Train')
    plt.plot(history_model1.history['val_accuracy'], label = 'Validation')
    plt.grid()
    plt.legend()
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('./Good_train/4_gdn/Train_'+str(i)+'/Accuracy.png')
    #plt.show()
    plt.clf()

    plt.plot(history_model1.history['loss'], label = 'Train')
    plt.plot(history_model1.history['val_loss'], label = 'Validation')
    plt.grid()
    plt.legend()
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('./Good_train/4_gdn/Train_'+str(i)+'/Loss.png')
    #plt.show()
    plt.clf()

    plt.plot(ious)
    plt.grid()
    plt.title('Model validation IoU')
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.savefig('./Good_train/4_gdn/Train_'+str(i)+'/Iou_validation.png')
    #plt.show()
    plt.clf()

    # Restore best weights
    model1.load_weights('./Good_train/4_gdn/Train_'+str(i)+'/best_model.h5')

    # Test predictions
    preds_buenas = model1.predict(test_dataset, batch_size = 32)
    preds_buenas = np.argmax(preds_buenas, axis = 3)

    # Visualize predictions
    visualize_image_mask(test_images_names, test_mask_names, preds_buenas, colors, img_height = 96, img_width = 256)

    #Evaluate
    print(model1.evaluate(train_dataset))
    print(model1.evaluate(val_dataset))
    print(model1.evaluate(test_dataset))

    # IoU
    print('Higher IoU acchieved on validation: ' + str(np.max(ious)))
    print('Mean test IOU: ', iou_metrics(y_real, preds_buenas))

