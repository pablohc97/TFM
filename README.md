# Autonomous driving semantic segmentation improvement using generalized divisive normalization

The main objective is to check if the segmentation of autonomous driving images is improved by applying the normalization technique known as generalized divisive normalization (GDN). Two main problems will be faced:

## Classification problem

### Dataset

The [Cifar-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is used in this problem. It consits of 60000 32 x 32 colour images in 10 classes. It is divided into 50000 training images and 10000 test images. 5000 random images are extracted from the training images to be used as validation images. Also, two new datasets are created from this one. First, the luminance of the images is randomly reduced by multiplying them by a factor between 0.2 and 1, where 1 means to leave the image as the original. Secondly, the contrast of the images is randomly changed (with the [PIL function](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html)) between the factor 0.2 and 1.8, where as before the value of 1 means to leave the image as the original.

### Experiments and results

Three different models are created. The first one, without any GDN layers, has two convolutional layers with RELU followed by an average pooling and a final 10 neurons layers with soft-max activation function to predict the class of each image. The second model has the same extructure but includes one GDN layer located in the first position, i.e. the first layer is a GDN layer. Finally, the last model has the same extructure too but it includes two GDN layers. In this case it has the first GDN layer and the other is located before the second convolution.

These three models are trained with the three previously mentioned datasets. The results on test can be seen in the next table.

| Dataset            | 2 GDN layers | 1 GDN layer | No GDN layers |
|:------------------:|:------------:|:-----------:|:-------------:|
| Original Cifar-10  |    0.5721    |    0.4439   |    0.4102     |
| Luminance Cifar-10 |    0.5464    |    0.4323   |    0.4043     |
| Contrast Cifar-10  |    0.5012    |    0.3957   |    0.3954     |


## Segmentation problem

### Dataset

The Cityscapes dataset is used in this problem. It is a large-scale dataset that contains a diverse set of stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5000 frames in addition to a larger set of 20000 weakly annotated frames. You can access it from [here](https://www.cityscapes-dataset.com/).

<p>
    <img src="https://i.imgur.com/50UFABF.jpg" width="480" height="240" />
</p>


### Experiments and results
