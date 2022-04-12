# Autonomous driving semantic segmentation improvement using divisive normalization

One of the key problems in computer vision is adaptation: models are too rigid to follow the variability of the inputs. The canonical computation that explains adaptation in sensory neuroscience is divisive normalization, and it has appealing effects on image manifolds. In this work we show that including divisive normalization in current deep networks makes them more invariant to non-informative changes in the images. In particular, the main objective is to check if the segmentation of autonomous driving images is improved by applying the divisive normalization (DN), specially for bad weather condition, such as fog, which introduces variability in textures. However, we face two problems:

## Classification problem

### Dataset

We use the [Cifar-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) in this problem. It consits of 60000 32 x 32 colour images in 10 classes. It is divided into 50000 training images and 10000 test images. We extract 5000 random images from the training images to use them as validation images. As the Cifar-10 images are so perfect, we perform two random modifications which locally change the contrast or the luminance of the images.

<p>
    <img src="./Classification/Cifar_images.png" width="480" height="240" />
</p>

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
