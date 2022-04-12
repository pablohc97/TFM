# Autonomous driving semantic segmentation improvement using divisive normalization

One of the key problems in computer vision is adaptation: models are too rigid to follow the variability of the inputs. The canonical computation that explains adaptation in sensory neuroscience is divisive normalization, and it has appealing effects on image manifolds. In this work we show that including divisive normalization in current deep networks makes them more invariant to non-informative changes in the images. In particular, the main objective is to check if the segmentation of autonomous driving images is improved by applying the divisive normalization (DN), specially for bad weather condition, such as fog, which introduces variability in textures. However, we face two problems:

## Classification problem

### Dataset

We use the [Cifar-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) in this problem. It consits of 60000 32 x 32 colour images in 10 classes. It is divided into 50000 training images and 10000 test images. We extract 5000 random images from the training images to use them as validation images. As the Cifar-10 images are so perfect, we perform two random modifications which locally change the contrast or the luminance of the images.

<p align="center">
    <img src="./Classification/Cifar_images.png" width="480" height="200" />
</p>

### Experiments and results

We build and train three different models. The first one, without any DN layers, has three convolutional layers with RELU followed by an average pooling and a final 10 neurons layers with soft-max activation function to predict the class of each image. The second model has the same extructure but includes one GDN layer located in the first position, i.e. the first layer of the model is the DN layer. Finally, the last model has the same extructure too but it includes three DN layers located before each convolutional layer.

We train each model 2000 epochs with the modified dataset. We repeat 10 times with different seeds and then we evaluate the models with the modified and the original data. Next table shows the mean and standard deviation accuracy performance in test and the improvements of the use of DN layers with regard no using DN in parenthesis.

| Dataset           |   No DN layers   |        1 DN layer       |      3 DN layers        |
|:-----------------:|:----------------:|:-----------------------:|:-----------------------:|
| Original Cifar-10 |  0.75 &pm; 0.01  |  0.76 &pm; 0.01 (1.3%)  |  0.77 &pm; 0.01 (2.7%)  |
| Modified Cifar-10 |  0.74 &pm; 0.01  |  0.75 &pm; 0.01 (1.4%)  |  0.77 &pm; 0.01 (4.1%)  |



## Segmentation problem

### Dataset

The Cityscapes dataset is used in this problem. It is a large-scale dataset that contains a diverse set of stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5000 frames in addition to a larger set of 20000 weakly annotated frames. You can access it from [here](https://www.cityscapes-dataset.com/).

<p align="center">
    <img src="https://i.imgur.com/50UFABF.jpg" width="480" height="240" />
</p>


### Experiments and results
