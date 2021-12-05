#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.constraints import NonNeg


# In[6]:


class GDN(Layer):
    def __init__(self, 
                 filter_shape = (3,3), 
                 **kwargs):
      
        self.filter_shape = filter_shape

        super(GDN, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
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
        
        
        super(GDN, self).build(input_shape)  # Be sure to call this at the end

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

