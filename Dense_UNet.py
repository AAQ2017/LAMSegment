import tensorflow as tf
import numpy as np

#%%
###############################################################################
########################### FD-UNet Sub Function ##############################
###############################################################################

# Four layered dense block
def DenseBlock(input_tensor, growth_rate):

    Dense_layer_1 = tf.keras.layers.Conv2D(filters=input_tensor.get_shape()[-1], kernel_size=(1,1), strides=(1,1), padding='same', activation=None)(input_tensor)
    Dense_layer_1 = tf.keras.layers.BatchNormalization()(Dense_layer_1)
    Dense_layer_1 = tf.keras.layers.Activation(activation='relu')(Dense_layer_1)

    Dense_layer_1 = tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(Dense_layer_1)
    Dense_layer_1 = tf.keras.layers.BatchNormalization()(Dense_layer_1)
    Dense_layer_1 = tf.keras.layers.Activation(activation='relu')(Dense_layer_1)

    Dense_layer_2 = tf.keras.layers.concatenate(inputs=[input_tensor, Dense_layer_1])
    Dense_layer_2 = tf.keras.layers.Conv2D(filters=input_tensor.get_shape()[-1], kernel_size=(1,1), strides=(1,1), padding='same', activation=None)(Dense_layer_2)
    Dense_layer_2 = tf.keras.layers.BatchNormalization()(Dense_layer_2)
    Dense_layer_2 = tf.keras.layers.Activation(activation='relu')(Dense_layer_2)

    Dense_layer_2 = tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(Dense_layer_2)
    Dense_layer_2 = tf.keras.layers.BatchNormalization()(Dense_layer_2)
    Dense_layer_2 = tf.keras.layers.Activation(activation='relu')(Dense_layer_2)

    Dense_layer_3 = tf.keras.layers.concatenate(inputs=[input_tensor, Dense_layer_1, Dense_layer_2])
    Dense_layer_3 = tf.keras.layers.Conv2D(filters=input_tensor.get_shape()[-1], kernel_size=(1,1), strides=(1,1), padding='same', activation=None)(Dense_layer_3)
    Dense_layer_3 = tf.keras.layers.BatchNormalization()(Dense_layer_3)
    Dense_layer_3 = tf.keras.layers.Activation(activation='relu')(Dense_layer_3)

    Dense_layer_3 = tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(Dense_layer_3)
    Dense_layer_3 = tf.keras.layers.BatchNormalization()(Dense_layer_3)
    Dense_layer_3 = tf.keras.layers.Activation(activation='relu')(Dense_layer_3)

    Dense_layer_4 = tf.keras.layers.concatenate(inputs=[input_tensor, Dense_layer_1, Dense_layer_2, Dense_layer_3])
    Dense_layer_4 = tf.keras.layers.Conv2D(filters=input_tensor.get_shape()[-1], kernel_size=(1,1), strides=(1,1), padding='same', activation=None)(Dense_layer_4)
    Dense_layer_4 = tf.keras.layers.BatchNormalization()(Dense_layer_4)
    Dense_layer_4 = tf.keras.layers.Activation(activation='relu')(Dense_layer_4)

    Dense_layer_4 = tf.keras.layers.Conv2D(filters=growth_rate, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(Dense_layer_4)
    Dense_layer_4 = tf.keras.layers.BatchNormalization()(Dense_layer_4)
    Dense_layer_4 = tf.keras.layers.Activation(activation='relu')(Dense_layer_4)

    Dense_layer_concatenation = tf.keras.layers.concatenate(inputs=[input_tensor, Dense_layer_1, Dense_layer_2, Dense_layer_3, Dense_layer_4])

    return Dense_layer_concatenation

#%%
###############################################################################
############################ FD-UNet Based Dense UNet Model ##############################
###############################################################################
def Dense_UNet(image_height, image_width, image_channels, first_layer_growth_rate):

    Convolutional_layer = []
    Deconvolutional_layer = []
    num_of_layers = (int(np.log2(image_height)) - int(np.log2(8))) + 1
  
    for i in range(num_of_layers):
        Convolutional_layer.append([])
        Deconvolutional_layer.append([])
  
    # Contracting path
    Inputs = tf.keras.layers.Input(shape=(image_height, image_width, image_channels))
    Convolutional_layer[0] = tf.keras.layers.Conv2D(filters=first_layer_growth_rate*4, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(Inputs)
    Convolutional_layer[0] = tf.keras.layers.BatchNormalization()(Convolutional_layer[0])
    Convolutional_layer[0] = tf.keras.layers.Activation(activation='relu')(Convolutional_layer[0])
    Convolutional_layer[0] = DenseBlock(input_tensor=Convolutional_layer[0], growth_rate=first_layer_growth_rate)

    k_factor = int(np.log2(Convolutional_layer[0].shape[-1])) - 2

    for i in range(num_of_layers-1):
        Convolutional_layer[i] = tf.keras.layers.Conv2D(
            filters=Convolutional_layer[i].shape[-1], 
            kernel_size=(1,1), 
            strides=(1,1), 
            padding='same', 
            activation=None)(Convolutional_layer[i])
        Convolutional_layer[i] = tf.keras.layers.BatchNormalization()(Convolutional_layer[i])
        Convolutional_layer[i] = tf.keras.layers.Activation(activation='relu')(Convolutional_layer[i])
        
        Convolutional_layer[i+1] = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(Convolutional_layer[i])
        Convolutional_layer[i+1] = DenseBlock(input_tensor=Convolutional_layer[i+1], growth_rate=2 ** (k_factor + i))

    Deconvolutional_layer[0]= Convolutional_layer[-1]

    # Expansive path
    for i in range(num_of_layers-1):
        Deconvolutional_layer[i+1] = tf.keras.layers.Conv2DTranspose(filters=int(Deconvolutional_layer[i].get_shape()[-1]/2), 
                                                               kernel_size=(2,2), 
                                                               strides=(2,2), 
                                                               padding='same', 
                                                               activation=None)(Deconvolutional_layer[i])
        Deconvolutional_layer[i+1] = tf.keras.layers.BatchNormalization()(Deconvolutional_layer[i+1])
        Deconvolutional_layer[i+1] = tf.keras.layers.Activation(activation='relu')(Deconvolutional_layer[i+1])
        Deconvolutional_layer[i+1] = tf.keras.layers.concatenate(inputs=[Convolutional_layer[-2-i], Deconvolutional_layer[i+1]])

        Deconvolutional_layer[i+1] = tf.keras.layers.Conv2D(filters=int(Deconvolutional_layer[i+1].get_shape()[-1]/4), 
                                                      kernel_size=(1,1), 
                                                      strides=(1,1), 
                                                      padding='same', 
                                                      activation=None)(Deconvolutional_layer[i+1])
        Deconvolutional_layer[i+1] = tf.keras.layers.BatchNormalization()(Deconvolutional_layer[i+1])
        Deconvolutional_layer[i+1] = tf.keras.layers.Activation(activation='relu')(Deconvolutional_layer[i+1])
        Deconvolutional_layer[i+1] = DenseBlock(input_tensor=Deconvolutional_layer[i+1], growth_rate=2**int(np.log2(Convolutional_layer[-2-i].shape[-1]/2/4)))

    Outputs = tf.keras.layers.Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same', activation=None)(Deconvolutional_layer[-1])
    
    Outputs = tf.keras.layers.Conv2D(filters=2, kernel_size=(1,1), strides=(1,1), padding='same', activation=None)(Outputs)
    Outputs = tf.keras.layers.Softmax()(Outputs)
    
    return tf.keras.Model(inputs=[Inputs], outputs=[Outputs])