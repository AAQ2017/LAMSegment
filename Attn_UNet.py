import tensorflow as tf
import keras.layers as L
from keras.models import Model

def conv_block(x, num_filters):
    x = L.Conv2D(num_filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
 
    x = L.Conv2D(num_filters, 3, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("relu")(x)
 
    return x

def encoder_block(x, num_filters):
    x = conv_block(x, num_filters)
    p = L.MaxPool2D((2, 2))(x)
    return x, p

def attention_gate(g, s, num_filters):
    Wg = L.Conv2D(num_filters, 1, padding="same")(g)
    Wg = L.BatchNormalization()(Wg)
 
    Ws = L.Conv2D(num_filters, 1, padding="same")(s)
    Ws = L.BatchNormalization()(Ws)
 
    out = L.Activation("relu")(Wg + Ws)
    out = L.Conv2D(num_filters, 1, padding="same")(out)
    out = L.Activation("sigmoid")(out)
 
    return out * s

def decoder_block(x, s, num_filters):
    x = L.UpSampling2D(interpolation="bilinear")(x)
    s = attention_gate(x, s, num_filters)
    x = L.Concatenate()([x, s])
    x = conv_block(x, num_filters)
    return x

def attention_unet(input_shape):
    """ Inputs """
    inputs = L.Input(input_shape)      # (512, 512, 1)
 
    """ Encoder """
    s1, p1 = encoder_block(inputs, 17)
    s2, p2 = encoder_block(p1, 34)     
    s3, p3 = encoder_block(p2, 68)     
    s4, p4 = encoder_block(p3, 136)  
    s5, p5 = encoder_block(p4, 272)    
    s6, p6 = encoder_block(p5, 544)    
    
    b1 = conv_block(p6, 1088)          
 
    """ Decoder """
    d1 = decoder_block(b1, s6, 544)
    d2 = decoder_block(d1, s5, 272)
    d3 = decoder_block(d2, s4, 136)
    d4 = decoder_block(d3, s3, 68)
    d5 = decoder_block(d4, s2, 34)
    d6 = decoder_block(d5, s1, 17)
    
    """ Outputs """
    outputs = L.Conv2D(2, 1, padding="same", activation="softmax")(d6)
    
    """ Model """
    return Model(inputs, outputs, name="Attention-UNET")


# if __name__ == "__main__":
#     input_shape = (512, 512, 1)
#     model = attention_unet(input_shape)
#     model.summary()