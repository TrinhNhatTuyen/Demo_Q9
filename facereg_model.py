import numpy as np
import cv2

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
# from keras.preprocessing.image import load_img, save_img, img_to_array
# from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
from os import listdir


# from _future_ import print_function
# from _future_ import absolute_import
from keras import layers
from keras.regularizers import l2
from keras.layers import Input, Conv2D, Activation, BatchNormalization
from keras.layers import MaxPooling2D, AveragePooling2D, Flatten, Dense

global weight_decay
weight_decay = 1e-4
def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_2)(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_3)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):

    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1), strides=strides,
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size, padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_2)(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_3)(x)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    bn_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='orthogonal',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay),
                      trainable=trainable,
                      name=conv_name_4)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_4)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50_backend(inputs):
    bn_axis = 3
    # inputs are of size 224 x 224 x 3
    x = Conv2D(64, (7, 7), strides=(2, 2),
               kernel_initializer = 'orthogonal',
               use_bias=False,
               trainable=True,
               kernel_regularizer=l2(weight_decay),
               padding = 'same',
               name='conv1/7x7_s2')(inputs)

    # inputs are of size 112 x 112 x 64
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # inputs are of size 56 x 56
    x = conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1), trainable=True)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=2, trainable=True)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=3, trainable=True)

    # inputs are of size 28 x 28
    x = conv_block(x, 3, [128, 128, 512], stage=3, block=1, trainable=True)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=2, trainable=True)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=3, trainable=True)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=4, trainable=True)

    # inputs are of size 14 x 14
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block=1, trainable=True)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=2, trainable=True)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=3, trainable=True)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=4, trainable=True)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=5, trainable=True)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=6, trainable=True)

    # inputs are of size 7 x 7
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block=1, trainable=True)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=2, trainable=True)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=3, trainable=True)
    return x
# it 's pretrained model. Define att to known how it work
def loadVggFaceModel():

	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	return model

def resnet50_backend_truncated(inputs):
    bn_axis = 3
    # inputs are of size 224 x 224 x 3
    x = Conv2D(64, (7, 7), strides=(2, 2),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=True,
               kernel_regularizer=l2(weight_decay),
               padding = 'same',
               name='conv1/7x7_s2')(inputs)

    # inputs are of size 112 x 112 x 64
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # inputs are of size 56 x 56
    x = conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1), trainable=True)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=2, trainable=True)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=3, trainable=True)

    # inputs are of size 28 x 28
    x = conv_block(x, 3, [128, 128, 512], stage=3, block=1, trainable=True)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=2, trainable=True)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=3, trainable=True)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=4, trainable=True)

    # inputs are of size 14 x 14
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block=1, trainable=True)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=2, trainable=True)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=3, trainable=True)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=4, trainable=True)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=5, trainable=True)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=6, trainable=True)
    return x
loadVggFaceModel()