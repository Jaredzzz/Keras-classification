from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D,Dense
from keras.utils.vis_utils import plot_model
from keras.layers.merge import add
from keras import backend as K
from keras.initializers import glorot_uniform
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'
os.environ['CUDA_VISIBLE_DEVICES'] ="0"


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv_block(input, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6."""
    x = input
    x = Conv2D(filters, kernel, padding='same', strides=strides)(x)
    x = BatchNormalization(epsilon=0.001)(x)
    x = Activation(relu6)(x)
    return x


def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)


def _bottleneck(input, filters, kernel_size, t, s, skip=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    """

    tchannel = K.int_shape(input)[-1] * t

    x = _conv_block(input, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel_size, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(epsilon=0.001)(x)
    x = Activation(relu6)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(epsilon=0.001)(x)

    if skip:
        x = add([x, input])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


def create_mobilenet_v2_model(num_classes, alpha=1.0):
    input_image = Input(shape=(None, None, 3))

    x = _conv_block(input_image, 32, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

    if alpha > 1.0:
        last_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_filters = 1280

    x = _conv_block(x, last_filters, (1, 1), strides=(1, 1))
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes,
                   activation='softmax',
                   use_bias=True,
                   kernel_initializer=glorot_uniform(seed=0))(x)

    model = Model(input_image, output)

    return model
#
# model = create_mobilenet_v2_model(num_classes=6)
# model.summary()
#
# plot_model(model,"model_mobilenet_v2.png",show_shapes=True,show_layer_names=True)