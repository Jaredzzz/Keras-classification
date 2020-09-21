from keras.layers import Input
from keras.engine.topology import Layer
from keras import backend as K


class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
        tanh=(1 - e^{-2x})/(1 + e^{-2x})
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
         X_input = Input(input_shape)
         X = Mish()(X_input)
    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        '''
         compute_output_shape(self, input_shape)：为了能让Keras内部shape的匹配检查通过，
         这里需要重写compute_output_shape方法去覆盖父类中的同名方法，来保证输出shape是正确的。
         父类Layer中的compute_output_shape方法直接返回的是input_shape这明显是不对的，
         所以需要我们重写这个方法。所以这个方法也是4个要实现的基本方法之一。
        '''
        return input_shape