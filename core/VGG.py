from keras.layers import Dense,Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.merge import add
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def _conv(input, conv):
    x = input
    if conv['stride'] == 1 :
        padding = "same"
    else:
        padding = "valid"
    if conv['stride'] > 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # unlike tensorflow darknet prefer left and top paddings

    x = Conv2D(conv['filter'],
                   conv['kernel'],
                   strides=conv['stride'],
                   padding=padding,
                   # unlike tensorflow darknet prefer left and top paddings
                   name='conv_' + str(conv['layer_idx']),
                   use_bias=False if conv['bnorm'] else True)(x)
    if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
    if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
    return x

def _vgg_conv_block(input,convs):
    x = input
    for conv in convs:
        x = _conv(input=x,conv=conv)
    return x

def create_VGG_model(num_classes):
    input_image = Input(shape=(None,None,3))
    x = _vgg_conv_block(input=input_image,
                        convs=[
                            {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                            {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                            {'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 3},
                            {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 4},
                            {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                            {'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 8},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                            {'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 10},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 11},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                            {'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 14},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 15},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17},
                            {'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 18}])
    x = GlobalAveragePooling2D()(x)  # 全局平均池化，一个样本转换成特征图数量的向量
    x = Dense(num_classes, activation='softmax', name='fc', kernel_initializer=glorot_uniform(seed=0))(x)
    model = Model(inputs=input_image, outputs=x, name='VGG14')
    return model
# model = create_VGG_model(num_classes=6)
# model.summary()
#
# plot_model(model,"model.png",show_shapes=True,show_layer_names=True)

