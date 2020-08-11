from keras.layers import Dense,Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.merge import add
from keras.models import Model
from keras.initializers import glorot_uniform
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

'''
darknet53 model 
'''
def _darknet_conv(input,conv):
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


def _darknet_conv_block(input,convs):
    x = input
    for conv in convs:
        x = _darknet_conv(input=x,conv=conv)
    return x


def _darknet_res_block(input,convs,num_blocks):
    x = input
    x = _darknet_conv(x,convs[0])
    ori_layer_idx_1,ori_layer_idx_2 = convs[1]['layer_idx'],convs[2]['layer_idx']
    for i in range(num_blocks):
        update_layer_idx_1,update_layer_idx_2 = ori_layer_idx_1 + i*3,ori_layer_idx_2 + i*3
        convs[1].update({'layer_idx':update_layer_idx_1})
        convs[2].update({'layer_idx': update_layer_idx_2})
        y = _darknet_conv(x,convs[1])
        y = _darknet_conv(y,convs[2])
        x = add([x, y])
    return x


def create_darknet53_model(num_classes):
    input_image = Input(shape=(None,None,3))
    # layer:0
    x = _darknet_conv(input_image,{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0})

    # layer:1-4
    x = _darknet_res_block(x,[{'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                              {'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                              {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}],
                           num_blocks=1)
    # layer:5-11
    x = _darknet_res_block(x,[{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                              {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                              {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}],
                           num_blocks=2)
    # layer:12-36
    x = _darknet_res_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                               {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}],
                           num_blocks=8)
    # layer:37-61
    x = _darknet_res_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                               {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}],
                           num_blocks=8)
    # layer:62-74
    x = _darknet_res_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                               {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}],
                           num_blocks=4)
    x = GlobalAveragePooling2D()(x)  # 全局平均池化，一个样本转换成特征图数量的向量
    x = Dense(num_classes, activation='softmax', name='fc', kernel_initializer=glorot_uniform(seed=0))(x)
    model = Model(inputs=input_image, outputs=x, name='darknet53')
    return model


# model = create_darknet53_model(num_classes=6)
# model.summary()
