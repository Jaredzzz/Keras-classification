# set the matplotlib backend so figures can be saved in the background
import matplotlib
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import tensorflow as tf
import cv2
import os
from keras.callbacks import TensorBoard, ModelCheckpoint, BaseLogger
import json
from core.backbone import create_darknet53_model
from utils.dataset import load_data
from keras.utils import multi_gpu_model

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


def train(args):
    config_path = args.conf

    with open(config_path, encoding='UTF-8') as config_buffer:
        config = json.loads(config_buffer.read())

    print("[INFO] Loading Dataset...")
    train_x,train_y = load_data(path=config["train"]["train_image_folder"],
                                 cache_name=config["train"]["train_cache_name"],
                                 labels=config["model"]["labels"],
                                 img_size=config["model"]["input_size"])

    valid_x,valid_y = load_data(path=config["valid"]["valid_image_folder"],
                                 cache_name=config["valid"]["valid_cache_name"],
                                 labels=config["model"]["labels"],
                                 img_size=config["model"]["input_size"])
    # print('X_train.shape:', train_x.shape)
    # print('x_valid.shape:', valid_x.shape)
    # print('y_train.shape:', train_y.shape)
    # print('y_valid.shape:', valid_y.shape)
    # print(len(train_x))
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    print("[INFO] Loading Model...")
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    multi_gpu = len(config['train']['gpus'].split(','))
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            model = create_darknet53_model(num_classes=len(config["model"]["labels"]))
            model.summary()
    else:
        model = create_darknet53_model(num_classes=len(config["model"]["labels"]))
        model.summary()
    if multi_gpu > 1:
        train_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        train_model = model
    optimizer = Adam(lr=config["train"]["learning_rate"], decay=config["train"]["learning_rate"]/config["train"]["nb_epochs"])
    train_model.compile(loss="binary_crossentropy", optimizer=optimizer,metrics=["accuracy"])

    print("[INFO] Start Training...")
    # 可视化
    tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
                             histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             #                  batch_size=32,     # 用多大量的数据计算直方图
                             write_graph=True,  # 是否存储网络结构图
                             write_grads=True,  # 是否可视化梯度直方图
                             write_images=True,  # 是否可视化参数
                             embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None)
    filepath = config["train"]["saved_weights_name"]
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')  # 保存模型
    history = train_model.fit_generator(aug.flow(train_x, train_y, batch_size=config["train"]["batch_size"]),
                            validation_data=(valid_x, valid_y), steps_per_epoch=len(train_x) // config["train"]["batch_size"],
                            epochs=config["train"]["nb_epochs"],callbacks=[tbCallBack,checkpoint], verbose=1)

    print("[INFO] Saving Model...")
    # model.save(config["train"]["saved_weights_name"])


    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure(1,figsize=(6,4))
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['acc'],label="train accuracy")
    plt.plot(history.history['val_acc'],label="valid accuracy")
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig("Accuracy.png")
    plt.show()


    # 绘制训练 & 验证的损失值
    plt.figure(2, figsize=(6, 4))
    plt.plot(history.history['loss'],label="train loss")
    plt.plot(history.history['val_loss'],label="valid loss")
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.show()
    plt.savefig("Loss.png")
    print("[INFO] Completed...")




if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate classification model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()
    train(args)

