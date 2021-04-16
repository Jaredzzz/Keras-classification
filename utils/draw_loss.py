# -*- coding: utf-8 -*-
import keras
import numpy as np
import matplotlib.pyplot as plt
import time

class LossHistory(keras.callbacks.Callback):
    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.train_loss = {'batch': [], 'epoch': []}
        self.train_acc = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 按照batch来进行追加数据
    # def on_batch_end(self, batch, logs={}):
    #     # 每一个batch完成后向容器里面追加loss，acc
    #     self.loss['batch'].append(logs.get('yolo_layer_1_loss') + logs.get('yolo_layer_2_loss')
    #                               + logs.get('yolo_layer_3_loss'))
    #     self.loss_1['batch'].append(logs.get('yolo_layer_1_loss'))
    #     self.loss_2['batch'].append(logs.get('yolo_layer_2_loss'))
    #     self.loss_3['batch'].append(logs.get('yolo_layer_3_loss'))
    #     # 每10秒按照当前容器里的值来绘图
    #     if int(time.time()) % 10 == 0:
    #         self.draw_p([self.loss['batch'],self.loss_1['batch'],self.loss_2['batch'],self.loss_3['batch']],
    #                     ['loss','yolo_layer_1_loss','yolo_layer_2_loss','yolo_layer_3_loss'],
    #                     'train_batch')

    def on_epoch_end(self, batch, logs={}):
        self.train_loss['epoch'].append(logs.get('loss'))
        self.train_acc['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        print("\n[Train INFO]Max train accuracy:%s" % max(self.train_acc['epoch']))
        print("[Train INFO]Epoch number:%s" % self.train_acc['epoch'].index(max(self.train_acc['epoch'])))
        print("[Val INFO]Max val accuracy:%s" % max(self.val_acc['epoch']))
        print("[Val INFO]Epoch number:%s\n" % self.val_acc['epoch'].index(max(self.val_acc['epoch'])))
        self.draw_p(5, [self.train_acc['epoch'], self.val_acc['epoch']], ['accuracy', 'val_accuracy'], 'Epoch')
        self.draw_p(6, [self.train_loss['epoch'], self.val_loss['epoch']], ['loss', 'val_loss'], 'Epoch')

    # 绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
    def draw_p(self, figure, lists, label, type):
        plt.style.use("ggplot")
        plt.figure(figure, figsize=(6, 4))
        for i in range(len(lists)):
            plt.plot(range(len(lists[i])), lists[i], label=label[i])
        plt.title('Model %s' % label[0])
        plt.ylabel("%s" % label[0])
        plt.xlabel(type)
        plt.legend(loc="best")
        plt.savefig(type + '_' + label[0] + '.png')
        plt.clf()

    # 由于这里的绘图设置的是10s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程（最后一次绘图结束，有训练了0-10秒的时间）
    # 所以这里的方法会在整个训练结束以后调用
    def end_draw(self):
        self.draw_p(5, [self.train_acc['epoch'], self.val_acc['epoch']], ['accuracy', 'val_accuracy'], 'Epoch')
        self.draw_p(6, [self.train_loss['epoch'], self.val_loss['epoch']], ['loss', 'val_loss'], 'Epoch')