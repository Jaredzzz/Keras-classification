import argparse
import os
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import pickle


def load_data(path,cache_name,labels,img_size):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        return cache['all_images'],cache['all_labels']
    label_list = []
    image_list = []
    for index,label in enumerate(labels):
        filepath = os.path.join(path,label)
        # print(label)
        for img_name in os.listdir(filepath):
            img = cv2.imread(os.path.join(filepath,img_name))
            img = cv2.resize(img, (img_size, img_size))
            img = img_to_array(img)
            image_list.append(img)
            label_list.append(index)
    X = np.array(image_list, dtype="float") / 255.0
    y = np.array(label_list)
    y = to_categorical(y, num_classes=len(labels))  # one-hot
    cache = {'all_images': X, 'all_labels': y}
    with open(cache_name, 'wb') as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return X,y
