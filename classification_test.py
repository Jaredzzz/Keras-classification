import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import tensorflow as tf
import cv2
import sys
import os
import json
from keras.models import load_model
from sklearn.metrics import accuracy_score,recall_score
from keras.preprocessing.image import img_to_array

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


def test(args):
    config_path = args.conf

    with open(config_path, encoding='UTF-8') as config_buffer:
        config = json.loads(config_buffer.read())

    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'], compile=False)

    if config["test"]["test_flag"]:
        print("[INFO] Testing Dataset...")
        for index, label in enumerate(config["model"]["labels"]):
            y_true = []
            y_pred = []
            filepath = os.path.join(config["test"]["test_image_folder"], label)
            # print(label)
            for img_name in os.listdir(filepath):
                img = cv2.imread(os.path.join(filepath, img_name))
                img = cv2.resize(img, (config["model"]["input_size"], config["model"]["input_size"]))
                img = img.astype("float") / 255.0
                img = img_to_array(img)
                image = np.expand_dims(img, axis=0)
                predict = infer_model.predict(image)[0]
                prediction = np.argmax(predict)
                # print(prediction)
                y_true.append(index)
                y_pred.append(prediction)
            #accuracy_test = recall_score(y_true, y_pred, average='micro')  # 把实际的和预测的往里丢
            accuracy_test = accuracy_score(y_true, y_pred)
            print('-' * 35)
            print('Class %s test accuracy:' % label, accuracy_test)
    elif args.img:
        # load the image
        print("[INFO] Loading Image...")
        try:
            image = cv2.imread(args.img)
            orig = image.copy()
        except AttributeError:
            print("[INFO] Error in the test image... ")
            print('[INFO] Exiting...')
            sys.exit()
        img = cv2.resize(image, (config["model"]["input_size"], config["model"]["input_size"]))
        img = img.astype("float") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        predict = infer_model.predict(img)[0]
        prediction = np.argmax(predict)
        proba = predict[prediction]
        label = config["model"]["labels"][prediction]
        label = "{}: {:.2f}%".format(label, proba * 100)
        print(label)
        # draw the label on the image
        output = cv2.resize(orig, (config["model"]["input_size"], config["model"]["input_size"]))
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # show and save the output image
        cv2.imshow("Output", output)
        cv2.imwrite("result/%s _output.png"%args.img.split('/')[-1], output)
        cv2.waitKey()  # Press any key to exit the output image
    else:
        print('[ERROR] Please input dataset path or image path')
    print('[INFO] Exiting...')










if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate classification model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--img', help='path to configuration file')
    args = argparser.parse_args()
    test(args)