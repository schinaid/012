import os
from random import shuffle
from numpy import load 
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
 
 
## Hyperparameter
learning_rate=1e-3
IMG_SIZE = 50
MODEL = 'Classificação_usando_tflearn'
 
def create_label(image_name):
    """Create one-hote encoded vector from image name"""
    word_label = image_name.split('.')[0]
    if word_label == "Planta_boa":
        return np.array([1, 0])
    elif word_label == "Planta_com_deficiencia":
        return np.array([0, 1])
 
def create_train_data():
    """Read image as 50x50 and grayscale"""
    training_data = []
    for img in tqdm(os.listdir('data/train')):
        path = os.path.join('data/train', img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
 
    # Save processed data
    if not os.path.exists('data'):
        os.mkdir('data')
    np.save('data/train.npy', training_data)
 
    return training_data
 
def create_test_data():
    test_data = []
    for img in tqdm(os.listdir('test')):
        path = os.path.join('test', img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        test_data.append([np.array(img_data), img_num])
    shuffle(test_data)
 
    # Save processed data
    if not os.path.exists('data'):
        os.mkdir('data')
    np.save('data/test.npy', test_data)
 
    return test_data
 
 
if __name__ == '__main__':
    # Load data
    try:
        train_data = np.load('data/train.npy')
    except FileNotFoundError:
        train_data = create_train_data()
    try:
        test_data = np.load('data/test.npy')
    except FileNotFoundError:
        test_data = create_test_data()
 
    # Split data for train/evaluate set from training_data
    train_size = int(train_data.shape[0] * 0.8)
    train = train_data[:train_size]
    evaluate = train_data[train_size:]
 
    # Reshape data properly
    X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = [i[1] for i in train]
 
    X_test = np.array([i[0] for i in evaluate]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_test = [i[1] for i in evaluate]
 
 
    ## Build the model
    tf.reset_default_graph()
 
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
 
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
 
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
 
    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
 
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
 
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
 
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
 
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')
 
    model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
 
    # Try to load pre-trained model, else train model
    saved_model = "%s.tflearn" % MODEL
    try:
        model.load(saved_model)
        print("Load pre-trained model:", saved_model)
    except NotFoundError:
        model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
                  validation_set=({'input': X_test}, {'targets': y_test}),
                  snapshot_step=500, show_metric=True, run_id=MODEL)
        model.save(saved_model)
 
 
    ## Time to test
    fig=plt.figure(figsize=(16, 12))
 
    for num, data in enumerate(test_data[:16]):
        img_num = data[1]
        img_data = data[0]
 
        y = fig.add_subplot(4, 4, num+1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
 
        if np.argmax(model_out) == 1:
            str_label='Dog'
        else:
            str_label='Cat'
 
        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    # plt.show()
    plt.savefig('%s-test.png' % MODEL)
