"""
@Autor: Anderson Alves Schinaid
Data: 16/08/2018
"""

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm


# tamanho da imagem local aonde armazena as imagens do dataset
train_dir = 'data/train'
teste_dir = 'data/teste'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'model-{}-{}.model'.format(LR, '2conv-basic')

#função para a label da imagem a ser carregada e retornando valor adequado 
def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'Planta_boa': return [1,0]
    elif word_label == 'Planta_com_deficiencia': return [0,1]

#função para criar o treinamento
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data 
        
#função para processar o teste
def process_teste_data():
    testing_data = []
    for img in tqdm(os.listdir(teste_dir)):
        path = os.path.join(teste_dir, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img),img_num])
    
    np.save('teste_data.npy', testing_data)
    return testing_data 
                
#treinando
train_data = create_train_data()
#caso já tenha o treinamento feito
#train_data = np.Load('train_data.npy')


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name = 'input')

#convnet = conv_2d(convnet, 32, 2, ativation = 'relu')
#convnet = max_pool_2d(convert, 2)

#convnet = conv_2d(convnet, 64, 2, ativation = 'relu')
#convnet = max_pool_2d(convert, 2)

#convnet = conv_2d(convnet, 32, 2, ativation = 'relu')
#convnet = max_pool_2d(convert, 2)

#convnet = conv_2d(convnet, 64, 2, ativation = 'relu')
#convnet = max_pool_2d(convert, 2)

convnet = conv_2d(convnet, 32, 2, ativation = 'relu')
convnet = max_pool_2d(convert, 2)

convnet = conv_2d(convnet, 64, 2, ativation = 'relu')
convnet = max_pool_2d(convert, 2)

convnet = fully_connected(convnet, 1024, 2, ativation = 'relu')
convnet = dropout(convert, 0.8)

convnet = fully_connected(convnet, 2, ativation = 'softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='target')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exist('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('modelo carregado')

train = train_data[:-500]
teste = train_data[:-500]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

teste_X = np.array([i[0] for i in teste]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
teste_Y = [i[1] for i in teste]

model.fit({'input': X}, {'target': Y}, n_epoch=5, validation_set = ({'input': teste_X}, {'target': teste_Y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#tensorboard --logdir = foo:C:\Users\Alchemist\Desktop\Python\TCC\Classificação usando tflearn\log

if __name__ == '__main__':
    run()
