import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf



TRAIN_DIR = 'data/train'
TEST_DIR = 'data/teste'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'Classificação_usando_tflearn-{}-{}.model'.format(LR, '2conv-basic')

###

def label_img(img_name):
    word_label = img_name.split('.')[0] 
    if word_label == 'Tomato___Bacterial_spot': return[1,0]
    elif word_label == 'Tomato___Early_blight': return[0,1]
    elif word_label == 'Tomato___healthy': return[1,1]
    

def create_train_data():
    training_data = []
    for img_ in tqdm(os.listdir(TRAIN_DIR)):
        
        label = label_img(img_)
        path = os.path.join(TRAIN_DIR,img_)
        #if img_ is not None:
            #training_data.append([np.array(img), np.array(label)])
            #img = cv2.resize(np.array(img_), (IMG_SIZE,IMG_SIZE))
            #training_data.append([np.array(img_), np.array(label)])
        #else:
        #    print("image not loaded")
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img_,(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img_),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data
    
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        
        img_data = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #img_data = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data    
        
train_data = create_train_data()
teste_data = process_test_data()

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
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
    
train = train_data[:-500]
test = train_data[-500:] #test for accuracy not testing

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in train]

model.fit({'input': X}, {'targets': Y}, n_epoch=4, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    
model.save(MODEL_NAME)



import matplotlib.pyplot as plt
test_data = process_test_data()
fig=plt.figure()

for num,data in enumerate(test_data[:12]):
    # Planta_boa: [1,0]
    # Planta_com_deficiencia: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[1]
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Tomato___Bacterial_spot'
    elif np.argmax(model_out) == 0: str_label='Tomato___Early_blight'
    else: str_label='Tomato___healthy'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()



#FINISH
with open('submission_file.csv','w') as f:
    f.write('id,label\n')
            
with open('submission_file.csv','a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))
