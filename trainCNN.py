# coding=utf-8
# Train Face CNN
# Copyright 2018 Cleuton Sampaio

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build a CNN model using images in folder "train" and test with images in folder "test".
# Each image must be squared with only one face per image. Image file name must be the "category" or person's name.

# Some code parts based on: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ 

# Imports:

import keras 
import keras.backend as K
import os, random
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint
import numpy as np 
import cv2, dlib 
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image, resizeAndPad

# Parametros de controle

batch_sz = 64 # Batch size
nb_class = 4  # Numero de classes
nb_epochs = 30 # Numero de epochs de treinamento
img_h, img_w = 64, 64 # Altura e largura das imagens
train_test_ratio = 0.3
dir_raw = './raw/' # Imagens originais
dir_treino = './train/' 
dir_teste = './test/'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Preparação das imagens de treino e teste: 

pessoas = [] # Lista de pessoas encontradas

# Importante: Nas imagens de treino e teste só pode haver um rosto!

def prepare(raw_coll, dest_dir):
    for filename in raw_coll: 
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        s_height, s_width = img.shape[:2]
        dets = detector(img, 1)
        print('Processing ' + filename)
        for i, det in enumerate(dets):
            if i > 0: 
                print("Mais de um rosto detectado!!!!!")
                break
            shape = predictor(img, det)
            left_eye = extract_left_eye_center(shape)
            right_eye = extract_right_eye_center(shape)
            M = get_rotation_matrix(left_eye, right_eye)
            rotated = cv2.warpAffine(img, M, (s_height, s_width), flags=cv2.INTER_CUBIC)
            cropped = crop_image(rotated, det)
            squared = resizeAndPad(cropped, (img_h,img_w), 127)
            output_image_path = dest_dir + os.path.basename(filename)
            cv2.imwrite(output_image_path, squared)        

def preprocess():
    raw_images = [dir_raw+i for i in os.listdir(dir_raw) if '.jpg' in i]
    if len(raw_images) > 0: 
        raw_train ,raw_test = train_test_split(raw_images,test_size=train_test_ratio) 
        prepare(raw_train, dir_treino)
        prepare(raw_test, dir_teste)

def ler_imagem(file):
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imres = cv2.resize(gray, (img_h, img_w), interpolation=cv2.INTER_CUBIC)
    imres = image.img_to_array(imres.T)
    imres = np.expand_dims(imres, axis = 0)    
    return imres

def gerar_dataset(filenames):
    rotulos = []
    dataset = np.ndarray((len(filenames), img_h, img_w, 1), dtype=np.uint8)
    x = 0
    #
    # Obtem o nome da pessoa a partir do nome do arquivo:  <name>.<number>.jpg. i.e.: fulano.2.jpg will resultado: 'fulano" 
    #
    for arquivo in filenames:
        dataset[x] = ler_imagem(arquivo)
        nome = os.path.splitext(os.path.basename(arquivo))[0]
        nome = nome.split('.')[0]
        indx = 0
        if nome in pessoas:
            indx = pessoas.index(nome)
        else:
            pessoas.append(nome)
            indx = pessoas.index(nome)
        rotulos.append(indx)
        x = x + 1
        if x%1000==0:
            print("# processed ",x)
    return dataset, rotulos

preprocess()
imagens_treino = [dir_treino+i for i in os.listdir(dir_treino) if '.jpg' in i]
random.shuffle(imagens_treino)
imagens_teste  = [dir_teste+i for i in os.listdir(dir_teste) if '.jpg' in i]
x_treino, y_treino = gerar_dataset(imagens_treino)
x_teste, y_teste  = gerar_dataset(imagens_teste)
print(pessoas)

# Gerador de camadas convolucionais: 

def conv3x3(input_x,nb_filters):
    # Prepara a camada convolucional
    return Conv2D(nb_filters, kernel_size=(3,3), use_bias=False,
               activation='relu', padding="same")(input_x)

# Criação do modelo: 

# Normaliza os valores dos pixels

x_treino = x_treino.astype('float32')
x_teste = x_teste.astype('float32')
x_treino /= 255.0
x_teste /= 255.0

# Converte os rotulos para "One-hot encoding": 

y_treino = keras.utils.to_categorical(y_treino, nb_class)
y_teste = keras.utils.to_categorical(y_teste, nb_class)

# Cria o modelo executando um treino e avaliacao:

inputs = Input(shape=(img_h, img_w, 1))
x = conv3x3(inputs, 32)
x = conv3x3(x, 32)
x = MaxPooling2D(pool_size=(2,2))(x) 
x = conv3x3(x, 64)
x = conv3x3(x, 64)
x = MaxPooling2D(pool_size=(2,2))(x) 
x = conv3x3(x, 128)
x = MaxPooling2D(pool_size=(2,2))(x) 
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
preds = Dense(nb_class, activation='softmax')(x)
model = Model(inputs=inputs, outputs=preds)

# Compila o modelo: 

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Cria um callback para salvar o modelo a cada "epoch" de treinamento completada: 

callback = ModelCheckpoint('faces_saved.h5')

# Treina o modelo (demora cerca de 6 minutos sem GPU):

history = model.fit(x_treino, y_treino,
          batch_size=batch_sz,
          epochs=nb_epochs,
          verbose=1,
          validation_data=(x_teste, y_teste), 
          callbacks=[callback])

# Avalia o modelo com dados de teste:
score = model.evaluate(x_teste, y_teste, verbose=0)
print('Perda:', score[0])
print('Acuracia:', score[1])

# Validação com predições (Novas imagens): 

def validate():
    dir_validar = dir_val
    nomes = [dir_validar+i for i in os.listdir(dir_validar) if '.jpg' in i]
    print(nomes)

    def prepararImagem(imagem):
        test_image = image.img_to_array(imagem.T)
        test_image = np.expand_dims(test_image, axis = 0)    
        return test_image

    def mostraCateg(resultado):
        categs = pessoas
        print(resultado)
        for idx, val in enumerate(resultado[0]):
            if val == 1:
                return categs[idx]
        
    i = 1
    for nome in nomes:
        im = cv2.imread(nome)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        imres = cv2.resize(gray, (img_h, img_w), interpolation=cv2.INTER_CUBIC)
        dados = prepararImagem(imres)
        i = i + 1
        ret = model.predict(dados, batch_size=1) 
        print(mostraCateg(ret))

dir_val = './val/'
dir_rawval = './rawvalidation/'
raw_images = [dir_rawval+i for i in os.listdir(dir_rawval) if '.jpg' in i]
if len(raw_images) > 0: 
    prepare(raw_images, dir_val)
    validate()
