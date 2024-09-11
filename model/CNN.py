import os

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  img_to_array, load_img)
import pickle
# Parâmetros
image_width = 224
image_height = 168
channels = 1
batch_size = 32  # Tamanho do lote

# Inicializar o ImageDataGenerator para dados de treinamento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Configurar o gerador de dados de treinamento
train_generator = train_datagen.flow_from_directory(
    '../dataset/images',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

# Criar um dataset vazio para armazenar as imagens
dataset = np.zeros((len(train_generator.filenames), image_height, image_width, channels), dtype=np.float32)
y_dataset = []

# Carregar e processar as imagens em lotes
for i, (x_batch, y_batch) in enumerate(train_generator):
    dataset[i * batch_size:(i + 1) * batch_size] = x_batch
    y_dataset.extend(np.argmax(y_batch, axis=1))
    if (i + 1) * batch_size >= len(train_generator.filenames):
        break

print("Todas as imagens carregadas e processadas!")

# Exibir uma amostra de imagens
classes = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

# Dicionário para armazenar o índice da primeira imagem de cada classe
first_image_index = {}

# Encontra o índice da primeira imagem de cada classe
for i, label in enumerate(y_dataset):
    if label not in first_image_index:
        first_image_index[label] = i

# Configura a grade para exibir as imagens
num_classes = len(set(y_dataset))
num_images_per_class = 1
num_cols = num_classes
num_rows = num_images_per_class

# Cria uma figura com uma grade de subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))

# Loop através de cada classe
for i in range(num_classes):
    # Obtém o índice da primeira imagem da classe
    idx = first_image_index[i]

    # Obtém a imagem e converte para RGB
    pixels = dataset[idx].reshape(image_height, image_width)

    # Exibe a imagem no subplot correspondente
    axes[i].imshow(pixels, cmap='gray')
    axes[i].axis('off')

    # Adiciona um título para o subplot com o rótulo
    axes[i].set_title(f'{classes[i]}')

# Exibe a figura
plt.tight_layout()
plt.savefig('output/output_fig.png')

# Conversão dos labels para one-hot encoding
y_dataset = np.array(y_dataset)
y_dataset_ = to_categorical(y_dataset, num_classes)

# Ajuste o dataset para corresponder ao tamanho de y_dataset_
dataset_trimmed = dataset[:len(y_dataset_)]

# Dividir o dataset em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(dataset_trimmed, y_dataset_, test_size=0.2)

np.save('train_data/X_train.npy', X_train)
np.save('train_data/X_test.npy', X_test)
np.save('train_data/y_train.npy', y_train)
np.save('train_data/y_test.npy', y_test)

print(f"Conjunto de treinamento: {len(X_train)}, Conjunto de teste: {len(X_test)}")

# Criar listas vazias para armazenar as amostras balanceadas
balanced_X_train = []
balanced_y_train = []

# Determinar o número de amostras na classe majoritária
majority_samples = 1000

# Iterar sobre cada classe
for class_label in np.unique(np.argmax(y_train, axis=1)):
    # Filtrar amostras pertencentes a essa classe
    X_class = X_train[np.argmax(y_train, axis=1) == class_label]
    y_class = y_train[np.argmax(y_train, axis=1) == class_label]

    # Balancear as amostras aumentando a classe menos representada
    balanced_X_class, balanced_y_class = resample(X_class, y_class, replace=True, n_samples=majority_samples, random_state=42)

    # Adicionar amostras balanceadas à lista
    balanced_X_train.extend(balanced_X_class)
    balanced_y_train.extend(balanced_y_class)

# Converter listas em arrays numpy
balanced_X_train = np.array(balanced_X_train)
balanced_y_train = np.array(balanced_y_train)

# Embaralhar amostras
shuffled_indices = np.arange(len(balanced_X_train))
np.random.shuffle(shuffled_indices)
balanced_X_train = balanced_X_train[shuffled_indices]
balanced_y_train = balanced_y_train[shuffled_indices]

# Verificar o tamanho dos conjuntos de dados balanceados
print(f"Tamanho do conjunto de treinamento balanceado: {len(balanced_X_train)}")
print(f"Tamanho do conjunto de teste: {len(X_test)}")

for class_label in np.unique(np.argmax(balanced_y_train, axis=1)):
    count = np.sum(np.argmax(balanced_y_train, axis=1) == class_label)
    print(f"Classe {class_label}: {count} amostras")

from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential

# Criar o modelo
model = Sequential()

model.add(BatchNormalization(input_shape=(image_height, image_width, 1)))
# model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.2))  # Adiciona a camada de dropout

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))  # Adiciona a camada de dropout

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))  # Adiciona a camada de dropout

# model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.2))  # Adiciona a camada de dropout

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))  # Adiciona a camada de dropout
model.add(Dense(7, activation='softmax'))  # Especifica 'softmax' como a função de ativação


model.summary()

from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    
# Treinando o modelo
from tensorflow.keras.callbacks import EarlyStopping

# Configurar EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20)

# Treinar o modelo
history = model.fit(balanced_X_train, balanced_y_train,validation_split= 0.2, epochs=150, callbacks=[early_stopping], batch_size=64)

import pandas as pd

history_salvo = pd.DataFrame(history.history)
history_salvo.to_csv('history_salvo90valAcc.csv')
     

# save model structure in jason file
model_json = model.to_json()
with open("emotion_modelcnn90valAcc.json", "w") as json_file:
    json_file.write(model_json)
     

model.save('modelo_cnn90valAcc.h5')
     
