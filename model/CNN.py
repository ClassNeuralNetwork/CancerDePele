import numpy as np
import os, sys
from scipy import ndimage
import cv2
# import matplotlib.pyplot as plt
import itertools
import scipy.stats
import tensorflow as tf
from keras import applications, optimizers, Input
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.multiclass import unique_labels

# base_dir = os.getcwd()
folder = '/home/alanzin/Desktop/Facul 2024.1/Redes Neurais/CancerDePele/dataset/images'
print(folder)

image_width = 600
image_height = 450
channels = 1

train_files = []
i=0

for emotion in ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']:
    onlyfiles = [f for f in os.listdir(os.path.join(folder, emotion)) if os.path.isfile(os.path.join(folder, emotion, f))]
    for _file in onlyfiles:
        train_files.append(_file)

dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels), dtype=np.float32)
y_dataset = []

i = 0
for emotion in ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']:
    onlyfiles = [f for f in os.listdir(os.path.join(folder, emotion)) if os.path.isfile(os.path.join(folder, emotion, f))]
    for _file in onlyfiles:
        img_path = os.path.join(folder, emotion, _file)
        img = load_img(img_path, target_size=(image_height, image_width), color_mode='grayscale')
        x = img_to_array(img)
        dataset[i] = x
        mapping = {'AKIEC': 0 , 'BCC': 1, 'BKL': 2, 'DF': 3, 'MEL': 4, 'NV': 5, 'VASC': 6}
        y_dataset.append(mapping[emotion])
        i += 1
        if i == 30000:
            print("%d images to array" % i)
            break

print("All images to array!")

dataset = dataset.astype('float32')
dataset /= 255

import matplotlib.pyplot as plt

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
plt.show()

