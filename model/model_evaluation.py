import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.multiclass import unique_labels
import cv2

modelo_carregado = load_model('modelo_cnn90valAcc..h5')
classes = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
X_train = np.load('train_data/X_train.npy')
X_test = np.load('train_data/X_test.npy')
y_train = np.load('train_data/y_train.npy')
y_test = np.load('train_data/y_test.npy')

# Carregar o histórico de treinamento de um arquivo CSV
history = pd.read_csv('history_salvo90valAcc..csv')

# Acessar a lista de valores de validação de acurácia no histórico
val_accuracy = history.history['val_accuracy']

# Calcular a média da validação de acurácia
mean_val_accuracy = np.mean(val_accuracy)

print("Média da validação de acurácia:", mean_val_accuracy)

# Visualizar a perda durante o treinamento

plt.plot(history['loss'])
plt.plot(history['val_loss'])

#plt.title('Training Loss')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='upper right')
plt.show()

# Gráfico de treinamento e validação da acurácia
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Acurácia')
plt.ylabel('Acurácia')
plt.xlabel('Épocas')
plt.legend(['Treinamento', 'Validação'], loc='upper left')
plt.grid(True)
plt.show()
plt.close()


preds = modelo_carregado.predict(X_test)

def plot_confusion_matrix(
        cm,
        classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues
    ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe real')
    plt.xlabel('Classe predita')

y_test_ = [np.argmax(x) for x in y_test]
preds_ = [np.argmax(x) for x in preds]

cm = confusion_matrix(y_test_, preds_)
plot_confusion_matrix(cm, classes=['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC'], title='Confusion matrix')
plt.show()
plt.close()

# Calcular acurácia
accuracy = accuracy_score(y_test_, preds_)
print("Acurácia:", accuracy*float(100.0), "%")

# Calcular precisão
precision = precision_score(y_test_, preds_, average='macro')
print("Precisão:", precision*float(100.0), "%")

# Calcular recall
recall = recall_score(y_test_, preds_, average='macro')
print("Recall:", recall*float(100.0), "%")

# Calcular F1 score
f1 = f1_score(y_test_, preds_, average='macro')
print("F1-score:", f1*float(100.0), "%")

n = 4
for t in range(4):
    plt.figure(figsize=(10,10))
    for i in range(n*t, n*(t+1)):
        plt.subplot(1, n, i + 1 - n*t)
        plt.imshow(cv2.cvtColor(X_test[i], cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title('Real: {}\nPredito: {}'.format(classes[np.argmax(y_test[i])], classes[np.argmax(preds[i])]))
        plt.axis('off')
    plt.show()