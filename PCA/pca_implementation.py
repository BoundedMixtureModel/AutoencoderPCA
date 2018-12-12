import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import random
from sklearn.linear_model import LogisticRegression
import itertools
from sklearn import metrics
from PCA.get_dataset import *


X_train, X_test, Y_train, Y_test = get_dataset()


mu_train = np.mean(X_train, axis = 0)
mu_test = np.mean(X_test, axis = 0)
U_train,s_train,V_train = np.linalg.svd(X_train - mu_train, full_matrices=False)
U_test, s_test, V_test = np.linalg.svd(X_test - mu_test, full_matrices = False)



#Eigenvalues
#Scree plot
plt. figure()
x = np.arange(len(s_train)) + 1
plt.plot(x,s_train, lw=1)
plt.xlim(1, 1000)
plt.xlabel("# of Principal Components")
plt.ylabel("Explained Variance")


ratio = s_train / sum(s_train)
plt.figure()
plt.plot(x, ratio*100)
plt.xlim(1, 1000)
plt.xlabel("# of Principal Components")
plt.ylabel("Explained Variance")
plt.show()


cusum = np.cumsum(ratio)
plt.figure()
plt.plot(cusum)
plt.xlabel("# of Principal Components")
plt.ylabel("Cumulative % of Explained Variance")
plt.axvline(x = 400, linestyle =  "--", color = 'g', label = "Selected # of PCs")
#plt.axhline(y = 0.4, linestyle = "-.", color = 'g')
plt.show()


Zpca_train = np.dot(X_train - mu_train, V_train.transpose())
Rpca_train = np.dot(Zpca_train[:,:400], V_train[:400,:]); 
err_train = np.sum((X_train - Rpca_train)**2)/Rpca_train.shape[0]/Rpca_train.shape[1]

Zpca_test = np.dot(X_test - mu_test, V_test.transpose())
Rpca_test = np.dot(Zpca_test[:,:400], V_test[:400,:]); 
err_test = np.sum((X_test - Rpca_test)**2)/Rpca_test.shape[0]/Rpca_test.shape[1]


plt.figure(figsize=(9,3))
toPlot_train = (X_train[0:5], Rpca_train[0:5])
for i in range(5):
    for j in range(2):
        ax = plt.subplot(2, 5, 5*j+i+1)
        plt.imshow(toPlot_train[j][i,:].reshape(64,64), cmap = 'coolwarm', interpolation = 'nearest')
    
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()



plt.figure(figsize=(9,3))
toPlot_train = (X_train[0:10], Rpca_train[0:10])
for i in range(10):
    for j in range(2):
        ax = plt.subplot(2, 10, 10*j+i+1)
        plt.imshow(toPlot_train[j][i,:].reshape(64,64), cmap = 'gray', interpolation = 'nearest')
    
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()


plt.figure(figsize=(9,3))
toPlot_test = (X_test[0:10], Rpca_test[0:10])
for i in range(10):
    for j in range(2):
        ax = plt.subplot(2, 10, 10*j+i+1)
        plt.imshow(toPlot_test[j][i,:].reshape(64,64), cmap = 'gray', interpolation = 'nearest')
    
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()


plt.figure(figsize=(9,3))
YY = np.zeros(np.asarray(Y_test).shape)
YY = YY.tolist()
for i in range(len(Y_test)):
    if Y_test[i] == 0:
          YY[i] = 'dog'
    else:
        YY[i] = 'cat'
        

for index, (image, label) in enumerate(zip(X_test[0:10], YY[0:10])):
    ax = plt.subplot(1, 10, index + 1)
    plt.imshow(np.reshape(image, (64,64)), cmap=plt.cm.gray)
    plt.title('X_test:%s\n' % label, fontsize = 6)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()


plt.figure(figsize=(9,3))
for index, (image, label) in enumerate(zip(Rpca_test[0:10], YY[0:10])):
    ax = plt.subplot(1, 10, index + 1)
    plt.imshow(np.reshape(image, (64,64)), cmap=plt.cm.gray)
    plt.title('Rpca_test:%s\n' % label, fontsize = 6)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues_r):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()







# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, Y_train)
# Use score method to get accuracy of model
score_train = logisticRegr.score(X_train, Y_train)
logisticRegr.fit(Rpca_train, Y_train)
# Use score method to get accuracy of model
score_train_pc = logisticRegr.score(Rpca_train, Y_train)
prediction_train_pc = logisticRegr.predict(Rpca_train)
cm_train_pc = metrics.confusion_matrix(Y_train, prediction_train)
plt.figure(figsize = (9,3))
plot_confusion_matrix(cm_train_pc, ["cat", "dog"], normalize = True)
plt.show()




# Use score method to get accuracy of model
prediction_test = logisticRegr.predict(X_test)
prediction_test_pc = logisticRegr.predict(Rpca_test)
score_test_pc = logisticRegr.score(Rpca_test, Y_test)
score_test = logisticRegr.score(X_test, Y_test)
cm_test = metrics.confusion_matrix(Y_test, prediction_test)
cm_test_pc = metrics.confusion_matrix(Y_test, prediction_test_pc)
plt.figure(figsize = (9,3))
#plot_confusion_matrix(cm_test, ["cat", "dog"], normalize = True)
plt.figure(figsize = (9,3))
plot_confusion_matrix(cm_test_pc, ["cat", "dog"], normalize = True)
plt.show()

#print(metrics.classification_report(Y_test, prediction_test_pc, target_names = ["cat", "dog"]))
#print(metrics.classification_report(Y_test, prediction_test, target_names = ["cat", "dog"]))

print("score_test_pc is", score_test_pc)

plt.figure(figsize=(9,3))
YY_predict_pc = np.zeros(np.asarray(prediction_test_pc).shape)
YY_predict_pc = YY_predict_pc.tolist()
for i in range(len(Y_test)):
    if prediction_test_pc[i] == 0:
          YY_predict_pc[i] = 'cat'
    else:
        YY_predict_pc[i] = 'dog'
        

for index, (image, label) in enumerate(zip(Rpca_test[0:5], YY_predict_pc[0:5])):
    ax = plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (64,64)), cmap=plt.cm.gray)
    plt.title('%s\n' % label, fontsize = 15)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()



YY_predict = np.zeros(np.asarray(prediction_test).shape)
YY_predict = YY_predict.tolist()
for i in range(len(Y_test)):
    if prediction_test[i] == 0:
          YY_predict[i] = 'cat'
    else:
        YY_predict[i] = 'dog'

plt.figure(figsize=(9,3))
for index, (image, label) in enumerate(zip(X_test[0:5], YY_predict[0:5])):
    ax = plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (64,64)), cmap=plt.cm.gray)
    plt.title('%s\n' % label, fontsize = 15)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()

