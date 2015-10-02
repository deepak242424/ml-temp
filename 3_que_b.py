from sklearn.lda import LDA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import sklearn.linear_model as LM
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

fname = './3_que_data/train.csv'
train_X = np.genfromtxt(fname, delimiter = ',')
train_Y = np.genfromtxt('./3_que_data/train_labels.csv', delimiter=',')

test_X = np.genfromtxt('./3_que_data/test.csv', delimiter=',')
test_Y = np.genfromtxt('./3_que_data/test_labels.csv', delimiter=',')

clf = LDA()
clf.fit(train_X, train_Y)

train_X_transformed = clf.transform(train_X)
train_X_transformed = train_X_transformed.flatten()
print train_X_transformed.shape
print clf.coef_

plt.plot(train_X_transformed[:1000], [10]*1000, 'ro', label='Class 1')
plt.plot(train_X_transformed[1000:], [10]*1000, 'bo', label ='Class 2')
plt.plot([0]*21,range(21),'g', label = 'Decision Boundary')
plt.axis([-6, 6, 0, 20])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
print precision_recall_fscore_support(test_Y, clf.predict(test_X), labels=[1,2])

