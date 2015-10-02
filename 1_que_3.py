__author__ = 'deepak'

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def get_class_frm_score(predicted):
    final_pred = np.zeros(predicted.shape[0])
    cnt = 0
    for val in predicted:
        if abs(val-0) < abs(val-1):
            final_pred[cnt] = 0
        else:
            final_pred[cnt] = 1
        cnt+= 1
    return final_pred

def get_misclassification_error(truth, predic):
    cnt = 0
    for tru,pre in zip(truth,predic):
        if tru!=pre:
            cnt += 1
    return cnt/float(truth.shape[0])

# Mean of first class = 0
mean_1 = np.zeros((10))

# Mean of second class = 1
mean_2 = np.ones((10))

# Define covarince matrix with diagonal elememts as mean of class 2
covar = .1*np.ones((10,10))
np.fill_diagonal(covar,1)

# Generate 1000 samples for each class
class_0 = np.random.multivariate_normal(mean_1, covar, 1000)
class_1 = np.random.multivariate_normal(mean_2, covar, 1000)
'''
# Shuffle both the classes
np.random.shuffle(class_0)
np.random.shuffle(class_1)

# Take 40% X points as test set and 60% points as train set
class_0_test = class_0[:400]
class_0_train = class_0[400:]
class_1_test = class_1[:400]
class_1_train = class_1[400:]
'''
Y_0 = np.zeros(1000).reshape((1000,1))
Y_1 = np.ones(1000).reshape((1000,1))

X_n_Y_0 = np.concatenate((class_0, Y_0), axis=1)
X_n_Y_1 = np.concatenate((class_1, Y_1), axis=1)

#np.random.shuffle(X_n_Y_0)
#np.random.shuffle(X_n_Y_1)

class_0_test = X_n_Y_0[:400]
class_1_test = X_n_Y_1[:400]
class_0_train = X_n_Y_0[400:]
class_1_train = X_n_Y_1[400:]

test_X_Y = np.concatenate((class_0_test,class_1_test), axis=0)
train_X_Y = np.concatenate((class_0_train,class_1_train), axis=0)
#np.random.shuffle(test_X_Y)
#np.random.shuffle(train_X_Y)

test_X = test_X_Y[:, :10]
train_X = train_X_Y[:, :10]
test_Y = test_X_Y[:, -1]
train_Y = train_X_Y[:, -1]

error = []
for k in xrange(1,200,1):
    clf_KNN = KNeighborsClassifier(n_neighbors=k)
    clf_KNN.fit(train_X, train_Y)
    predicted_labels = get_class_frm_score(clf_KNN.predict(test_X))
    error.append(get_misclassification_error(test_Y, predicted_labels))
    print get_misclassification_error(test_Y, predicted_labels)

plt.plot(xrange(1,200,1),error,label='Misclassification Error')
plt.axis([0, 200, .10, .25])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
#plt.title('Misclassification Error for Vario')
plt.show()
#print np.mean((clf_KNN.predict(test_X)-test_Y) ** 2)

print np.argmin(np.asarray(error))
print np.mean(np.asarray(error))