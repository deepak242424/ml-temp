import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA
import sklearn.linear_model as LM
from sklearn.metrics import precision_recall_fscore_support
fname = './3_que_data/train.csv'
train_X = np.genfromtxt(fname, delimiter = ',')
train_Y = np.genfromtxt('./3_que_data/train_labels.csv', delimiter=',')

class1 = train_X[:1000,:]
class2 = train_X[1000:,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(class1[:,0], class1[:,1], class1[:,2], c='r', marker='o', label='Class1')
ax.scatter(class2[:,0], class2[:,1], class2[:,2], c='b', marker='o', label='Class2')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
#plt.zlabel('Z-axis')
plt.legend()
plt.title('Data points before PCA & LDA')
plt.show()
#train_Y = train_Y.reshape((train_Y.shape[0],1))
test_X = np.genfromtxt('./3_que_data/test.csv', delimiter=',')
test_Y = np.genfromtxt('./3_que_data/test_labels.csv', delimiter=',')

#trainXY = np.concatenate((train_X,train_Y), axis=1)
#np.random.shuffle(trainXY)

#train_X = trainXY[:,:3]
#train_Y = trainXY[:,3]

pca = PCA(n_components=1)
pca.fit(train_X)
train_X = pca.transform(train_X)
test_X = pca.transform(test_X)

clf_LR = LM.LinearRegression()
clf_LR.fit(train_X, train_Y)
predicted  = clf_LR.predict(test_X)

def get_accuracy(predicted):
    final_pred = np.zeros(predicted.shape[0])
    cnt = 0
    for val in predicted:
        if abs(val-1) < abs(val-2):
            final_pred[cnt] = 1
        else:
            final_pred[cnt] = 2
        cnt += 1
    return 1-np.mean(np.abs(final_pred-test_Y)),final_pred
'''
plt.plot(train_X[:1000], [10]*1000, label='Class 1', color='r', marker='o')
plt.plot(train_X[1000:], [10]*1000, label='Class 2', color='b', marker='o')
plt.plot([clf_LR.coef_]*21,range(21),'g', label='Decision Boundary')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Projection of data in new subspace after PCA')
plt.axis([-7, 7, 0, 20])
plt.show()
'''
print clf_LR.coef_

acc = get_accuracy(predicted)[0]
predicted_classes = get_accuracy(predicted)[1]
print precision_recall_fscore_support(test_Y, predicted_classes, labels=[1,2])
