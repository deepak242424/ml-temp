import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA
import sklearn.linear_model as LM

fname = './3_que_data/train.csv'
train_X = np.genfromtxt(fname, delimiter = ',')
train_Y = np.genfromtxt('./3_que_data/train_labels.csv', delimiter=',')
'''
class1 = train_X[:1000,:]
class2 = train_X[1000:,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(class1[:,0], class1[:,1], class1[:,2], c='r', marker='o')
ax.scatter(class2[:,0], class2[:,1], class2[:,2], c='b', marker='o')
#plt.show()
'''




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
    return 1-np.mean(np.abs(final_pred-test_Y))

plt.plot(train_X[:1000], [10]*1000, 'ro')
plt.plot(train_X[1000:], [10]*1000, 'bo')
plt.plot([clf_LR.coef_]*20,range(20),'g')
plt.axis([-6, 6, 0, 20])
plt.show()

print clf_LR.coef_
print get_accuracy(predicted)






