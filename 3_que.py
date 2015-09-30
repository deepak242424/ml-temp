import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA
import sklearn.linear_model as LM

fname = './3_que_data/train.csv'
train_X = np.genfromtxt(fname, delimiter = ',')
train_Y = np.genfromtxt('./3_que_data/train_labels.csv', delimiter=',')
train_Y = train_Y.reshape((train_Y.shape[0],1))
test_X = np.genfromtxt('./3_que_data/test.csv', delimiter=',')
test_Y = np.genfromtxt('./3_que_data/test_labels.csv', delimiter=',')

trainXY = np.concatenate((train_X,train_Y), axis=1)
np.random.shuffle(trainXY)

train_X = trainXY[:,:3]
train_Y = trainXY[:,3]

# 1.now have to decide how to use test samples
# as dimensionality of test sample is 3
# but i am performing regression on one samlpe

# 2. also see if my method is correct or not for
# choosing one dimension
pca = PCA(n_components=1)
pca.fit(train_X)
train_X = pca.transform(train_X)

clf_LR = LM.LinearRegression()
clf_LR.fit(train_X, train_Y)

print train_Y
print train_X.shape, type(train_X)






'''
class1 = train_X[:1000,:]
class2 = train_X[1000:,:]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(class1[0,:], class1[1,:], class1[2,:],
        'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(class2[0,:], class2[1,:], class2[2,:],
        '^', markersize=8, alpha=0.5, color='red', label='class2')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')

plt.show()
'''