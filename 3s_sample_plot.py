import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


fname = './3_que_data/train.csv'
train_X = np.genfromtxt(fname, delimiter = ',')
train_Y = np.genfromtxt('./3_que_data/train_labels.csv', delimiter=',')

class1 = train_X[:1000,:]
class2 = train_X[1000:,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(class1[:,0], class1[:,1], class1[:,2], c='r', marker='o')
ax.scatter(class2[:,0], class2[:,1], class2[:,2], c='b', marker='o')

'''
def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zl, zh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
'''
plt.show()