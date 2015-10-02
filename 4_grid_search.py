import cPickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def get_data():
    img_types = ['coast', 'forest', 'insidecity', 'mountain']
    folders = ['Test', 'Train']

    load_train = []
    load_test = []

    for img_type in img_types:
        pickl_file_name = img_type + '_' + 'Train' + '.save'
        f1 = open('./4_que_data/'+pickl_file_name, 'rb')
        pickl_file_name = img_type + '_' + 'Test' + '.save'
        f2 = open('./4_que_data/'+pickl_file_name, 'rb')
        load_train.append(np.asarray(cPickle.load(f1)))
        load_test.append(np.asarray(cPickle.load(f2)))
        f2.close()
        f1.close()

    temp_i = 0
    trainData_XpY = []
    for arr in load_train:
        label = temp_i*(np.ones((arr.shape[0],1)))
        arr = np.concatenate((arr,label), axis=1)
        trainData_XpY.append(arr)
        temp_i += 1

    temp_i = 0
    testData_XpY = []
    for arr in load_test:
        label = temp_i*(np.ones((arr.shape[0],1)))
        arr = np.concatenate((arr,label), axis=1)
        testData_XpY.append(arr)
        temp_i += 1

    train_full = np.concatenate((trainData_XpY[0], trainData_XpY[1]), axis=0)
    train_full = np.concatenate((train_full, trainData_XpY[2]), axis=0)
    train_full = np.concatenate((train_full, trainData_XpY[3]), axis=0)

    test_full = np.concatenate((testData_XpY[0], testData_XpY[1]), axis=0)
    test_full = np.concatenate((test_full, testData_XpY[2]), axis=0)
    test_full = np.concatenate((test_full, testData_XpY[3]), axis=0)

    return train_full, test_full

def get_norm_nFoldData(trainXY, testXY):
    trainX = trainXY[:,:-1]
    trainY = trainXY[:,-1]
    testX = testXY[:,:-1]
    testY = testXY[:,-1]

    #standardise only x values not labels
    scaler = StandardScaler()
    scaler.fit(trainX)
    trainX = scaler.transform(trainX)

    scaler.fit(testX)
    testX = scaler.transform(testX)

    trainY = trainY.reshape((trainY.shape[0],1))
    testY = testY.reshape((testY.shape[0],1))
    train_X_Y = np.concatenate((trainX,trainY),axis=1)
    test_X_Y = np.concatenate((testX,testY),axis=1)

    folds_tr = []
    folds_te = []
    nfolds = 5
    for i in range(nfolds):
        xp = int(train_X_Y.shape[0]*.8)
        np.random.shuffle(train_X_Y)
        folds_tr.append(train_X_Y[:xp,:])
        folds_te.append(train_X_Y[xp:,:])
    return folds_tr, folds_te

def train_svm(folds_tr, folds_te, clf):
    for f_trn, f_tes in zip(folds_tr, folds_te):
        clf.fit(f_trn[:,:-1], f_trn[:,-1])
        cnt = 0
        for tes in f_tes:
            print clf.predict(tes[:-1])
            if clf.predict(tes[:-1]) != tes[-1]:
                cnt += 1
    print 1-cnt/float(202)



np.set_printoptions(suppress=True)

train_dataXY, test_dataXY = get_data()
folds_tr, folds_te = get_norm_nFoldData(train_dataXY, test_dataXY)

clf = SVC(kernel='linear', C=.072)#, gamma=.07)
print len(folds_tr[0]), len(folds_te[0])
train_svm(folds_tr, folds_te, clf)

import cPickle
cPickle.dump(clf.coef_, open('Linear_Kernel.save','wb'),protocol=cPickle.HIGHEST_PROTOCOL)

'''
trainX = train_dataXY[:,:-1]
trainY = train_dataXY[:,-1]
testX = test_dataXY[:,:-1]
testY = test_dataXY[:,-1]

#standardise only x values not labels
scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)

scaler.fit(testX)
testX = scaler.transform(testX)

#[.01,.02,.03,.04,.05, .06,.07,.08,.09,.10]
#[.01,.02,.03,.04,.05,.06,.07,.08]
c = [ .05 + .002*i for i in range(100)]
g = [.01+.001*i for i in range(10)]
#c =[1.1,1.2,1.3,1.4]
from sklearn import svm, grid_search, datasets
parameters = {'C':c}#, 'gamma': g}
svr = svm.SVC(kernel='linear')
clf = grid_search.GridSearchCV(svr, parameters, cv=5)
clf.fit(trainX, trainY)
print clf.best_params_

#np.random.shuffle(test_dataXY)
#print test_X_Y.shape
'''

