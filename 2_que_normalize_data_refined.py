import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as LM
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV

def fill_missing_values(filename):
    f = open(filename)
    data = f.readlines()
    print len(data)
    data = map(lambda x: x.strip().split(','), data)

    data_mat = np.array(data)
    labels = data_mat[:,-1]
    labels = np.asarray(labels, dtype=float)
    data_mat = data_mat[:,5:-1]

    #for analysis part only
    useful_cols = []
    for j in range(data_mat.shape[1]):
        sum = 0
        for i in range(data_mat.shape[0]):
            if data_mat[i][j] == '?':
                sum += 1
        if sum < 2:
            useful_cols.append(j)
            #print sum, 'column number : ', j
        #print sum
    #fill missing values with zeros
    for i in range(data_mat.shape[0]):
        for j in range(data_mat.shape[1]):
            if data_mat[i][j] == '?':
                data_mat[i][j] = '0'

    data_mat = np.asarray(data_mat, dtype=float)

    # print data_mat[1][96] # it contains a missig value
    #fill missing values with average of colums
    cols_avg = np.zeros(data_mat.shape[1])

    for j in range(data_mat.shape[1]):
        sum = 0
        for i in range(data_mat.shape[0]):
            sum = sum + data_mat[i][j]
        cols_avg[j] = sum/data_mat.shape[1]

    for j in range(data_mat.shape[1]):
        sum = 0
        for i in range(data_mat.shape[0]):
            if data_mat[i][j] == 0:
                data_mat[i][j] = cols_avg[j]

    #get the reduced features by removing columns having large missing values
    reduced_data = data_mat[:,useful_cols]
    reduced_labels = labels[useful_cols]
    return data_mat, labels, reduced_data, reduced_labels

def make_data_splits(X, Y, num_data_points, train_split, prefix):
    Y = np.reshape(Y,(Y.shape[0],1))
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_Y = np.concatenate((X, Y), axis=1)
    xp = int(num_data_points*train_split)
    for i in range(5):
        fname_x = prefix + '-train00' + str(i+1) + '.csv'
        fname_y = prefix + '-test00' + str(i+1) + '.csv'
        np.random.shuffle(X_Y)
        np.savetxt(fname_x, X_Y[:xp], delimiter=',')
        np.savetxt(fname_y, X_Y[xp:], delimiter=',')


train_split = .8
filename = './2_que_data/communities.data'
#data_X, labels_Y, reduced_X, reduced_Y = fill_missing_values(filename)
#make_data_splits(data_X, labels_Y, data_X.shape[0], train_split, prefix='./2_que_data/CandC')
#make_data_splits(data_X, labels_Y, data_X.shape[0], train_split, prefix='./2_que_data/Reduced')


def linear_regression_over_splits(splits, prefix):
    rss = []
    coeff = []
    for cnt in range(splits):
        fname_x = prefix + '-train00' + str(cnt+1) + '.csv'
        fname_y = prefix + '-test00' + str(cnt+1) + '.csv'
        trdata = np.genfromtxt(fname_x, delimiter = ',')
        testdata = np.genfromtxt(fname_y, delimiter = ',')

        tr_X = trdata[:,:-1]
        tr_Y = trdata[:,-1]
        te_X = testdata[:,:-1]
        te_Y = testdata[:,-1]

        clf_LR = LM.LinearRegression()
        clf_LR.fit(tr_X, tr_Y)
        rss.append(np.mean((clf_LR.predict(te_X)- te_Y) ** 2))
        print 'RSS for Split ', cnt+1, ':', np.mean((clf_LR.predict(te_X)- te_Y) ** 2)
        coeff.append(clf_LR.coef_)
        #print 'Coefficients Learned : ', clf_LR.coef_
    print 'Mean RSS : ', np.mean(rss)

    return rss,coeff

def ridge_regression(splits, prefix):
    alp_val = 0
    rss_temp2 = np.zeros(100)
    rss_return = []
    alp_return = np.zeros(100)
    coefficients = []
    for alp in range(100):
        alp_val += .1
        alp_return[alp] = alp_val
        rss = []
        rss_temp = np.zeros(5)
        for cnt in range(splits):
            fname_x = prefix + '-train00' + str(cnt+1) + '.csv'
            fname_y = prefix + '-test00' + str(cnt+1) + '.csv'
            trdata = np.genfromtxt(fname_x, delimiter = ',')
            testdata = np.genfromtxt(fname_y, delimiter = ',')

            tr_X = trdata[:,:-1]
            tr_Y = trdata[:,-1]
            te_X = testdata[:,:-1]
            te_Y = testdata[:,-1]

            clf_LR = LM.Ridge(alpha=alp_val, normalize=True)
            clf_LR.fit(tr_X, tr_Y)
            rss.append(np.mean((clf_LR.predict(te_X)- te_Y) ** 2))
            #print 'RSS for Split ', cnt+1, ':', np.mean((clf_LR.predict(te_X)- te_Y) ** 2)
            #print 'Coefficients Learned : ', clf_LR.coef_
            rss_temp[cnt] = np.mean(rss)
            #print 'Mean RSS : ', np.mean(rss)
        print np.mean(rss_temp)
        rss_temp2[alp] = np.mean(rss_temp)
        coefficients.append(clf_LR.coef_)
    print np.argmin(rss_temp2)
    return rss_temp2, alp_return, coefficients

num_splits = 5

#rss_y, coeff= linear_regression_over_splits(num_splits, prefix='./2_que_data/CandC')

#import cPickle
#cPickle.dump(coeff,open('LROutput.save', 'wb'),protocol=cPickle.HIGHEST_PROTOCOL)

'''
rss_z = linear_regression_over_splits(num_splits, prefix='./2_que_data/Reduced')

from matplotlib import pylab as plt
#y=[.499,.12,.013,.0004]
x=[0,1,2,3,4]
plt.plot(x,rss_y,label='Mean RSS Using All Features')
plt.plot(x,rss_z,label='Mean RSS Using Reduced Features')
plt.axis([0, 3.5, 0, .55])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
#plt.title('Misclassification Error for Vario')
plt.show()
'''
#linear_regression_over_splits(num_splits, prefix='Reduced')
#rss_ridge, alp, coeff = ridge_regression(splits=num_splits, prefix='./2_que_data/CandC')


'''
out_ridge = (rss_ridge, coeff)
import cPickle
cPickle.dump(out_ridge,open('RidgeOutput.save', 'wb'),protocol=cPickle.HIGHEST_PROTOCOL)
'''


'''
print rss_ridge
from matplotlib import pylab as plt
#x=[0,1,2,3,4,5,6,7,8,9]
x=range(100)
plt.plot(alp,rss_ridge,label='Mean RSS Using All Features')
#plt.plot(x,rss_z,label='Mean RSS Using Reduced Features')
plt.axis([1, 12, 0.030, .037])
plt.xlabel('Lambda Values')
plt.ylabel('Mean RSS')
plt.legend()
#plt.title('Misclassification Error for Vario')
plt.show()
'''


def perform_final_ridge_regression(splits, prefix):
        rss = []
        #coeff = np.array((5,))
        for cnt in range(splits):
            fname_x = prefix + '-train00' + str(cnt+1) + '.csv'
            fname_y = prefix + '-test00' + str(cnt+1) + '.csv'
            trdata = np.genfromtxt(fname_x, delimiter = ',')
            testdata = np.genfromtxt(fname_y, delimiter = ',')

            tr_X = trdata[:,:-1]
            tr_Y = trdata[:,-1]
            te_X = testdata[:,:-1]
            te_Y = testdata[:,-1]

            clf_LR = LM.Ridge(alpha=1.64, normalize=True)
            clf_LR.fit(tr_X, tr_Y)
            rss.append(np.mean((clf_LR.predict(te_X)- te_Y) ** 2))
            #print 'RSS for Split ', cnt+1, ':', np.mean((clf_LR.predict(te_X)- te_Y) ** 2)
            print 'Coefficients Learned : ', clf_LR.coef_.shape

            print 'Mean RSS : ', np.mean(rss)

perform_final_ridge_regression(splits=num_splits, prefix='./2_que_data/CandC')