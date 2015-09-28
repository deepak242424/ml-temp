import numpy as np
import sklearn.linear_model as LM

def fill_missing_values(filename):
    f = open(filename)
    data = f.readlines()
    print len(data)
    data = map(lambda x: x.strip().split(','), data)

    data_mat = np.array(data)
    labels = data_mat[:,-1]
    labels = np.asarray(labels, dtype=float)
    data_mat = data_mat[:,5:-1]

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
    return data_mat, labels

def make_data_splits(X, Y, num_data_points, train_split):
    Y = np.reshape(Y,(Y.shape[0],1))
    X_Y = np.concatenate((X, Y), axis=1)
    xp = int(num_data_points*train_split)
    for i in range(5):
        fname_x = 'CandC-train0' + str(i+1) + '.csv'
        fname_y = 'CandC-test0' + str(i+1) + '.csv'
        np.random.shuffle(X_Y)
        np.savetxt(fname_x, X_Y[:xp], delimiter=',')
        np.savetxt(fname_y, X_Y[xp:], delimiter=',')

def linear_regression_over_splits(splits):
    rss = []
    for cnt in range(splits):
        fname_x = 'CandC-train0' + str(cnt+1) + '.csv'
        fname_y = 'CandC-test0' + str(cnt+1) + '.csv'
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
        #print 'Coefficients Learned : ', clf_LR.coef_
    print 'Mean RSS : ', np.mean(rss)

def ridge_regression(splits):
    for alp_val in range(10):
        alp_val *= 100
        rss = []
        for cnt in range(splits):
            fname_x = 'CandC-train0' + str(cnt+1) + '.csv'
            fname_y = 'CandC-test0' + str(cnt+1) + '.csv'
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
        print 'Mean RSS : ', np.mean(rss)


filename = './2_que_data/communities.data'
data_X, labels_Y = fill_missing_values(filename)
train_split = .8
num_splits = 5
make_data_splits(data_X, labels_Y, data_X.shape[0], train_split)

#linear_regression_over_splits(num_splits)
#ridge_regression(splits=num_splits)




