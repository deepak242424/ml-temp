import numpy as np
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

def fill_missing_values(filename):
    f = open(filename)
    data = f.readlines()
    #print len(data)
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

def ridge_regression(splits,tr_X,tr_Y):
        alphas = np.array([10000,10,0.1,0.01,0.001,0.0001,0])
        model = Ridge()
        grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas), cv=splits)
        grid.fit(tr_X, tr_Y)
        print(grid.best_estimator_.alpha)
        print grid.best_score_

filename = './2_que_data/communities.data'
data_X, labels_Y, reduced_X, reduced_Y = fill_missing_values(filename)
scaler = StandardScaler()
scaler.fit(data_X)
data_X = scaler.transform(data_X)
num_splits = 5

ridge_regression(num_splits,data_X,labels_Y)



