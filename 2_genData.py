import numpy as np

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
    X_Y = np.concatenate((X, Y), axis=1)
    xp = int(num_data_points*train_split)
    for i in range(5):
        fname_x = prefix + '-train0' + str(i+1) + '.csv'
        fname_y = prefix + '-test0' + str(i+1) + '.csv'
        np.random.shuffle(X_Y)
        np.savetxt(fname_x, X_Y[:xp], delimiter=',')
        np.savetxt(fname_y, X_Y[xp:], delimiter=',')


train_split = .8
filename = './2_que_data/communities.data'
data_X, labels_Y, reduced_X, reduced_Y = fill_missing_values(filename)
#make_data_splits(data_X, labels_Y, data_X.shape[0], train_split, prefix='CandC')
make_data_splits(data_X, labels_Y, data_X.shape[0], train_split, prefix='Reduced')

