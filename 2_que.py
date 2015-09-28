import numpy as np

f = open('./2_que_data/communities.data')
data = f.readlines()
print len(data)
data = map(lambda x: x.strip().split(','), data)

data_mat = np.array(data)
data_mat = data_mat[:,5:]

count = 1
'''
for item in np.nditer(data_mat, op_flags=['readwrite']):
    if item == '?':
        item = '0'
    print item
    item = float(item)
'''
#fill missing values with zeros
for i in range(data_mat.shape[0]):
    for j in range(data_mat.shape[1]):
        if data_mat[i][j] == '?':
            data_mat[i][j] = '0'

data_mat = np.asarray(data_mat, dtype=float)

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
            
print data_mat[0]




